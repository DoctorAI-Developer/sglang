"""Best-first DFlash2 selector-tree construction.

The DFlash2 selector is a first-order Markov lattice: the score row for a
token at depth ``d`` depends on the local candidate selected at ``d - 1``.
This is deliberately different from EAGLE/DDTree's independent-position
topology builder.  The kernel below allocates a prefix-closed tree by exact
cumulative log probability and emits the synthetic EAGLE parent encoding used
by SGLang's existing attention-mask and tree-verification kernels.

The optimized path is intentionally small and bounded.  Qwen3.8-27B's draft
uses depth 7, top-k 16, and the first production experiment uses seven
non-root nodes (eight target verification rows including the root).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class DFlashSelectorTree:
    """Tree tensors consumed by ``build_tree_kernel_efficient``.

    ``draft_tokens`` excludes the verified root/bonus token.  ``parent_list``
    and ``selected_index`` form a synthetic EAGLE encoding: for node ``i``
    (one based), ``selected_index // top_k`` is its parent node position and
    ``parent_list[i]`` stores that node's own selected index.
    """

    draft_tokens: torch.Tensor
    parent_list: torch.Tensor
    selected_index: torch.Tensor
    parent: torch.Tensor
    depth: torch.Tensor
    candidate_index: torch.Tensor
    cumulative_log_probability: torch.Tensor


def build_dflash_selector_tree_reference(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    budget: int,
) -> DFlashSelectorTree:
    """Deterministic CPU reference for the best-first Markov selector tree."""
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, depth, top_k]")
    batch, depth_limit, top_k = map(int, candidate_ids.shape)
    if edge_scores.shape != (batch, depth_limit, top_k, top_k):
        raise ValueError(
            "edge_scores must have shape [batch, depth, top_k, top_k], got "
            f"{tuple(edge_scores.shape)}"
        )
    if candidate_ids.dtype != torch.int64:
        raise ValueError("candidate_ids must use torch.int64")
    if budget <= 0:
        raise ValueError("tree budget must be positive")
    if depth_limit <= 0 or top_k <= 0:
        raise ValueError("candidate lattice dimensions must be positive")
    if budget > depth_limit * top_k:
        raise ValueError(
            "tree budget exceeds the bounded selector lattice: "
            f"budget={budget}, depth={depth_limit}, top_k={top_k}"
        )
    if not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("edge scores must be finite")

    ids = candidate_ids.detach().cpu()
    log_probs = torch.log_softmax(edge_scores.detach().float(), dim=-1).cpu()
    tokens = torch.empty((batch, budget), dtype=torch.int64)
    parents = torch.empty((batch, budget), dtype=torch.int32)
    depths = torch.empty((batch, budget), dtype=torch.int32)
    child_indices = torch.empty((batch, budget), dtype=torch.int32)
    cumulative = torch.empty((batch, budget), dtype=torch.float32)
    selected = torch.empty((batch, budget), dtype=torch.int64)
    parent_list = torch.zeros((batch, budget + 1), dtype=torch.int64)

    for b in range(batch):
        # (cumulative log p, next depth, parent node, predecessor candidate,
        #  child candidate).  Rebuilding this tiny frontier makes the ordering
        # explicit and mirrors the fused GPU kernel exactly.
        frontier: list[tuple[float, int, int, int, int]] = []
        for child in range(top_k):
            frontier.append((float(log_probs[b, 0, 0, child]), 0, 0, 0, child))

        used: set[tuple[int, int]] = set()
        for out_idx in range(budget):
            eligible = [entry for entry in frontier if (entry[2], entry[4]) not in used]
            if not eligible:
                raise RuntimeError("selector tree frontier was exhausted")
            score, depth_index, parent, predecessor, child = min(
                eligible,
                key=lambda entry: (
                    -entry[0],
                    entry[1],
                    entry[2],
                    int(ids[b, entry[1], entry[4]]),
                    entry[3],
                    entry[4],
                ),
            )
            used.add((parent, child))
            node = out_idx + 1
            token = int(ids[b, depth_index, child])
            synthetic = parent * top_k + child
            tokens[b, out_idx] = token
            parents[b, out_idx] = parent
            depths[b, out_idx] = depth_index + 1
            child_indices[b, out_idx] = child
            cumulative[b, out_idx] = score
            selected[b, out_idx] = synthetic
            parent_list[b, node] = synthetic

            next_depth = depth_index + 1
            if next_depth < depth_limit:
                for next_child in range(top_k):
                    frontier.append(
                        (
                            score
                            + float(log_probs[b, next_depth, child, next_child]),
                            next_depth,
                            node,
                            child,
                            next_child,
                        )
                    )

    device = candidate_ids.device
    return DFlashSelectorTree(
        draft_tokens=tokens.to(device),
        parent_list=parent_list.to(device),
        selected_index=selected.to(device),
        parent=parents.to(device),
        depth=depths.to(device),
        candidate_index=child_indices.to(device),
        cumulative_log_probability=cumulative.to(device),
    )


@triton.jit
def _dflash_selector_tree_kernel(
    candidate_ids_ptr,
    edge_scores_ptr,
    draft_tokens_ptr,
    parent_list_ptr,
    selected_index_ptr,
    parent_ptr,
    depth_ptr,
    candidate_index_ptr,
    cumulative_ptr,
    candidate_batch_stride: tl.constexpr,
    candidate_depth_stride: tl.constexpr,
    candidate_k_stride: tl.constexpr,
    score_batch_stride: tl.constexpr,
    score_depth_stride: tl.constexpr,
    score_pred_stride: tl.constexpr,
    score_child_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    parent_list_batch_stride: tl.constexpr,
    depth_limit: tl.constexpr,
    top_k: tl.constexpr,
    budget: tl.constexpr,
    frontier_block: tl.constexpr,
):
    """One program per request; all frontier state remains in registers."""
    batch_idx = tl.program_id(0)
    edge = tl.arange(0, frontier_block)
    parent_slot = edge // top_k
    child = edge % top_k
    edge_in_bounds = edge < (budget + 1) * top_k

    out_base = batch_idx * output_batch_stride
    candidate_base = batch_idx * candidate_batch_stride
    score_base = batch_idx * score_batch_stride
    parent_list_base = batch_idx * parent_list_batch_stride
    tl.store(parent_list_ptr + parent_list_base, 0)

    for iteration in range(budget):
        # Slot zero is the synthetic root.  Slot s>0 becomes valid after node s
        # has been admitted (node tensors use zero-based output index s-1).
        admitted_parent = (parent_slot == 0) | (
            (parent_slot > 0) & (parent_slot <= iteration)
        )
        parent_out_idx = tl.maximum(parent_slot - 1, 0)
        parent_depth = tl.load(
            depth_ptr + out_base + parent_out_idx,
            mask=(parent_slot > 0) & (parent_slot <= iteration),
            other=0,
        )
        next_depth = tl.where(parent_slot == 0, 0, parent_depth)
        predecessor = tl.load(
            candidate_index_ptr + out_base + parent_out_idx,
            mask=(parent_slot > 0) & (parent_slot <= iteration),
            other=0,
        )
        parent_cumulative = tl.load(
            cumulative_ptr + out_base + parent_out_idx,
            mask=(parent_slot > 0) & (parent_slot <= iteration),
            other=0.0,
        )
        valid = edge_in_bounds & admitted_parent & (next_depth < depth_limit)

        score_addr = (
            edge_scores_ptr
            + score_base
            + next_depth * score_depth_stride
            + predecessor * score_pred_stride
            + child * score_child_stride
        )
        edge_score = tl.load(score_addr, mask=valid, other=-float("inf")).to(tl.float32)

        # Row logsumexp.  Every edge in a parent slot shares the same row, so
        # this is redundant arithmetic but avoids global scratch, allocation,
        # synchronization, and graph-unstable pointers for a tiny K=16 lattice.
        row_max = tl.full((frontier_block,), -float("inf"), tl.float32)
        for k in range(top_k):
            value = tl.load(
                edge_scores_ptr
                + score_base
                + next_depth * score_depth_stride
                + predecessor * score_pred_stride
                + k * score_child_stride,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
            row_max = tl.maximum(row_max, value)
        row_sum = tl.zeros((frontier_block,), tl.float32)
        for k in range(top_k):
            value = tl.load(
                edge_scores_ptr
                + score_base
                + next_depth * score_depth_stride
                + predecessor * score_pred_stride
                + k * score_child_stride,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
            row_sum += tl.where(valid, tl.exp(value - row_max), 0.0)
        cumulative_score = parent_cumulative + edge_score - row_max - tl.log(row_sum)

        # An edge is uniquely identified by (parent node, local child).  Mask
        # every edge already admitted in an earlier iteration.
        already_used = tl.zeros((frontier_block,), tl.int1)
        for previous in range(iteration):
            previous_parent = tl.load(parent_ptr + out_base + previous)
            previous_child = tl.load(candidate_index_ptr + out_base + previous)
            already_used |= (parent_slot == previous_parent) & (child == previous_child)
        valid &= ~already_used

        candidate_token = tl.load(
            candidate_ids_ptr
            + candidate_base
            + next_depth * candidate_depth_stride
            + child * candidate_k_stride,
            mask=valid,
            other=2**30,
        ).to(tl.int32)

        # Match the CPU/oracle heap ordering for deterministic score ties:
        # cumulative probability desc, depth, parent, token id, child index.
        best_score = tl.max(tl.where(valid, cumulative_score, -float("inf")))
        tied = valid & (cumulative_score == best_score)
        best_depth = tl.min(tl.where(tied, next_depth, 2**30))
        tied &= next_depth == best_depth
        best_parent = tl.min(tl.where(tied, parent_slot, 2**30))
        tied &= parent_slot == best_parent
        best_token = tl.min(tl.where(tied, candidate_token, 2**30))
        tied &= candidate_token == best_token
        best_child = tl.min(tl.where(tied, child, 2**30))

        synthetic_index = best_parent * top_k + best_child
        tl.store(draft_tokens_ptr + out_base + iteration, best_token)
        tl.store(parent_ptr + out_base + iteration, best_parent)
        tl.store(depth_ptr + out_base + iteration, best_depth + 1)
        tl.store(candidate_index_ptr + out_base + iteration, best_child)
        tl.store(cumulative_ptr + out_base + iteration, best_score)
        tl.store(selected_index_ptr + out_base + iteration, synthetic_index)
        # Entry ``node`` describes that node's own synthetic selected index;
        # descendants divide their selected index by top_k to find this slot.
        tl.store(parent_list_ptr + parent_list_base + iteration + 1, synthetic_index)


def _build_dflash_selector_tree_triton_unchecked(
    *,
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    draft_tokens_out: torch.Tensor,
    parent_list_out: torch.Tensor,
    selected_index_out: torch.Tensor,
    parent_out: torch.Tensor,
    depth_out: torch.Tensor,
    candidate_index_out: torch.Tensor,
    cumulative_out: torch.Tensor,
) -> None:
    """Launch into caller-owned static outputs (safe inside CUDA Graph capture)."""
    batch, depth_limit, top_k = map(int, candidate_ids.shape)
    budget = int(draft_tokens_out.shape[1])
    frontier_size = (budget + 1) * top_k
    frontier_block = triton.next_power_of_2(frontier_size)
    _dflash_selector_tree_kernel[(batch,)](
        candidate_ids,
        edge_scores,
        draft_tokens_out,
        parent_list_out,
        selected_index_out,
        parent_out,
        depth_out,
        candidate_index_out,
        cumulative_out,
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        candidate_ids.stride(2),
        edge_scores.stride(0),
        edge_scores.stride(1),
        edge_scores.stride(2),
        edge_scores.stride(3),
        draft_tokens_out.stride(0),
        parent_list_out.stride(0),
        depth_limit=depth_limit,
        top_k=top_k,
        budget=budget,
        frontier_block=frontier_block,
        num_warps=4,
    )


def build_dflash_selector_tree_triton(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    budget: int,
) -> DFlashSelectorTree:
    """Validated allocating wrapper used by tests and eager execution."""
    if not candidate_ids.is_cuda or not edge_scores.is_cuda:
        raise ValueError("the Triton selector-tree builder requires CUDA tensors")
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, depth, top_k]")
    batch, depth_limit, top_k = map(int, candidate_ids.shape)
    if edge_scores.shape != (batch, depth_limit, top_k, top_k):
        raise ValueError("candidate IDs and edge scores have incompatible shapes")
    if candidate_ids.dtype != torch.int64:
        raise ValueError("candidate_ids must use torch.int64")
    if budget <= 0 or budget > depth_limit * top_k:
        raise ValueError("tree budget must be in [1, depth * top_k]")

    device = candidate_ids.device
    tokens = torch.empty((batch, budget), dtype=torch.int64, device=device)
    parent_list = torch.empty((batch, budget + 1), dtype=torch.int64, device=device)
    selected = torch.empty((batch, budget), dtype=torch.int64, device=device)
    parent = torch.empty((batch, budget), dtype=torch.int32, device=device)
    depth = torch.empty((batch, budget), dtype=torch.int32, device=device)
    child = torch.empty((batch, budget), dtype=torch.int32, device=device)
    cumulative = torch.empty((batch, budget), dtype=torch.float32, device=device)
    _build_dflash_selector_tree_triton_unchecked(
        candidate_ids=candidate_ids,
        edge_scores=edge_scores,
        draft_tokens_out=tokens,
        parent_list_out=parent_list,
        selected_index_out=selected,
        parent_out=parent,
        depth_out=depth,
        candidate_index_out=child,
        cumulative_out=cumulative,
    )
    return DFlashSelectorTree(
        draft_tokens=tokens,
        parent_list=parent_list,
        selected_index=selected,
        parent=parent,
        depth=depth,
        candidate_index=child,
        cumulative_log_probability=cumulative,
    )
