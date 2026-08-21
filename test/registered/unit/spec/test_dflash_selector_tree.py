from __future__ import annotations

import pytest
import torch

from sglang.srt.speculative.dflash_tree import (
    build_dflash_selector_tree_reference,
)


def _branching_lattice(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    candidate_ids = torch.tensor(
        [[[10, 11], [20, 21]]], dtype=torch.int64, device=device
    )
    edge_scores = torch.tensor(
        [[[[4.0, 3.0], [4.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]]]],
        dtype=torch.float32,
        device=device,
    )
    return candidate_ids, edge_scores


def test_reference_tree_is_prefix_closed_and_recovers_branch() -> None:
    candidate_ids, edge_scores = _branching_lattice()
    tree = build_dflash_selector_tree_reference(candidate_ids, edge_scores, budget=4)

    assert tree.draft_tokens.tolist() == [[10, 20, 11, 21]]
    assert tree.parent.tolist() == [[0, 1, 0, 3]]
    assert tree.depth.tolist() == [[1, 2, 1, 2]]
    assert tree.candidate_index.tolist() == [[0, 0, 1, 1]]
    assert tree.selected_index.tolist() == [[0, 2, 1, 7]]
    assert tree.parent_list.tolist() == [[0, 0, 2, 1, 7]]
    assert all(
        parent < node
        for node, parent in enumerate(tree.parent[0].tolist(), start=1)
    )


def test_reference_tree_ties_are_deterministic_by_token_id() -> None:
    candidate_ids = torch.tensor([[[9, 3, 5]]], dtype=torch.int64)
    scores = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    first = build_dflash_selector_tree_reference(candidate_ids, scores, budget=3)
    second = build_dflash_selector_tree_reference(candidate_ids, scores, budget=3)
    assert first.draft_tokens.tolist() == [[3, 5, 9]]
    assert torch.equal(first.draft_tokens, second.draft_tokens)
    assert torch.equal(first.parent, second.parent)


@pytest.mark.parametrize("budget", [0, -1])
def test_reference_tree_rejects_nonpositive_budget(budget: int) -> None:
    candidate_ids, edge_scores = _branching_lattice()
    with pytest.raises(ValueError, match="positive"):
        build_dflash_selector_tree_reference(candidate_ids, edge_scores, budget)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("budget", [1, 3, 7])
def test_triton_tree_matches_reference_random_lattice(budget: int) -> None:
    from sglang.kernels.ops.speculative.dflash_tree import (
        build_dflash_selector_tree_triton,
    )

    generator = torch.Generator().manual_seed(20260821 + budget)
    candidate_ids = torch.stack(
        [torch.randperm(512, generator=generator)[: 7 * 16].view(7, 16) for _ in range(3)]
    ).to(torch.int64)
    edge_scores = torch.randn((3, 7, 16, 16), generator=generator)
    expected = build_dflash_selector_tree_reference(
        candidate_ids, edge_scores, budget=budget
    )
    actual = build_dflash_selector_tree_triton(
        candidate_ids.cuda(), edge_scores.cuda(), budget=budget
    )

    for field in (
        "draft_tokens",
        "parent_list",
        "selected_index",
        "parent",
        "depth",
        "candidate_index",
    ):
        assert torch.equal(getattr(actual, field).cpu(), getattr(expected, field))
    torch.testing.assert_close(
        actual.cumulative_log_probability.cpu(),
        expected.cumulative_log_probability,
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_tree_matches_reference_score_ties() -> None:
    from sglang.kernels.ops.speculative.dflash_tree import (
        build_dflash_selector_tree_triton,
    )

    candidate_ids = torch.tensor([[[9, 3, 5]]], dtype=torch.int64)
    scores = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    expected = build_dflash_selector_tree_reference(candidate_ids, scores, budget=3)
    actual = build_dflash_selector_tree_triton(
        candidate_ids.cuda(), scores.cuda(), budget=3
    )
    assert torch.equal(actual.draft_tokens.cpu(), expected.draft_tokens)
    assert torch.equal(actual.selected_index.cpu(), expected.selected_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_tree_paths_expand_predecessor_chains() -> None:
    from sglang.kernels.ops.speculative.dflash_tree import (
        _build_dflash_tree_paths_triton_unchecked,
    )

    parent = torch.tensor(
        [[0, 1, 0, 3, 4, 0, 6]], dtype=torch.int32, device="cuda"
    )
    depth = torch.tensor(
        [[1, 2, 1, 2, 3, 1, 2]], dtype=torch.int32, device="cuda"
    )
    paths = torch.empty((1, 8, 8), dtype=torch.int64, device="cuda")
    _build_dflash_tree_paths_triton_unchecked(
        parent=parent,
        depth=depth,
        paths_out=paths,
    )

    assert paths.cpu().tolist() == [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 3, 4, 0, 0, 0, 0, 0],
            [0, 3, 4, 5, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 0, 0],
            [0, 6, 7, 0, 0, 0, 0, 0],
        ]
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_tree_paths_match_cpu_chase_for_random_selector_tree() -> None:
    from sglang.kernels.ops.speculative.dflash_tree import (
        _build_dflash_tree_paths_triton_unchecked,
        build_dflash_selector_tree_triton,
    )

    generator = torch.Generator().manual_seed(20260821)
    candidate_ids = torch.stack(
        [torch.randperm(2048, generator=generator)[: 7 * 16].view(7, 16)]
        * 4
    ).to(torch.int64)
    edge_scores = torch.randn((4, 7, 16, 16), generator=generator)
    tree = build_dflash_selector_tree_triton(
        candidate_ids.cuda(), edge_scores.cuda(), budget=7
    )
    paths = torch.empty((4, 8, 8), dtype=torch.int64, device="cuda")
    _build_dflash_tree_paths_triton_unchecked(
        parent=tree.parent,
        depth=tree.depth,
        paths_out=paths,
    )

    expected = torch.zeros((4, 8, 8), dtype=torch.int64)
    parent_cpu = tree.parent.cpu()
    depth_cpu = tree.depth.cpu()
    for batch in range(4):
        for node in range(1, 8):
            current = node
            while current:
                expected[batch, node, depth_cpu[batch, current - 1]] = current
                current = int(parent_cpu[batch, current - 1])
    assert torch.equal(paths.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selector_tree_encoding_drives_existing_sglang_verifier() -> None:
    from sglang.kernels.ops.speculative.dflash_tree import (
        build_dflash_selector_tree_triton,
    )
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode,
        build_tree_kernel_efficient,
        verify_tree_greedy_func,
    )

    candidate_ids, edge_scores = _branching_lattice("cuda")
    tree = build_dflash_selector_tree_triton(candidate_ids, edge_scores, budget=4)
    (
        mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        candidates,
    ) = build_tree_kernel_efficient(
        bonus_tokens=torch.tensor([99], dtype=torch.int64, device="cuda"),
        parent_list=tree.parent_list,
        top_scores_index=tree.selected_index,
        draft_tokens=tree.draft_tokens,
        seq_lens=torch.tensor([5], dtype=torch.int64, device="cuda"),
        seq_lens_sum=5,
        topk=2,
        spec_steps=2,
        num_verify_tokens=5,
        tree_mask_mode=TreeMaskMode.FULL_MASK,
    )

    assert candidates.view(1, 5).cpu().tolist() == [[99, 10, 20, 11, 21]]
    assert positions.view(1, 5).cpu().tolist() == [[5, 6, 7, 6, 7]]
    tree_mask = mask.view(5, 10)[:, 5:].cpu()
    assert tree_mask.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, False, False, True, False],
        [True, False, False, True, True],
    ]

    target_predict = torch.tensor(
        [[11, 0, 0, 21, 77]], dtype=torch.int64, device="cuda"
    )
    predicts = torch.zeros((5,), dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
    accepted = torch.empty((1,), dtype=torch.int32, device="cuda")
    verify_tree_greedy_func(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted,
        candidates=candidates.view(1, 5),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_predict=target_predict,
        topk=2,
    )
    assert accepted.cpu().tolist() == [2]
    assert accept_index.cpu().tolist() == [[0, 3, 4]]
    assert predicts.cpu().tolist() == [11, 0, 0, 21, 77]
