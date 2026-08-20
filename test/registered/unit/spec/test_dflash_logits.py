import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.models.dflash as dflash_module
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _grouped_conv,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    build_dflash_rejected_draft_metadata,
    build_dflash_verify_target_probs,
    compute_dflash_candidate_logprobs,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _map_reduced_target_top1,
    _missing_target_top1_ids,
    _resolve_dflash_target_token_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def test_selector_token_map_scores_only_selected_rows_and_restores_global_ids(
    monkeypatch,
):
    monkeypatch.setattr(dflash_module, "_flashinfer_top_k", None)
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, -1.0],
            [0.5, 2.0],
        ]
    )
    lm_head = SimpleNamespace(weight=weight, org_vocab_size=6)
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=2),
        draft_config=parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {"selector_rank": 2, "selector_top_k": 2},
            }
        ),
        _selector_hot_token_id=None,
        _selector_lm_head_weight=None,
    )
    model._transform_unary_logits = lambda logits: (
        DFlash2DraftModel._transform_unary_logits(model, logits)
    )
    DFlash2DraftModel.set_selector_token_map(model, torch.tensor([5, 1, 3]), lm_head)

    hidden = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    with get_parallel().override(tp_size=1):
        ids, vals = DFlash2DraftModel.compute_candidates(model, hidden)
    expected_vals, expected_local_ids = torch.topk(
        hidden @ weight[torch.tensor([1, 3, 5])].T, 2, dim=-1
    )
    expected_ids = torch.tensor([1, 3, 5])[expected_local_ids]
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(vals, expected_vals)


def test_selector_fp8_proposal_head_restores_global_ids(monkeypatch):
    """FP8 is proposal-only: its local ranking still has to leave the draft as
    original Qwen token ids, and the projection must receive BF16 activations."""
    monkeypatch.setattr(dflash_module, "_flashinfer_top_k", None)
    hot_token_id = torch.tensor([1, 3, 5])
    fp8_weight = torch.zeros((3, 2), dtype=torch.uint8)
    weight_scale = torch.ones((3, 1), dtype=torch.float32)
    projected = torch.tensor([[0.1, 3.0, 2.0], [4.0, 0.2, 1.0]])

    def fake_fp8_matmul(hidden, weight, scale):
        assert hidden.dtype == torch.bfloat16
        assert weight is fp8_weight
        assert scale is weight_scale
        return projected

    monkeypatch.setattr(
        dflash_module, "_dflash_selector_fp8_matmul", fake_fp8_matmul
    )
    lm_head = SimpleNamespace(weight=torch.randn(6, 2), org_vocab_size=6)
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=2),
        draft_config=parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {"selector_rank": 2, "selector_top_k": 2},
            }
        ),
        _selector_hot_token_id=hot_token_id,
        _selector_lm_head_weight=fp8_weight,
        _selector_lm_head_weight_scale=weight_scale,
    )
    model._transform_unary_logits = lambda logits: (
        DFlash2DraftModel._transform_unary_logits(model, logits)
    )

    with get_parallel().override(tp_size=1):
        ids, vals = DFlash2DraftModel.compute_candidates(model, torch.randn(2, 2))

    torch.testing.assert_close(ids, torch.tensor([[3, 5], [1, 5]]))
    torch.testing.assert_close(vals, torch.tensor([[3.0, 2.0], [4.0, 1.0]]))


def test_selector_fp8_shortlist_is_bf16_reranked_before_global_id_restore(
    monkeypatch,
):
    monkeypatch.setattr(dflash_module, "_flashinfer_top_k", None)
    hot_token_id = torch.tensor([1, 3, 4, 5])
    fp8_weight = torch.zeros((4, 2), dtype=torch.uint8)
    weight_scale = torch.ones((4, 1), dtype=torch.float32)

    monkeypatch.setattr(
        dflash_module,
        "_dflash_selector_fp8_matmul",
        lambda hidden, weight, scale: torch.tensor([[10.0, 9.0, 8.0, 0.0]]),
    )

    def fake_refine(hidden, full_weight, token_ids, shortlist_ids):
        torch.testing.assert_close(token_ids, hot_token_id)
        torch.testing.assert_close(shortlist_ids, torch.tensor([[0, 1, 2]]))
        return torch.tensor([[0.1, 4.0, 2.0]])

    monkeypatch.setattr(
        dflash_module, "_dflash_selector_refine_logits", fake_refine
    )
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.randn(6, 2), org_vocab_size=6),
        candidate_selector=SimpleNamespace(top_k=2),
        draft_config=parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {"selector_rank": 2, "selector_top_k": 2},
            }
        ),
        _selector_hot_token_id=hot_token_id,
        _selector_lm_head_weight=fp8_weight,
        _selector_lm_head_weight_scale=weight_scale,
        _selector_fp8_refine_topk=3,
    )
    model._transform_unary_logits = lambda logits: (
        DFlash2DraftModel._transform_unary_logits(model, logits)
    )

    with get_parallel().override(tp_size=1):
        ids, vals = DFlash2DraftModel.compute_candidates(model, torch.randn(1, 2))

    torch.testing.assert_close(ids, torch.tensor([[3, 4]]))
    torch.testing.assert_close(vals, torch.tensor([[4.0, 2.0]]))


@pytest.mark.parametrize(
    ("token_ids", "match"),
    [
        (torch.tensor([1]), "smaller than selector_top_k"),
        (torch.tensor([1, 1]), "unique ids"),
        (torch.tensor([1, 8]), "outside the target vocabulary"),
    ],
)
def test_selector_token_map_validation(token_ids, match):
    model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=2))
    lm_head = SimpleNamespace(weight=torch.randn(6, 2), org_vocab_size=6)
    with pytest.raises(ValueError, match=match):
        DFlash2DraftModel.set_selector_token_map(model, token_ids, lm_head)


def test_reduced_target_head_projects_only_selected_rows():
    processor = object.__new__(LogitsProcessor)
    processor.use_fp32_lm_head = False
    processor.rl_on_policy_target = None
    hidden = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    full_weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [-1.0, 2.0]]
    )
    selected = full_weight[torch.tensor([0, 2, 3])].contiguous()
    actual = LogitsProcessor._compute_lm_head(
        processor,
        hidden,
        SimpleNamespace(weight=full_weight),
        weight_override=selected,
    )
    torch.testing.assert_close(actual, hidden @ selected.T)
    assert actual.shape == (2, 3)


def test_reduced_target_top1_restores_global_ids_and_tie_order():
    global_ids = torch.tensor([3, 11, 29], dtype=torch.int64)
    logits = torch.tensor([[0.0, 4.0, 1.0], [2.0, 2.0, -1.0]])
    actual = _map_reduced_target_top1(logits, global_ids)
    torch.testing.assert_close(actual, torch.tensor([11, 3]))

    with pytest.raises(ValueError, match="width mismatch"):
        _map_reduced_target_top1(logits, torch.tensor([3, 11]))


def test_reduced_target_head_accepts_only_unmodified_greedy_batches():
    worker = object.__new__(DFlashWorkerV2)
    worker._use_reduced_target_head = True
    batch = SimpleNamespace(return_logprob=False, has_grammar=False)
    sampling_info = SimpleNamespace(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        logit_bias=None,
    )
    worker._validate_reduced_target_batch(batch, sampling_info)

    ineligible = [
        ("non-greedy sampling", {"is_all_greedy": False}),
        ("custom logit processor", {"has_custom_logit_processor": True}),
        (
            "sampling penalties",
            {"penalizer_orchestrator": SimpleNamespace(is_required=True)},
        ),
        ("logit bias", {"logit_bias": torch.zeros(1)}),
    ]
    for reason, override in ineligible:
        fields = vars(sampling_info).copy()
        fields.update(override)
        with pytest.raises(RuntimeError, match=reason):
            worker._validate_reduced_target_batch(
                batch, SimpleNamespace(**fields)
            )

    with pytest.raises(RuntimeError, match="returned logprobs"):
        worker._validate_reduced_target_batch(
            SimpleNamespace(return_logprob=True, has_grammar=False), sampling_info
        )
    with pytest.raises(RuntimeError, match="grammar"):
        worker._validate_reduced_target_batch(
            SimpleNamespace(return_logprob=False, has_grammar=True), sampling_info
        )


def test_target_top1_audit_reports_only_unmapped_full_head_winners():
    logits = torch.tensor(
        [[0.0, 4.0, 1.0, 2.0], [5.0, 1.0, 0.0, 2.0], [0.0, 1.0, 7.0, 3.0]]
    )
    selected = torch.tensor([True, True, False, True])
    actual = _missing_target_top1_ids(logits, selected)
    torch.testing.assert_close(actual, torch.tensor([2]))

    with pytest.raises(ValueError, match="width mismatch"):
        _missing_target_top1_ids(logits, torch.ones(3, dtype=torch.bool))


def test_dflash_target_map_can_be_independent_or_shared():
    assert (
        _resolve_dflash_target_token_map(
            SimpleNamespace(
                dflash_target_token_map="target.pt",
                speculative_token_map="proposal.pt",
            )
        )
        == "target.pt"
    )
    assert (
        _resolve_dflash_target_token_map(
            SimpleNamespace(
                dflash_target_token_map=None,
                speculative_token_map="shared.pt",
            )
        )
        == "shared.pt"
    )
    assert _resolve_dflash_target_token_map(SimpleNamespace()) is None


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


def test_dflash_candidate_logprobs_match_greedy_verifier_rows():
    candidates = torch.tensor([[7, 2, 1]])
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )

    actual = compute_dflash_candidate_logprobs(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=None,
        use_sampling_distribution=False,
    )
    expected = torch.stack(
        [
            torch.log_softmax(logits[0], dim=-1)[2],
            torch.log_softmax(logits[1], dim=-1)[1],
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_dflash_candidate_logprobs_match_top_k_sampling_distribution():
    candidates = torch.tensor([[7, 2, 1]])
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.75]]),
        top_ks=torch.tensor([2]),
        top_ps=torch.tensor([1.0]),
        need_top_k_sampling=True,
        # The CPU unit-test environment may have CUDA-only FlashInfer renorm
        # symbols installed. Top-p is exercised by the shared target-probability
        # builder on GPU; this analytic test keeps the exact sparse top-k path.
        need_top_p_sampling=False,
    )

    actual = compute_dflash_candidate_logprobs(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        use_sampling_distribution=True,
        max_top_k=2,
        uniform_top_k_value=2,
    )
    probs = build_dflash_verify_target_probs(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=3,
        bs=1,
        max_top_k=2,
        uniform_top_k_value=2,
    )
    expected = torch.stack(
        [
            probs[0, 0, 2].clamp_min(torch.finfo(probs.dtype).tiny).log(),
            probs[0, 1, 1].clamp_min(torch.finfo(probs.dtype).tiny).log(),
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_dflash_rejected_suffix_starts_at_first_unaccepted_proposal():
    metadata = build_dflash_rejected_draft_metadata(
        candidates=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        candidate_teacher_logprobs=torch.tensor(
            [[-0.1, -0.2, -0.3], [-1.1, -1.2, -1.3]]
        ),
        accept_len=torch.tensor([1, 3]),
    )

    assert metadata["offsets"] == [[2, 3], []]
    assert metadata["token_ids"] == [[12, 13], []]
    assert metadata["teacher_logprobs"][0] == pytest.approx([-0.2, -0.3])
    assert metadata["teacher_logprobs"][1] == []


def test_dflash_rejected_suffix_rejects_invalid_accept_length():
    with pytest.raises(ValueError, match="outside"):
        build_dflash_rejected_draft_metadata(
            candidates=torch.tensor([[10, 11, 12]]),
            candidate_teacher_logprobs=torch.tensor([[-0.1, -0.2]]),
            accept_len=torch.tensor([3]),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
