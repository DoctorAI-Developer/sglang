from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dflash,
    handle_speculative_decoding,
)
from sglang.srt.managers.customized_info_utils import (
    append_dflash_reject_token_mask,
    append_dflash_rejected_draft_metadata,
    extend_customized_info_chunk,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dflash_opd_metadata_stays_token_aligned_then_flattens():
    req = SimpleNamespace(output_ids=[100, 101, 102], customized_info=None)

    append_dflash_reject_token_mask(req, 2, output_ids_already_updated=True)
    append_dflash_rejected_draft_metadata(
        req,
        2,
        anchor_index=0,
        offsets=[2, 3],
        token_ids=[42, 43],
        teacher_logprobs=[-1.25, -2.5],
        output_ids_already_updated=True,
    )

    assert req.customized_info["dflash_reject_token_mask"] == [False, False, True]
    assert req.customized_info["dflash_rejected_draft_offsets"] == [[], [], [2, 3]]

    accumulated = {}
    for key, values in req.customized_info.items():
        # Token zero was already streamed before speculative decode began.
        extend_customized_info_chunk(accumulated, key, values[1:])

    assert accumulated["dflash_reject_token_mask"] == [False, True]
    assert accumulated["dflash_rejected_draft_anchor_indices"] == [0, 0]
    assert accumulated["dflash_rejected_draft_offsets"] == [2, 3]
    assert accumulated["dflash_rejected_draft_token_ids"] == [42, 43]
    assert accumulated["dflash_rejected_draft_teacher_logprobs"] == [-1.25, -2.5]


def test_dflash_opd_metadata_rejects_mismatched_suffix_columns():
    req = SimpleNamespace(output_ids=[100, 101], customized_info=None)
    with pytest.raises(ValueError, match="length mismatch"):
        append_dflash_rejected_draft_metadata(
            req,
            1,
            anchor_index=0,
            offsets=[1, 2],
            token_ids=[42],
            teacher_logprobs=[-1.0, -2.0],
            output_ids_already_updated=True,
        )


def test_dflash_opd_metadata_flag_requires_dflash_algorithm():
    server_args = ServerArgs(model_path="dummy", enable_dflash_opd_metadata=True)
    with pytest.raises(ValueError, match="requires --speculative-algorithm DFLASH"):
        handle_speculative_decoding(server_args)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dflash_selector_tree_budget": 12}, "linear rejected suffixes"),
        ({"enable_dflash_reduced_target_head": True}, "full-vocabulary"),
    ],
)
def test_dflash_opd_metadata_rejects_incompatible_verifier_modes(overrides, match):
    server_args = ServerArgs(
        model_path="dummy",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="dummy-draft",
        enable_dflash_opd_metadata=True,
        device="cuda",
        **overrides,
    )
    with pytest.raises(ValueError, match=match):
        _handle_dflash(server_args)
