from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInputType


def _capture_mask_choice(spec_input_type, custom_mask):
    backend = object.__new__(FlashInferAttnBackend)
    backend.prefill_cuda_graph_metadata = {}
    choices = []

    def make_wrappers(bs, use_custom_mask=False):
        choices.append((bs, use_custom_mask))
        return [object()]

    backend._create_prefill_wrappers = make_wrappers
    spec_info = SimpleNamespace(
        spec_input_type=spec_input_type,
        custom_mask=custom_mask,
    )
    backend._prepare_cuda_graph_metadata(
        bs=1,
        num_tokens=13,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec_info,
    )
    return choices


def test_dflash_verify_reserves_custom_mask_buffers_before_live_mask_exists():
    assert _capture_mask_choice(SpecInputType.DFLASH_VERIFY, None) == [(1, True)]


def test_non_dflash_verify_without_mask_keeps_mask_buffers_disabled():
    assert _capture_mask_choice(SpecInputType.EAGLE_VERIFY, None) == [(1, False)]
