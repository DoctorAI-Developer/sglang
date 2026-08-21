from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
)
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP


def _eligible_mlp() -> Qwen2MoeMLP:
    quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    quant_method.block_quant = True
    quant_method.use_mxfp8 = False
    quant_method.weight_block_size = [128, 128]
    quant_method.w8a8_block_fp8_linear = deepgemm_w8a8_block_fp8_linear_with_fallback

    down_proj = nn.Module()
    down_proj.quant_method = quant_method
    down_proj.input_size_per_partition = 17_408

    mlp = Qwen2MoeMLP.__new__(Qwen2MoeMLP)
    nn.Module.__init__(mlp)
    mlp.down_proj = down_proj
    mlp.act_fn = nn.Identity()
    return mlp


def test_qwen_silu_fp8_fusion_installs_only_on_qualified_path(monkeypatch):
    mlp = _eligible_mlp()
    monkeypatch.setattr(qwen3_5, "_enable_qwen35_silu_fp8_quant_fusion", lambda: True)
    monkeypatch.setattr(qwen3_5, "_is_cuda", True)
    monkeypatch.setattr(qwen3_5, "get_parallel", lambda: SimpleNamespace(tp_size=1))
    monkeypatch.setattr(
        qwen3_5, "_log_qwen35_silu_fp8_quant_fusion", lambda *args: None
    )

    assert qwen3_5._maybe_enable_qwen35_silu_fp8_quant_fusion(mlp)
    assert isinstance(mlp.act_fn, qwen3_5._Qwen35SiluFp8QuantFusion)
    assert mlp.act_fn.hidden_size == 17_408


def test_qwen_silu_fp8_fusion_fails_closed_for_tp2(monkeypatch):
    mlp = _eligible_mlp()
    original = mlp.act_fn
    monkeypatch.setattr(qwen3_5, "_enable_qwen35_silu_fp8_quant_fusion", lambda: True)
    monkeypatch.setattr(qwen3_5, "_is_cuda", True)
    monkeypatch.setattr(qwen3_5, "get_parallel", lambda: SimpleNamespace(tp_size=2))
    monkeypatch.setattr(
        qwen3_5, "_log_qwen35_silu_fp8_quant_fusion", lambda *args: None
    )

    assert not qwen3_5._maybe_enable_qwen35_silu_fp8_quant_fusion(mlp)
    assert mlp.act_fn is original


def test_qwen_silu_fp8_wrapper_falls_back_for_unsupported_runtime_input():
    fallback = nn.Identity()
    wrapper = qwen3_5._Qwen35SiluFp8QuantFusion(fallback, hidden_size=128)
    cpu_input = torch.randn(2, 256, dtype=torch.bfloat16)

    assert wrapper(cpu_input) is cpu_input
