# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, reuse_packed_weight_storage
from vllm.utils.b12x import (
    get_b12x_mxfp8_linear as _import_b12x_mxfp8,
)
from vllm.utils.torch_utils import current_stream

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _mbx_a16_kwargs() -> dict:
    """MBX 48: pin the A16 route (and optionally its tile) so precision cannot drift.

    b12x picks the route from its policy table (`_linear.py:1070` pops mode with default "auto").
    The shipped GB10 profile has 10 blockscaled_precision leaves and not one of them is a Qwen MXFP8
    GEMM, so resolution falls to `_heuristic` (`_policy.py:63-70`), which backfills a16 rows for
    M in 1..8 only: `auto` is W8A16 at c=1 decode and the activation-quantized route at every
    prefill and every batch above ~8 (`_a16.py:279-291`). MBX_MXFP8_A16=1 makes that a hard choice
    instead of a silent flip — b12x raises rather than switching. MBX_MXFP8_A16_CONFIG="BN,BK,SPLIT"
    additionally pins the A16 tile, which no profile leaf ever measured for these shapes (unset keeps
    b12x's own default (64,64,1), `_a16.py:131-135`).
    """
    import os

    if os.environ.get("MBX_MXFP8_A16", "0") != "1":
        return {}
    kwargs = {"mode": "a16"}
    tile = os.environ.get("MBX_MXFP8_A16_CONFIG", "").strip()
    if tile:
        kwargs["_config"] = tuple(int(v) for v in tile.split(","))
    if not getattr(_mbx_a16_kwargs, "_logged", False):
        _mbx_a16_kwargs._logged = True
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "MBX 48: b12x MXFP8 pinned to %s (bf16 activations, E4M3 weights and UE8M0 scales as"
            " shipped; never the activation-quantized route).",
            kwargs,
        )
    return kwargs


def _apply_b12x_mxfp8_packed_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    packed_weight = layer.b12x_mxfp8_packed_weight

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_shape = [*x.shape[:-1], int(packed_weight.out_features)]

    mxfp8 = _import_b12x_mxfp8()
    assert mxfp8 is not None
    output = mxfp8.mm(
        input_2d,
        packed_weight,
        bias=bias,
        expected_m=max(1, int(input_2d.shape[0])),
        **_mbx_a16_kwargs(),  # MBX 48
    )
    return output.view(*output_shape)


class B12xMxfp8LinearKernel(Mxfp8LinearKernel):
    """ModelOpt MXFP8 linear through the native b12x SM120 dense GEMM path."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x MXFP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x MXFP8 kernels require a Blackwell 12x device"
        mxfp8 = _import_b12x_mxfp8()
        if mxfp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not mxfp8.is_supported():
            return False, "b12x.gemm.mxfp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        # MBX 48: was `return True, None`. The A16 TMA kernel needs K % 32 == 0 and N % 8 == 0
        # (`_a16.py:149`), pack_mxfp8_linear_weight needs K a multiple of the MXFP8 block
        # (`_linear.py:291-295`), and process_weights_after_loading asserts the weight is 2D (:75)
        # — so a packed bmm layer or a shape outside those rules must decline here instead of
        # raising inside weight loading, which aborts the boot. Every MXFP8 GEMM of the Qwen v5.3
        # FP4-Local checkpoint passes (K in 2560/6144/640, all N%8==0, none packed bmm), so this is
        # insurance: with the arm's linear_backend_per_quant={"mxfp8":"b12x"} the candidate list
        # holds only this class and a decline is a named boot error; in the unfiltered walk the
        # next candidate is EmulationMxfp8LinearKernel, not Marlin (linear/__init__.py:552-561).
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            MXFP8_BLOCK_SIZE,
        )

        if c.bmm_batch_size is not None:
            return False, "b12x MXFP8 does not take packed bmm weights"
        n, k = (int(v) for v in c.weight_shape)  # (out_features, in_features)
        if k % MXFP8_BLOCK_SIZE:
            return False, f"b12x MXFP8 needs K%{MXFP8_BLOCK_SIZE} == 0, got K={k}"
        if k % 32 or n % 8:
            return False, f"b12x MXFP8 A16 needs K%32 and N%8 == 0, got K={k} N={n}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        assert weight.dtype == MXFP8_VALUE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_VALUE_DTYPE}, got {weight.dtype}"
        )
        assert weight.ndim == 2, f"b12x MXFP8 weight must be 2D, got {weight.ndim}D"
        assert hasattr(layer, "weight_scale"), "b12x MXFP8 linear requires weight_scale"

        out_features, in_features = map(int, weight.shape)
        assert in_features % MXFP8_BLOCK_SIZE == 0, (
            "b12x MXFP8 requires input features divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {in_features}"
        )
        weight_scale = layer.weight_scale.data
        assert weight_scale.dtype == MXFP8_SCALE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_SCALE_DTYPE} weight_scale, "
            f"got {weight_scale.dtype}"
        )
        assert weight_scale.ndim == 2, (
            f"b12x MXFP8 weight_scale must be 2D, got {weight_scale.ndim}D"
        )

        mxfp8 = _import_b12x_mxfp8()
        assert mxfp8 is not None
        scale_k = in_features // MXFP8_BLOCK_SIZE
        packed_weight = mxfp8.pack_weight(
            weight[:out_features, :in_features].detach(),
            weight_scale[:out_features, :scale_k].detach(),
        )
        layer.b12x_mxfp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_mxfp8_packed_weight", None),
            packed_weight,
        )
        replace_parameter(layer, "weight", weight.new_empty((0,)))
        replace_parameter(layer, "weight_scale", weight_scale.new_empty((0,)))
        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        packed_weight = layer.b12x_mxfp8_packed_weight
        device = torch.device(packed_weight.weight.values.device)

        def compile() -> None:
            mxfp8 = _import_b12x_mxfp8()
            assert mxfp8 is not None
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, int(packed_weight.in_features)),
                    dtype=output_dtype,
                    device=device,
                )
                mxfp8.mm(
                    source,
                    packed_weight,
                    expected_m=max(1, int(tokens)),
                    stream=current_stream().cuda_stream,
                    **_mbx_a16_kwargs(),  # MBX 48: warm the route we will actually run
                )

        return B12xWarmupUnit(
            name="MXFP8",
            key=(
                type(self),
                device,
                int(packed_weight.in_features),
                int(packed_weight.padded_in_features),
                int(packed_weight.out_features),
                output_dtype,
            ),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_b12x_mxfp8_packed_linear(layer, x, bias)
