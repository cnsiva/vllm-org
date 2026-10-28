# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm b12x JIT kernels used by a loaded model."""

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _collect_warmup_units(
    model: torch.nn.Module,
    token_counts: tuple[int, ...],
    output_dtype: torch.dtype,
) -> Iterable[B12xWarmupUnit]:
    units: dict[object, B12xWarmupUnit] = {}
    for layer in model.modules():
        provider = getattr(layer, "b12x_warmup_provider", None)
        get_unit = getattr(provider, "get_b12x_warmup_unit", None)
        if not callable(get_unit):
            continue
        unit = get_unit(layer, token_counts, output_dtype)
        assert isinstance(unit, B12xWarmupUnit)
        units.setdefault(unit.key, unit)
    return units.values()


def _compile_warmup_units(
    units: Iterable[B12xWarmupUnit],
) -> Counter[str]:
    warmed: Counter[str] = Counter()
    with torch.inference_mode():
        for unit in units:
            unit.compile()
            warmed[unit.name] += 1
        if warmed:
            torch.accelerator.synchronize()
    return warmed


def b12x_warmup(worker: "Worker", cudagraph_capture_sizes: list[int]) -> None:
    if not current_platform.is_cuda():
        return
    if not current_platform.is_device_capability_family(120):
        return

    output_dtype = getattr(
        getattr(worker, "model_config", None),
        "dtype",
        torch.bfloat16,
    )
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16
    compile_sizes = worker.vllm_config.compilation_config.compile_sizes or []
    max_num_batched_tokens = worker.scheduler_config.max_num_batched_tokens
    serving_sizes = [
        max_num_batched_tokens,
        *cudagraph_capture_sizes,
        *(size for size in compile_sizes if isinstance(size, int)),
    ]
    max_tokens = max_num_batched_tokens
    max_num_scheduled_tokens = worker.scheduler_config.max_num_scheduled_tokens
    if max_num_scheduled_tokens is not None:
        max_tokens = max(max_tokens, max_num_scheduled_tokens)
    token_counts = b12x_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=serving_sizes,
    )
    # MBX 48: an MXFP8 drafter would compile its first signature inside CUDA-graph capture, where
    # resolve_precision refuses to run (`_policy.py:90-91`). FP4-Local's MTP ships bf16 projections
    # and NVFP4 experts (qwen-v5.3.md rows 23/24), so with the env unset — and even with it set, on
    # this checkpoint — this walks exactly the one model it walks today. Defensive on purpose: if
    # this build names the speculator's module differently we collect nothing and carry on rather
    # than fail the boot over an optimisation.
    import os

    models = [worker.get_model()]
    if os.environ.get("MBX_B12X_WARM_DRAFTER", "0") == "1":
        for holder in (worker, getattr(worker, "model_runner", None)):
            drafter_model = getattr(getattr(holder, "drafter", None), "model", None)
            if drafter_model is not None and all(drafter_model is not m for m in models):
                models.append(drafter_model)
    warmup_units: dict = {}
    for model in models:
        try:
            for unit in _collect_warmup_units(model, token_counts, output_dtype):
                warmup_units.setdefault(unit.key, unit)
        except Exception as exc:  # warm-up is an optimisation, never a boot failure
            logger.info_once(
                "b12x warm-up: skipped a model's units (%s: %s)",
                type(exc).__name__,
                exc,
            )
    for name, count in _compile_warmup_units(warmup_units.values()).items():
        logger.info_once(
            "Warmed up %d b12x %s kernel signature(s).",
            count,
            name,
        )
