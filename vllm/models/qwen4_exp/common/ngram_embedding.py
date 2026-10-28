# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import threading
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding

logger = init_logger(__name__)


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
            reduce_results=reduce_results,
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError

    @abstractmethod
    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Return prefetched embeddings for the full input token set."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if str(embedding_dtype).lower() == "nvfp4" or _mbx_ple_nvfp4_cfg():  # MBX NVFP4 PLE
            return _MbxPLENvfp4EmbeddingMethod()
        if _mbx_ple_fp8_dir():  # MBX 28: fp8 table from its own folder
            return _MbxPLEFp8DiskMethod()
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            if quant_config._resolve_quant_algo(prefix) == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, CompressedTensorsConfig)
            and quant_config.get_scheme_dict(None, layer_name=prefix) is None
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, INCConfig)
            and not quant_config.config_parser.resolve(None, prefix).quantized
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # Quark quantizes only Linear and MoE layers; PLE tables stay BF16.
        if isinstance(quant_config, QuarkConfig):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Reject prefetch finalization for resident embeddings."""
        raise RuntimeError("Device PLE embedding does not support prefetch")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP IDs, optionally reduce lookup results, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    ids_ptr,
    output_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Copy TP-owned PLE rows as raw bytes from a CUDA view of pinned host memory.

    Byte pointers keep the storage dtype out of the kernel signature, so
    dtypes Triton cannot lower on the GPU (FP8 E4M3FN before SM89) still work.
    """
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < row_bytes
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        output_ptr + row_id * row_bytes + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._row_bytes = self.embedding_dim * self.weight.element_size()
        self._block_d = triton.next_power_of_2(self._row_bytes)
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        self._prefetch_alloc_lock = threading.Lock()
        self._prefetch_rows = max_total_tokens * self.etp_data_parallel_size
        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self._row_bytes,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        embeddings = self._reduce_etp_embeddings(embeddings)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        buffer = self._prefetch_buffer
        if buffer is None:
            # First use allocates. The eager profile run always precedes
            # cudagraph capture, so allocation never happens mid-capture;
            # the lock keeps concurrent first callers from tearing the
            # stream/buffer pair.
            with self._prefetch_alloc_lock:
                buffer = self._prefetch_buffer
                if buffer is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "pinned PLE prefetch buffer must be allocated "
                            "eagerly, before cudagraph capture"
                        )
                    self._prefetch_stream = torch.cuda.Stream(
                        device=self._uva_weight.device
                    )
                    buffer = torch.empty(
                        self._prefetch_rows,
                        self._num_ngram_heads,
                        self.embedding_dim,
                        dtype=self.weight.dtype,
                        device=self._uva_weight.device,
                    )
                    self._prefetch_buffer = buffer
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE prefetch stream was not allocated")
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        if gathered_ids.shape[0] > buffer.shape[0]:
            raise ValueError(
                f"pinned PLE prefetch buffer holds {buffer.shape[0]} rows, "
                f"but the batch needs {gathered_ids.shape[0]}"
            )
        active_output = buffer[: gathered_ids.shape[0]]
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, optionally reduce ETP shards, and select rows."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        embeddings = prefetch_output[: slot_size * self.etp_data_parallel_size]
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Finalize the full-token pinned lookup with optional ETP reduction."""
        buffer = self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        # total_tokens counts this DP rank's tokens before SP sharding.
        output = buffer.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Pinned-host PLE embedding requires "
            "start_prefetch() and finalize_prefetch()."
        )


# ---- MBX: GPU-resident NVFP4 PLE table (vLLM 0.32, qwen4_exp) -------------------------------------------
import os as _mbx_os


def _mbx_ple_nvfp4_cfg() -> dict:
    """Our hibrid checkpoints' top-level `ple_quantization` block, iff it declares format "nvfp4"."""
    try:
        from vllm.config import get_current_vllm_config
        mc = get_current_vllm_config().model_config
        for obj in (getattr(mc, "hf_config", None), getattr(mc, "hf_text_config", None)):
            if obj is None:
                continue
            q = getattr(obj, "ple_quantization", None)
            if q is None and hasattr(obj, "to_dict"):
                q = obj.to_dict().get("ple_quantization")
            if isinstance(q, dict) and str(q.get("format", "")).lower() == "nvfp4":
                return q
    except Exception:
        pass
    return {}


class _MbxPLENvfp4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """PLE rows as NVFP4: uint8 packed e2m1 pairs + fp8 per-16 scales + one fp32 global; dequantized in embedding()."""

    BLOCK = 16
    _LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size, output_size, extra_weight_attrs
        if isinstance(layer, Qwen4ExpPLEPinnedHostEmbedding):
            raise RuntimeError("MBX NVFP4 PLE is device-resident only: set engram-config {\"cpu_offload\": false}")
        rows = int(sum(output_partition_sizes))
        dim = int(input_size_per_partition)
        assert dim % self.BLOCK == 0, f"PLE dim {dim} is not NVFP4-blockable"
        layer._mbx_mmap = _mbx_ple_mmap()
        layer._mbx_mmap_rows = rows
        if layer._mbx_mmap:
            layer._mbx_model_dir = _mbx_ple_model_dir()
        n_alloc = 0 if layer._mbx_mmap else rows
        packed = nn.Parameter(layer.allocate_embedding_weight(n_alloc, dim // 2, torch.uint8), requires_grad=False)
        layer.register_parameter("nvfp4_packed", packed)
        layer.register_parameter("weight", packed)   # alias: upstream logs weight.dtype/device/is_pinned()
        layer.register_parameter("nvfp4_scales", nn.Parameter(
            layer.allocate_embedding_weight(n_alloc, dim // self.BLOCK, torch.float8_e4m3fn), requires_grad=False))
        layer.register_parameter("nvfp4_global", nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False))
        layer.register_buffer("nvfp4_lut", torch.tensor(self._LUT, dtype=torch.float32), persistent=False)
        layer._mbx_nvfp4 = True
        layer._mbx_out_dtype = params_dtype
        layer._mbx_rows_loaded = 0
        print(f"MBX NVFP4 PLE: partition {rows} rows x {dim} (block {self.BLOCK})", flush=True)

    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "_mbx_mmap", False):
            got = int(getattr(layer, "_mbx_rows_loaded", 0))
            if got != int(layer._mbx_mmap_rows):
                raise RuntimeError(f"PLE: partition has {layer._mbx_mmap_rows} rows, {got} seen")
            layer.nvfp4_lut.data = layer.nvfp4_lut.data.to(layer.nvfp4_global.device)
            _mbx_mmap_attach(layer)
            return
        exp = int(layer.nvfp4_packed.shape[0])
        got = int(getattr(layer, "_mbx_rows_loaded", 0))
        si = layer.shard_indices
        if got != int(si.org_vocab_end_index - si.org_vocab_start_index):
            raise RuntimeError(f"MBX NVFP4 PLE: partition has {exp} rows but only {got} were loaded")
        layer.nvfp4_lut.data = layer.nvfp4_lut.data.to(layer.nvfp4_packed.device)
        print(f"PLE: NVFP4 table resident on {layer.nvfp4_packed.device} ({got} rows loaded of {exp}, "
              f"ETP={int(getattr(layer, 'tp_size', 1) or 1)}, global scale {float(layer.nvfp4_global):.4e})", flush=True)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        shape = input_.shape
        ids = input_.reshape(-1)
        sc8 = None
        if getattr(layer, "_mbx_mmap", False):
            pk, sc8 = _mbx_ple_rows(layer, ids)
        else:
            pk = layer.nvfp4_packed.index_select(0, ids)
        rows = pk.shape[0]
        codes = torch.stack(((pk & 0x0F), (pk >> 4)), dim=-1).view(rows, -1).long()
        x = layer.nvfp4_lut[codes]
        sc = (layer.nvfp4_scales.index_select(0, ids) if sc8 is None else sc8).to(torch.float32) * layer.nvfp4_global
        x = x.view(rows, -1, self.BLOCK) * sc.unsqueeze(-1)
        return x.view(*shape, -1).to(getattr(layer, "_mbx_out_dtype", None) or torch.bfloat16)

    def dequantize(self, layer, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        del layer
        return embeddings.to(output_dtype)


# ---- MBX 19 ----------------------------------------------------------------------------------------------------------
import ctypes as _mbx_ct
import re as _mbx_re


def _mbx_ple_mmap() -> bool:
    return _mbx_os.environ.get("MBX_PLE_MMAP", "0") == "1"


def _mbx_ple_model_dir() -> str:
    d = _mbx_os.environ.get("MBX_PLE_MMAP_DIR")
    if d:
        return d
    from vllm.config import get_current_vllm_config
    return get_current_vllm_config().model_config.model


_MBX_TAB_RE = _mbx_re.compile(r"^(.*)ngram_embedding\.(?:shard_(\d+)\.(weight|weight_scale)|nvfp4_shard_(\d+)\.(packed|scales))$")


def _mbx_ple_table(model_dir: str) -> dict:
    import json, struct
    idx = json.load(open(_mbx_os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    want, owners = {}, set()
    for name, f in idx.items():
        m = _MBX_TAB_RE.match(name)
        if not m:
            continue
        owners.add(m.group(1))
        if m.group(2) is not None:
            k, kd = int(m.group(2)), 0 if m.group(3) == "weight" else 1
        else:
            k, kd = int(m.group(4)), 0 if m.group(5) == "packed" else 1
        want[(k, kd)] = (name, _mbx_os.path.join(model_dir, f))
    if len(owners) != 1:
        raise RuntimeError(f"PLE: {len(owners)} tables in the index, expected 1")
    ns = len(want) // 2
    if not ns or sorted(want) != [(k, kd) for k in range(ns) for kd in (0, 1)]:
        raise RuntimeError("PLE: incomplete table in the index")
    hdrs, ent = {}, []
    for k in range(ns):
        for kd in (0, 1):
            name, path = want[(k, kd)]
            if path not in hdrs:
                with open(path, "rb") as fh:
                    hl = struct.unpack("<Q", fh.read(8))[0]
                    hdrs[path] = (8 + hl, json.loads(fh.read(hl)))
            base, h = hdrs[path]
            t = h[name]
            a, b = t["data_offsets"]
            r, cols = int(t["shape"][0]), int(t["shape"][1])
            if b - a != r * cols:
                raise RuntimeError(f"PLE: {name}: size mismatch")
            ent.append((path, base + a, r, cols))
    S, cp, cs = ent[0][2], ent[0][3], ent[1][3]
    for i, (_, _, r, cols) in enumerate(ent):
        if r != S or cols != (cs if i & 1 else cp):
            raise RuntimeError("PLE: unequal table shards")
    return {"ent": ent, "S": S, "ns": ns, "cp": cp, "cs": cs, "rows": S * ns, "dir": model_dir}


class _MbxDLDevice(_mbx_ct.Structure):
    _fields_ = [("device_type", _mbx_ct.c_int32), ("device_id", _mbx_ct.c_int32)]


class _MbxDLDataType(_mbx_ct.Structure):
    _fields_ = [("code", _mbx_ct.c_uint8), ("bits", _mbx_ct.c_uint8), ("lanes", _mbx_ct.c_uint16)]


class _MbxDLTensor(_mbx_ct.Structure):
    _fields_ = [("data", _mbx_ct.c_void_p), ("device", _MbxDLDevice), ("ndim", _mbx_ct.c_int32),
                ("dtype", _MbxDLDataType), ("shape", _mbx_ct.POINTER(_mbx_ct.c_int64)),
                ("strides", _mbx_ct.POINTER(_mbx_ct.c_int64)), ("byte_offset", _mbx_ct.c_uint64)]


class _MbxDLManagedTensor(_mbx_ct.Structure):
    pass


_MbxDLDeleter = _mbx_ct.CFUNCTYPE(None, _mbx_ct.POINTER(_MbxDLManagedTensor))
_MbxDLManagedTensor._fields_ = [("dl_tensor", _MbxDLTensor), ("manager_ctx", _mbx_ct.c_void_p),
                                ("deleter", _MbxDLDeleter)]


def _mbx_dl_u8(ptr: int, numel: int, device_id: int, keep: list, cuda: bool = True) -> torch.Tensor:
    shape = (_mbx_ct.c_int64 * 1)(numel)
    m = _MbxDLManagedTensor()
    m.dl_tensor.data = _mbx_ct.c_void_p(ptr)
    m.dl_tensor.device = _MbxDLDevice(2 if cuda else 1, device_id if cuda else 0)
    m.dl_tensor.ndim = 1
    m.dl_tensor.dtype = _MbxDLDataType(1, 8, 1)
    m.dl_tensor.shape = shape
    m.dl_tensor.strides = None
    m.dl_tensor.byte_offset = 0
    m.manager_ctx = None
    m.deleter = _MbxDLDeleter()
    keep.extend((shape, m))
    _mbx_ct.pythonapi.PyCapsule_New.restype = _mbx_ct.py_object
    _mbx_ct.pythonapi.PyCapsule_New.argtypes = [_mbx_ct.c_void_p, _mbx_ct.c_char_p, _mbx_ct.c_void_p]
    return torch.from_dlpack(_mbx_ct.pythonapi.PyCapsule_New(_mbx_ct.addressof(m), b"dltensor", None))


class _MbxTab:
    pass


_MBX_PLE_KEEP = []


def _mbx_mmap_map(tab: dict, device_id: int, cuda: bool = True) -> _MbxTab:
    import mmap
    import numpy as np
    T = _MbxTab()
    T.keep, T.maps, addr = [], [], {}
    for path, _, _, _ in tab["ent"]:
        if path in addr:
            continue
        fd = _mbx_os.open(path, _mbx_os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
        finally:
            _mbx_os.close(fd)
        T.maps.append(mm)
        addr[path] = (int(np.frombuffer(mm, dtype=np.uint8).ctypes.data), len(mm))
    lo = min(a for a, _ in addr.values())
    hi = max(a + ln for a, ln in addr.values())
    dev = torch.device("cuda", device_id) if cuda else torch.device("cpu")
    T.span = _mbx_dl_u8(lo, hi - lo, device_id, T.keep, cuda)
    offs = [addr[p][0] - lo + o for p, o, _, _ in tab["ent"]]
    T.offp = torch.tensor(offs[0::2], dtype=torch.int64, device=dev)
    T.offs = torch.tensor(offs[1::2], dtype=torch.int64, device=dev)
    T.S, T.ns, T.cp, T.cs, T.rows = tab["S"], tab["ns"], tab["cp"], tab["cs"], tab["rows"]
    T.ap = torch.arange(T.cp, dtype=torch.int64, device=dev)
    T.as_ = torch.arange(T.cs, dtype=torch.int64, device=dev)
    T.tab = tab
    return T


def _mbx_mm_gather(layer, ids: torch.Tensor):
    T = layer._mbx_tab
    ids = ids.clamp(0, T.rows - 1)
    k = torch.div(ids, T.S, rounding_mode="floor")
    r = ids - k * T.S
    pk = T.span[(T.offp[k] + r * T.cp).unsqueeze(1) + T.ap]
    sc = T.span[(T.offs[k] + r * T.cs).unsqueeze(1) + T.as_]
    return pk, sc.view(torch.float8_e4m3fn)


def _mbx_ple_rows(layer, ids: torch.Tensor):
    return _mbx_mm_gather(layer, ids)


def _mbx_mmap_populate(T, why: str) -> None:
    import mmap
    import time
    t0, tot = time.time(), 0
    adv = getattr(mmap, "MADV_POPULATE_READ", 22)
    for mm in T.maps:
        tot += len(mm)
        try:
            mm.madvise(adv)
        except Exception:  # noqa: BLE001
            for off in range(0, len(mm), mmap.PAGESIZE):
                mm[off]
    print(f"PLE: populated {tot / 2**30:.2f} GiB in {time.time() - t0:.1f}s ({why})", flush=True)


def _mbx_hook_warmup_populate() -> None:
    try:
        from vllm.v1.worker import gpu_worker as _gw
    except Exception as e:  # noqa: BLE001
        print(f"PLE: no warm-up hook ({e})", flush=True)
        return
    W = _gw.Worker
    if getattr(W.compile_or_warm_up_model, "_mbx_ple", False):
        return
    orig = W.compile_or_warm_up_model

    def wrapped(self, *a, **kw):
        out = orig(self, *a, **kw)
        for T in _MBX_PLE_KEEP:
            _mbx_mmap_populate(T, "after warm-up")
        return out

    wrapped._mbx_ple = True
    W.compile_or_warm_up_model = wrapped


def _mbx_mmap_attach(layer) -> None:
    import threading
    import time
    t0 = time.time()
    tab = _mbx_ple_table(layer._mbx_model_dir)
    if int(layer._mbx_mmap_rows) > tab["rows"] or int(layer.shard_indices.org_vocab_start_index) != 0 \
            or int(layer.shard_indices.org_vocab_end_index) > tab["rows"]:
        raise RuntimeError("PLE: MBX_PLE_MMAP needs the whole table on this rank (TP=1)")
    T = _mbx_mmap_map(tab, torch.cuda.current_device())
    layer._mbx_tab = T
    _MBX_PLE_KEEP.append(T)
    mode = _mbx_os.environ.get("MBX_PLE_MMAP_PREWARM", "auto")
    trigger = _mbx_os.environ.get("MBX_PLE_MMAP_TRIGGER", "/cache/mbx-ple-prewarm")
    if mode == "1":
        _mbx_mmap_populate(T, "at load")
    elif mode == "auto":
        _mbx_hook_warmup_populate()
    delay = int(mode) if mode.isdigit() and int(mode) > 1 else None

    def _watch():
        t_start, fired = time.time(), False
        while True:
            time.sleep(2.0)
            if _mbx_os.path.exists(trigger):
                _mbx_mmap_populate(T, "trigger")
                try:
                    _mbx_os.remove(trigger)
                except OSError:
                    pass
            elif delay is not None and not fired and time.time() - t_start >= delay:
                _mbx_mmap_populate(T, f"{delay}s after load")
                fired = True

    threading.Thread(target=_watch, name="mbx-ple-prewarm", daemon=True).start()
    gib = tab["rows"] * (tab["cp"] + tab["cs"]) / 2**30
    print(f"PLE: table mapped from {tab['dir']} ({tab['ns']} shards x {tab['S']} rows, {gib:.2f} GiB, prewarm={mode}, "
          f"{time.time() - t0:.1f}s)", flush=True)
    for f in _MBX_ATTACH_HOOKS:
        f(layer)


_MBX_ATTACH_HOOKS = []


# ---- MBX 20 ----------------------------------------------------------------------------------------------------------
_MBX_T = {}
_MBX_T_LAYERS = {}


def _mbx_t_mode() -> str:
    m = _mbx_os.environ.get("MBX_PLE_MMAP_MODE", "mmap").lower()
    return m if m in ("nvme", "mmap") else "mmap"


def _mbx_t_lib():
    if "lib" not in _MBX_T:
        c_ = _mbx_ct
        lib = c_.CDLL(_mbx_os.environ.get("MBX_PLE_LIB", "/opt/mbx/lib/libmbx_ple_nvme.so"))
        P, I32, I64 = c_.c_void_p, c_.c_int32, c_.c_int64
        lib.mbx_t_prepare.argtypes = [P, P, I32, I64, I32, I32, c_.c_char_p]
        lib.mbx_t_prepare.restype = c_.c_int
        lib.mbx_t_ok2.argtypes = [c_.c_char_p, I64, I32, I32]
        lib.mbx_t_ok2.restype = c_.c_int
        lib.mbx_t_fd.argtypes = [c_.c_char_p]
        lib.mbx_t_fd.restype = c_.c_int
        lib.mbx_t_open.argtypes = [c_.c_int]
        lib.mbx_t_open.restype = c_.c_int
        lib.mbx_t_read.argtypes = [c_.c_int, P, I64, P, P, I32, I32, c_.c_int]
        lib.mbx_t_read.restype = c_.c_int
        _MBX_T["lib"] = lib
        _MBX_T["qd"] = int(_mbx_os.environ.get("MBX_PLE_QD", "256"))
        _MBX_T["q"] = lib.mbx_t_open(_MBX_T["qd"]) == 0
        _MBX_T["nt"] = int(_mbx_os.environ.get("MBX_PLE_NVME_THREADS", "32"))
    return _MBX_T["lib"]


def _mbx_t_setup(layer) -> None:
    import hashlib
    import json
    import time
    lib = _mbx_t_lib()
    tab = layer._mbx_tab.tab
    ent, S, NS, cp, cs = tab["ent"], tab["S"], tab["ns"], tab["cp"], tab["cs"]
    sig = [_mbx_os.path.realpath(tab["dir"])]
    for p, o, r, cols in ent:
        st = _mbx_os.stat(p)
        sig.append([_mbx_os.path.basename(p), o, r, cols, st.st_size, st.st_mtime_ns])
    tag = hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:12]
    d = _mbx_os.environ.get("MBX_PLE_NVME_DIR", "/cache/ple-nvme")
    _mbx_os.makedirs(d, exist_ok=True)
    path = f"{d}/map-{tag}"
    t0 = time.time()
    reuse = lib.mbx_t_ok2(path.encode(), S * NS, cp, cs) == 1
    if not reuse:
        print(f"PLE: preparing map {path}", flush=True)
        paths = (_mbx_ct.c_char_p * len(ent))(*[e[0].encode() for e in ent])
        offs = (_mbx_ct.c_int64 * len(ent))(*[e[1] for e in ent])
        rc = lib.mbx_t_prepare(paths, offs, NS, S, cp, cs, path.encode())
        if rc == 5:
            raise RuntimeError("PLE: invalid quant")
        if rc:
            raise RuntimeError(f"PLE: map failed ({rc})")
    layer._mbx_t_fd = lib.mbx_t_fd(path.encode())
    if layer._mbx_t_fd < 0:
        raise RuntimeError("PLE: map open failed")
    print(f"PLE: map ready ({'reused' if reuse else f'built in {time.time() - t0:.0f} s'})", flush=True)


def _mbx_t_attach(layer) -> None:
    layer._mbx_name = f"t{len(_MBX_T_LAYERS)}"
    _MBX_T_LAYERS[layer._mbx_name] = layer
    if _mbx_t_mode() == "nvme":
        _mbx_t_setup(layer)


_MBX_ATTACH_HOOKS.append(_mbx_t_attach)


def _mbx_t_gather(layer, ids: torch.Tensor):
    import numpy as np
    lib = _mbx_t_lib()
    T = layer._mbx_tab
    uniq, inv = np.unique(ids.detach().cpu().numpy().astype(np.int64), return_inverse=True)
    n = len(uniq)
    op = np.empty((n, T.cp), dtype=np.uint8)
    os_ = np.empty((n, T.cs), dtype=np.uint8)
    rows = np.ascontiguousarray(uniq)
    rc = lib.mbx_t_read(layer._mbx_t_fd, rows.ctypes.data, n, op.ctypes.data, os_.ctypes.data, T.cp, T.cs,
                        0 if _MBX_T.get("q") else _MBX_T["nt"])
    if rc:
        raise RuntimeError(f"PLE: table read failed ({rc})")
    return torch.from_numpy(op[inv.reshape(-1)]), torch.from_numpy(os_[inv.reshape(-1)])


def _mbx_t_gather_sync(layer, ids, packed_out, scales_out) -> None:
    if _mbx_t_mode() != "nvme" or (ids.is_cuda and torch.cuda.is_current_stream_capturing()):
        pk, sc = _mbx_mm_gather(layer, ids)
        sc = sc.view(torch.uint8)
    else:
        pk, sc = _mbx_t_gather(layer, ids)
    packed_out.copy_(pk)
    scales_out.copy_(sc)


def _mbx_t_gather_impl(ids: torch.Tensor, packed_out: torch.Tensor, scales_out: torch.Tensor, layer_name: str) -> None:
    layer = _MBX_T_LAYERS[layer_name]
    _mbx_t_gather_sync(layer, ids.clamp(0, layer._mbx_tab.rows - 1), packed_out, scales_out)


_mbx_t_gather_eager = eager_break_during_capture(_mbx_t_gather_impl)


def _mbx_ple_rows(layer, ids: torch.Tensor):
    if _mbx_t_mode() != "nvme":
        return _mbx_mm_gather(layer, ids)
    T = layer._mbx_tab
    n = ids.shape[0]
    pk = torch.empty((n, T.cp), dtype=torch.uint8, device=ids.device)
    sc = torch.empty((n, T.cs), dtype=torch.uint8, device=ids.device)
    _mbx_t_gather_eager(ids, pk, sc, layer._mbx_name)
    return pk, sc.view(torch.float8_e4m3fn)


# ---- MBX 21 (async reads; based on Alssndr0's opt_ple_async, Apache-2.0) ---------------------------------------------
def _mbx_pa_mode() -> str:
    m = _mbx_os.environ.get("MBX_PLE_ASYNC", "0").lower()
    return "1" if m == "1" else "0"


class _MbxPA:
    pass


def _mbx_pa_init(layer) -> None:
    mode = _mbx_pa_mode()
    if mode == "0" or _mbx_t_mode() != "nvme":
        return
    try:
        from vllm.config import get_current_vllm_config
        mbt = int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)
    except Exception:  # noqa: BLE001
        mbt = 8192
    T = layer._mbx_tab
    A = _MbxPA()
    A.max_n = int(_mbx_os.environ.get("MBX_PLE_ASYNC_MAX", str(mbt * 64)))
    A.pin_ids = torch.zeros(A.max_n, dtype=torch.int64, pin_memory=True)
    A.pin_pk = torch.zeros((A.max_n, T.cp), dtype=torch.uint8, pin_memory=True)
    A.pin_sc = torch.zeros((A.max_n, T.cs), dtype=torch.uint8, pin_memory=True)
    A.flags = torch.zeros(32, dtype=torch.int32, pin_memory=True)
    A.graphs = True
    A.prefetch = _mbx_os.environ.get("MBX_PLE_ASYNC_PREFETCH", "1") == "1"
    A.checks = 3
    A.pending = None
    _mbx_t_lib()
    lib = _mbx_ct.CDLL(_mbx_os.environ.get("MBX_PLE_ASYNC_LIB", "/opt/mbx/lib/libmbx_ple_async.so"))
    U64, I = _mbx_ct.c_uint64, _mbx_ct.c_int
    lib.mbx_pa_init.argtypes = [_mbx_ct.c_char_p, I, I, I, I, I, I, U64, U64, U64, U64, U64]
    lib.mbx_pa_post.argtypes = [U64, I]
    lib.mbx_pa_wait.argtypes = [U64]
    lib.mbx_pa_jobs.restype = _mbx_ct.c_long
    lib.mbx_pa_read_us.restype = _mbx_ct.c_double
    fp = A.flags.data_ptr()
    rc = lib.mbx_pa_init(_mbx_os.environ.get("MBX_PLE_LIB", "/opt/mbx/lib/libmbx_ple_nvme.so").encode(),
                         layer._mbx_t_fd, T.cp, T.cs, A.max_n, _MBX_T["nt"], _MBX_T["qd"], A.pin_ids.data_ptr(),
                         A.pin_pk.data_ptr(), A.pin_sc.data_ptr(), fp, fp + 64)
    if rc:
        print(f"PLE async: init failed ({rc}), synchronous reads", flush=True)
        return
    A.lib = lib
    layer._mbx_pa = A
    print(f"PLE async: on (graphs {'on' if A.graphs else 'off'}, prefetch {'on' if A.prefetch else 'off'}, "
          f"max {A.max_n} ids)", flush=True)


_MBX_ATTACH_HOOKS.append(_mbx_pa_init)


def _mbx_pa_ok(A) -> bool:
    return A.graphs or not torch.cuda.is_current_stream_capturing()


def _mbx_pa_post(A, ids: torch.Tensor, n: int) -> None:
    A.pin_ids[:n].copy_(ids.reshape(-1).to(torch.int64), non_blocking=True)
    if A.lib.mbx_pa_post(torch.cuda.current_stream().cuda_stream, n):
        raise RuntimeError("PLE async: post failed")


def _mbx_pa_wait(A) -> None:
    if A.lib.mbx_pa_wait(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("PLE async: wait failed")


def _mbx_pa_drop(A) -> None:
    if A.pending is not None:
        A.pending = None
        _mbx_pa_wait(A)


_mbx_t_gather_sync_base = _mbx_t_gather_sync


def _mbx_t_gather_sync(layer, ids, packed_out, scales_out) -> None:  # noqa: F811
    A = getattr(layer, "_mbx_pa", None)
    n = ids.numel()
    if A is None or not ids.is_cuda or n == 0 or n > A.max_n or not _mbx_pa_ok(A):
        if A is not None and _mbx_pa_ok(A):
            _mbx_pa_drop(A)
        _mbx_t_gather_sync_base(layer, ids, packed_out, scales_out)
        return
    pend, A.pending = A.pending, None
    if pend is not None and pend != n:
        _mbx_pa_wait(A)
        pend = None
    if pend is not None and A.checks > 0 and not torch.cuda.is_current_stream_capturing():
        A.checks -= 1
        ref = ids.reshape(-1).to(torch.int64).cpu()
        if not torch.equal(A.pin_ids[:n], ref):
            A.prefetch = False
            print("PLE async: prefetched ids differ, prefetch off", flush=True)
            _mbx_pa_wait(A)
            pend = None
    if pend is None:
        _mbx_pa_post(A, ids, n)
    _mbx_pa_wait(A)
    packed_out.copy_(A.pin_pk[:n].view(packed_out.shape), non_blocking=True)
    scales_out.copy_(A.pin_sc[:n].view(scales_out.shape), non_blocking=True)
    if A.lib.mbx_pa_failed():
        raise RuntimeError("PLE async: table read failed")


def _mbx_pa_start_impl(mod, input_ids, query_start_loc, ngram_context, layer_name) -> None:
    layer = _MBX_T_LAYERS.get(layer_name)
    A = getattr(layer, "_mbx_pa", None)
    if A is None or not _mbx_pa_ok(A):
        return
    _mbx_pa_drop(A)
    if not A.prefetch:
        return
    ids = mod.compute_ngram_ids(input_ids, query_start_loc, ngram_context).reshape(-1)
    n = ids.numel()
    if n == 0 or n > A.max_n:
        return
    _mbx_pa_post(A, ids.clamp(0, layer._mbx_tab.rows - 1), n)
    A.pending = n


_mbx_pa_start_eager = eager_break_during_capture(_mbx_pa_start_impl)



# ---- MBX 28: fp8 PLE table (global scale) from its own folder, on the table reader -----------------------------------
def _mbx_ple_fp8_dir() -> str:
    d = _mbx_os.environ.get("MBX_PLE_FP8_DIR", "")
    return d if d and _mbx_ple_mmap() else ""


def _mbx_ple_fp8_table(d: str):
    import json, struct
    pat = _mbx_re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")
    shards, scale = {}, None
    for f in sorted(_mbx_os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        path = _mbx_os.path.join(d, f)
        with open(path, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
            for name, t in hdr.items():
                if name == "__metadata__" or ".ngram_embedding." not in name:
                    continue
                m = pat.search(name)
                if m:
                    if t["dtype"] != "F8_E4M3":
                        raise RuntimeError(f"PLE fp8: {name} is {t['dtype']}")
                    r, cols = int(t["shape"][0]), int(t["shape"][1])
                    shards[int(m.group(1))] = (path, 8 + hl + t["data_offsets"][0], r, cols)
                elif name.endswith("ngram_embedding.weight_scale"):
                    lo, hi = t["data_offsets"]
                    fh.seek(8 + hl + lo)
                    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[t["dtype"]]
                    scale = float(torch.frombuffer(bytearray(fh.read(hi - lo)), dtype=dt).float().reshape(-1)[0])
    ns = len(shards)
    if not ns or sorted(shards) != list(range(ns)) or scale is None:
        raise RuntimeError(f"PLE fp8: incomplete table in {d}")
    S, cp = shards[0][2], shards[0][3]
    ent = []
    for k in range(ns):
        path, off, r, cols = shards[k]
        if r != S or cols != cp:
            raise RuntimeError("PLE fp8: unequal table shards")
        ent += [(path, off, S, cp), (path, off, S, 0)]
    return {"ent": ent, "S": S, "ns": ns, "cp": cp, "cs": 0, "rows": S * ns, "dir": d}, scale


class _MbxPLEFp8DiskMethod(Qwen4ExpPLEFp8EmbeddingMethod):
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size, output_size, extra_weight_attrs
        if isinstance(layer, Qwen4ExpPLEPinnedHostEmbedding):
            raise RuntimeError("MBX fp8 PLE table needs engram-config {\"cpu_offload\": false}")
        layer.register_parameter("weight", nn.Parameter(
            torch.empty((0, int(input_size_per_partition)), dtype=torch.float8_e4m3fn), requires_grad=False))
        layer.register_parameter("weight_scale", nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False))
        layer._mbx_mmap = True
        layer._mbx_mmap_rows = int(sum(output_partition_sizes))
        layer._mbx_out_dtype = params_dtype

    def process_weights_after_loading(self, layer) -> None:
        import time
        t0 = time.time()
        tab, scale = _mbx_ple_fp8_table(_mbx_ple_fp8_dir())
        if int(layer._mbx_mmap_rows) > tab["rows"]:
            raise RuntimeError("PLE fp8: the whole table must be on this rank (TP=1)")
        layer.weight_scale.data.fill_(scale)
        T = _mbx_mmap_map(tab, torch.cuda.current_device())
        layer._mbx_tab = T
        _MBX_PLE_KEEP.append(T)
        print(f"PLE: fp8 table mapped from {tab['dir']} ({tab['ns']} shards x {tab['S']} rows x {tab['cp']}, "
              f"{tab['rows'] * tab['cp'] / 2**30:.2f} GiB, scale {scale:.4e}, {time.time() - t0:.1f}s)", flush=True)
        for f in _MBX_ATTACH_HOOKS:
            f(layer)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        shape = input_.shape
        pk, _ = _mbx_ple_rows(layer, input_.reshape(-1))
        return pk.view(torch.float8_e4m3fn).view(*shape, -1)
