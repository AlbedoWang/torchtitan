# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for Llama3.

Uses AutoParallelGraph to apply solver-based SPMD sharding, then lets
graph_trainer trace and compile the placed model through its normal
`aot_fx_trace` train-step pipeline.
"""

import logging
import time
from pathlib import Path

import torch
from autoparallel import collectives, ForwardInputs
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenInnerAttention,
    VarlenMetadata,
)
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type


logger = logging.getLogger(__name__)


class LocalMapVarlenAttention(Module):
    """Varlen attention run on local shards inside an AutoParallel local_map.

    Packed document offsets are rank-local and the solver has no varlen rule,
    so the kernel runs per rank. With CP this is Ulysses: all-to-all over cp
    from token shards to head shards, attention over the full dp-local
    sequence, then the inverse all-to-all.
    """

    def __init__(
        self,
        inner_attention: VarlenInnerAttention,
        mesh: DeviceMesh,
        max_seqlen: int,
    ):
        super().__init__()
        self.inner_attention = inner_attention
        self.mesh = mesh
        self.max_seqlen = max_seqlen
        names = mesh.mesh_dim_names
        self.cp = mesh.size(names.index("cp")) if "cp" in names else 1
        # q/k/v/out are (T, H, D): tokens over dp/cp, heads over tp. Offsets are
        # concatenated per dp rank and replicated over cp/tp.
        self.qkv_placements = tuple(
            Shard(1) if name == "tp" else Shard(0) for name in names
        )
        self.offsets_placements = tuple(
            Shard(0) if name.startswith("dp") else Replicate() for name in names
        )

    @classmethod
    def from_inner(
        cls, inner_attention: torch.nn.Module, mesh: DeviceMesh, max_seqlen: int
    ) -> "LocalMapVarlenAttention":
        if not isinstance(inner_attention, VarlenInnerAttention):
            raise ValueError(
                "AutoParallel Llama3 requires varlen attention, got "
                f"{type(inner_attention).__name__}"
            )
        # Keep only the varlen kernel. With CP the all-to-alls run here on the
        # AutoParallel cp axis, so UlyssesCPVarlenInnerAttention's own
        # redistribute must not run as well.
        kernel = VarlenInnerAttention.Config(
            window_size=inner_attention.window_size
        ).build()
        return cls(kernel, mesh, max_seqlen)

    def _tokens_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (T/cp, H, D) -> (T, H/cp, D); chunk i of dim 0 goes to cp rank i.
        cp = self.cp
        t, h, d = x.shape
        x = x.view(t, cp, h // cp, d).movedim(1, 0).contiguous()
        return collectives.all_to_all(x.view(cp * t, h // cp, d), None, None, "cp")

    def _heads_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # (T, H/cp, D) -> (T/cp, H, D)
        cp = self.cp
        x = collectives.all_to_all(x, None, None, "cp")
        t, h, d = x.shape
        return x.view(cp, t // cp, h, d).movedim(0, 1).reshape(t // cp, cp * h, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        **kwargs,
    ) -> torch.Tensor:
        def local_attention(q, k, v, cu_seqlens):
            if self.cp > 1:
                q, k, v = (self._tokens_to_heads(x) for x in (q, k, v))
            out = self.inner_attention(
                q,
                k,
                v,
                attention_masks=VarlenMetadata(
                    cu_seqlens, cu_seqlens, self.max_seqlen, self.max_seqlen
                ),
                **kwargs,
            )
            return self._heads_to_tokens(out) if self.cp > 1 else out

        return collectives.local_map(
            local_attention,
            out_placements=(self.qkv_placements,),
            in_placements=(self.qkv_placements,) * 3 + (self.offsets_placements,),
            in_grad_placements=None,
            device_mesh=self.mesh,
        )(q, k, v, attention_masks.cu_seq_q)


def parallelize_autoparallel_llama(
    model,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    max_num_documents: int | None = None,
):
    """Apply AutoParallelGraph SPMD sharding to Llama3.

    Returns a sharded model carrying AutoParallel train-step metadata for
    graph_trainer's aot_fx_trace path.
    """
    if parallelism_context.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")
    if max_num_documents is None:
        raise ValueError(
            "AutoParallel Llama3 traces fixed-shape varlen metadata and requires "
            "dataloader.max_num_documents"
        )

    dense_names = ["dp_replicate", "dp_shard", "cp", "tp"]
    dense_names = [
        name
        for name in dense_names
        if parallelism_context.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallelism_context.get_mesh(dense_names)

    for layer in model.layers.values():
        layer.attention.inner_attention = LocalMapVarlenAttention.from_inner(
            layer.attention.inner_attention,
            dense_mesh,
            max_seqlen=training.max_context_length,
        )

    def input_fn():
        # One microbatch, matching what GraphTrainer feeds the module per step.
        dp_degree = parallelism_context.dp_replicate * parallelism_context.dp_shard
        num_tokens_per_dp_rank = training.num_tokens_per_microbatch_per_dp_rank
        num_tokens = num_tokens_per_dp_rank * dp_degree
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (num_tokens,),
            device=torch.device(device_type),
        )
        positions = (
            torch.arange(
                num_tokens,
                dtype=torch.int32,
                device=torch.device(device_type),
            )
            % training.max_context_length
        )
        # The dataloader always emits padding_mask, which also sizes the
        # varlen offsets (padding segments); llama3 layers ignore it.
        padding_mask = torch.zeros(
            num_tokens, dtype=torch.bool, device=torch.device(device_type)
        )
        # Each dp rank builds offsets for its own tokens (Decoder.preprocess_inputs).
        cu_seqlens = torch.cat(
            [
                create_varlen_metadata_for_document(
                    positions_dp,
                    padding_mask=padding_mask_dp,
                    max_num_documents=max_num_documents,
                    max_context_length=training.max_context_length,
                ).cu_seq_q
                for positions_dp, padding_mask_dp in zip(
                    positions.split(num_tokens_per_dp_rank),
                    padding_mask.split(num_tokens_per_dp_rank),
                    strict=True,
                )
            ]
        )
        attention_masks = VarlenMetadata(
            cu_seqlens,
            cu_seqlens,
            training.max_context_length,
            training.max_context_length,
        )
        return ForwardInputs(
            args=(tokens,),
            kwargs={
                "positions": positions,
                "padding_mask": padding_mask,
                "attention_masks": attention_masks,
            },
        )

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallelism_context.pp_enabled,
    )

    possible_input_shardings = {
        "dp_replicate": Shard(0),
        "dp_shard": Shard(0),
        "cp": Shard(0),
        "tp": Replicate(),
    }
    unsupported_axes = [
        name
        for name in dense_mesh.mesh_dim_names
        if name not in possible_input_shardings
    ]
    if unsupported_axes:
        raise ValueError(
            "Unsupported mesh axis for AutoParallel Llama3: "
            f"{unsupported_axes}. Supported axes: "
            f"{tuple(possible_input_shardings.keys())}"
        )
    x_sharding = tuple(
        possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
    )
    offsets_sharding = tuple(
        Shard(0) if name.startswith("dp") else Replicate()
        for name in dense_mesh.mesh_dim_names
    )

    output_sharding = tuple(
        Shard(1) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        solver=compile_config.autoparallel_solver,
        strategy_radius=(0 if compile_config.autoparallel_placements_load_path else 2),
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        # Flattened inputs: tokens, positions, padding_mask, cu_seq_q, cu_seq_k,
        # max_q, max_k. The int max lengths are graph inputs with only a
        # replicated option.
        replicated = (Replicate(),) * dense_mesh.ndim
        autop.add_input_constraints(
            [
                x_sharding,
                x_sharding,
                x_sharding,
                offsets_sharding,
                offsets_sharding,
                replicated,
                replicated,
            ]
        )
        autop.add_output_constraints([output_sharding])
        # HSDP: parameters are replicated on dp_replicate; the solver chooses
        # their placements on the remaining axes.
        if "dp_replicate" in dense_mesh.mesh_dim_names:
            autop.add_parameter_axis_constraint("dp_replicate", Replicate())

        if compile_config.autoparallel_placements_load_path:
            sharding_placement = autop.sharding_optimizer.load_placements(
                compile_config.autoparallel_placements_load_path
            )
            logger.info(
                "Loaded AutoParallel placements from %s",
                compile_config.autoparallel_placements_load_path,
            )
        else:
            t0 = time.time()
            sharding_placement = autop.optimize_placement(verbose=False)
            t1 = time.time()
            logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

            if compile_config.autoparallel_placements_save_path:
                save_path = Path(compile_config.autoparallel_placements_save_path)
                if (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                ):
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    autop.sharding_optimizer.save_placements(save_path)
                    logger.info("Saved AutoParallel placements to %s", save_path)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        # The local logits are the (T/dp, V/tp) shard that TorchTitan's
        # vocab-parallel cross-entropy consumes.
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
        )

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism_context=parallelism_context,
    )
    return model
