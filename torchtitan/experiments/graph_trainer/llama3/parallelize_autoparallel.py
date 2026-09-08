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

import time
from pathlib import Path

import torch
from autoparallel import make_context_parallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
    autoparallel_constructor_kwargs,
    autoparallel_optimize_kwargs,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


def _build_autoparallel_mesh(parallel_dims: ParallelDims):
    if not parallel_dims.cp_enabled:
        mesh_axis_names = [
            name
            for name in ("dp_replicate", "fsdp", "tp")
            if parallel_dims.get_optional_mesh(name) is not None
        ]
        return parallel_dims.get_mesh(mesh_axis_names)

    if not parallel_dims.tp_enabled:
        raise ValueError("3D AutoParallel requires CP and TP mesh axes")

    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard != 1:
            raise ValueError(
                "3D AutoParallel does not support simultaneous dp_replicate and "
                "dp_shard axes"
            )
        return parallel_dims.get_mesh(["dp_replicate", "fsdp", "tp"])

    mesh_axes = [
        ("dp_shard", parallel_dims.dp_shard),
        ("cp", parallel_dims.cp),
        ("tp", parallel_dims.tp),
    ]
    active_axes = [(name, degree) for name, degree in mesh_axes if degree > 1]
    return init_device_mesh(
        device_type,
        tuple(degree for _, degree in active_axes),
        mesh_dim_names=tuple(name for name, _ in active_axes),
    )


def _apply_autoparallel_context_parallel_attention(model, dense_mesh) -> None:
    """Use AutoParallel's CP-aware SDPA while preserving Llama's BLNH API."""
    for layer in model.layers.values():
        attention = layer.attention
        inner_attention = attention.inner_attention
        if not isinstance(inner_attention, ScaledDotProductAttention):
            raise ValueError(
                "AutoParallel Llama context parallelism currently requires SDPA"
            )

        cp_attention = make_context_parallel(
            dense_mesh,
            kind="sdpa",
            is_causal=True,
            scale=attention.scaling,
            enable_gqa=attention.enable_gqa,
        )

        def make_forward(cp_attention, expected_scale, expected_enable_gqa):
            def forward(
                q_BLNH,
                k_BLNH,
                v_BLNH,
                *,
                attention_masks=None,
                scale=None,
                enable_gqa=False,
                is_causal=True,
                **kwargs,
            ):
                if attention_masks is not None:
                    raise ValueError(
                        "AutoParallel context-parallel SDPA does not support "
                        "attention_masks"
                    )
                if (
                    scale != expected_scale
                    or enable_gqa != expected_enable_gqa
                    or not is_causal
                ):
                    raise ValueError(
                        "AutoParallel context-parallel SDPA runtime options must "
                        "match the options captured during model construction"
                    )
                q_BNLH, k_BNLH, v_BNLH = (
                    tensor.transpose(1, 2) for tensor in (q_BLNH, k_BLNH, v_BLNH)
                )
                return cp_attention(q_BNLH, k_BNLH, v_BNLH).transpose(1, 2)

            return forward

        inner_attention.forward = make_forward(
            cp_attention,
            attention.scaling,
            attention.enable_gqa,
        )


def parallelize_autoparallel_llama(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph SPMD sharding to Llama3.

    Returns a sharded model carrying AutoParallel train-step metadata for
    graph_trainer's aot_fx_trace path.
    """
    validate_autoparallel_config(compile_config)

    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")

    cp_enabled = parallel_dims.cp_enabled
    if not cp_enabled and parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP without 3D AP")

    dense_mesh = _build_autoparallel_mesh(parallel_dims)
    explicit_cp_axis = "cp" in (dense_mesh.mesh_dim_names or ())
    if cp_enabled:
        _apply_autoparallel_context_parallel_attention(model, dense_mesh)

    vocab_size = model.config.vocab_size

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        return tokens, positions

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )

    possible_input_shardings = (
        {
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "fsdp": Replicate(),
            "cp": Shard(1),
            "tp": Replicate(),
        }
        if cp_enabled
        else {
            "dp_replicate": Shard(0),
            "fsdp": Shard(0),
            "tp": Replicate(),
        }
    )
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

    output_sharding = (
        tuple(
            Shard(2)
            if name == "tp"
            else Shard(1)
            if name == "cp"
            else Shard(0)
            for name in dense_mesh.mesh_dim_names
        )
        if explicit_cp_axis or not cp_enabled
        else x_sharding
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        **autoparallel_constructor_kwargs(compile_config),
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding, x_sharding])
        autop.add_output_constraints([output_sharding])

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
            sharding_placement = autop.optimize_placement(
                **autoparallel_optimize_kwargs(compile_config)
            )
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

        model_output = (
            AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            )
            if parallel_dims.tp_enabled and (explicit_cp_axis or not cp_enabled)
            else None
        )
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=model_output,
            manages_context_parallel_input=not explicit_cp_axis,
        )

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    return model
