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
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


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

    autoparallel_managed_cp = parallel_dims.cp_enabled
    if autoparallel_managed_cp:
        if parallel_dims.dp_shard != 1:
            raise ValueError(
                "3D AutoParallel requires data_parallel_shard_degree=1 so the "
                "fsdp axis is backed only by the AutoParallel-managed CP degree"
            )
        if not parallel_dims.dp_replicate_enabled or not parallel_dims.tp_enabled:
            raise ValueError(
                "3D AutoParallel requires dp_replicate, CP, and TP mesh axes"
            )
    elif parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP without 3D AP")

    dense_names = ["dp_replicate", "fsdp", "tp"]
    dense_names = [
        name
        for name in dense_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallel_dims.get_mesh(dense_names)

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int32,
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
            "fsdp": Replicate(),
            "tp": Replicate(),
        }
        if autoparallel_managed_cp
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
        x_sharding
        if autoparallel_managed_cp
        else tuple(
            Shard(2) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
        )
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

        model_output = (
            AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            )
            if parallel_dims.tp_enabled and not autoparallel_managed_cp
            else None
        )
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=model_output,
        )

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    return model
