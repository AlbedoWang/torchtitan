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
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.tools.utils import device_type


logger = logging.getLogger(__name__)


def parallelize_autoparallel_llama(
    model,
    *,
    parallelism_context: ParallelismContext,
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
    if parallelism_context.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP yet")
    if parallelism_context.cp_enabled:
        raise ValueError("AutoParallel Llama3 does not support CP yet")
    if parallelism_context.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")

    # CP is rejected above, so the former flattened ``fsdp = dp_shard * cp``
    # axis maps exactly to ``dp_shard`` here.
    dense_names = ["dp_replicate", "dp_shard", "tp"]
    dense_names = [
        name
        for name in dense_names
        if parallelism_context.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallelism_context.get_mesh(dense_names)

    def input_fn():
        dp_degree = parallelism_context.dp_replicate * parallelism_context.dp_shard
        num_tokens_per_train_step = training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = (
                training.num_tokens_per_microbatch_per_dp_rank * dp_degree
            )
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (num_tokens_per_train_step,),
            device=torch.device(device_type),
        )
        positions = (
            torch.arange(
                num_tokens_per_train_step,
                dtype=torch.int32,
                device=torch.device(device_type),
            )
            % training.max_context_length
        )
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
        parallelism_context.pp_enabled,
    )

    possible_input_shardings = {
        "dp_replicate": Shard(0),
        "dp_shard": Shard(0),
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
                output_mesh=parallelism_context.get_mesh("tp"),
                output_placements=(Shard(1),),
                sharded_output_axis=1,
            )
            if parallelism_context.tp_enabled
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
        parallelism_context=parallelism_context,
    )
    return model
