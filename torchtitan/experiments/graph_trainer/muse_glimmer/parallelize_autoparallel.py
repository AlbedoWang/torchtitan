# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

import torch
from autoparallel import ForwardInputs
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    autoparallel_constructor_kwargs,
    autoparallel_optimize_kwargs,
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

from .sdpa import build_packed_document_attention_masks


def parallelize_autoparallel_muse_glimmer(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph SPMD sharding to Muse Glimmer."""
    validate_autoparallel_config(compile_config)

    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Muse Glimmer does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Muse Glimmer does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Muse Glimmer does not support PP yet")

    dense_axis_names = ["dp_replicate", "fsdp", "tp"]
    dense_axis_names = [
        name
        for name in dense_axis_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallel_dims.get_mesh(dense_axis_names)
    window_sizes = {
        layer.attention.window_size
        for layer in model.config.layers
        if layer.attention.window_size is not None
    }

    def input_fn():
        dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        placement_batch_size = training.local_batch_size * dp_degree
        if (
            training.global_batch_size > 0
            and training.global_batch_size % placement_batch_size != 0
        ):
            raise ValueError(
                "AutoParallel Muse Glimmer global batch size must be divisible "
                "by one distributed microbatch: "
                f"{training.global_batch_size} % {placement_batch_size} != 0"
            )
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (placement_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(placement_batch_size, 1)
        attention_masks = build_packed_document_attention_masks(
            positions,
            window_sizes,
        )
        return ForwardInputs(
            args=(tokens,),
            kwargs={
                "positions": positions,
                "attention_masks": attention_masks,
            },
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )

    possible_input_shardings = {
        "dp_replicate": Shard(0),
        "fsdp": Shard(0),
        "tp": Replicate(),
    }
    unsupported_axes = [
        name
        for name in dense_mesh.mesh_dim_names
        if name not in possible_input_shardings
    ]
    if unsupported_axes:
        raise ValueError(
            "Unsupported mesh axis for AutoParallel Muse Glimmer: "
            f"{unsupported_axes}. Supported axes: "
            f"{tuple(possible_input_shardings.keys())}"
        )
    input_sharding = tuple(
        possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
    )
    output_sharding = tuple(
        Shard(2) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        repeated_subgraphs=True,
        **autoparallel_constructor_kwargs(compile_config),
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding] * (2 + 1 + len(window_sizes)))
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
            start = time.perf_counter()
            sharding_placement = autop.optimize_placement(
                **autoparallel_optimize_kwargs(compile_config)
            )
            logger.info(
                "AutoParallelGraph placement search took %.2f seconds",
                time.perf_counter() - start,
            )

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
            if parallel_dims.tp_enabled
            else None
        )
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=model_output,
        )

    return apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
