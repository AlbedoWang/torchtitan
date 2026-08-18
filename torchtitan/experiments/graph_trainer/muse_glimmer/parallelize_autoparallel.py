# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from autoparallel import ForwardInputs
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from .sdpa import (
    build_packed_document_attention_masks,
    MuseGlimmerPackedDocumentSDPA,
)


def _attention_masks(model, positions, window_sizes):
    inner_attention = model.config.first_attention.inner_attention
    if isinstance(inner_attention, FlexAttention.Config):
        maybe_register_blockmask_pytree_node()
        return model._get_attention_masks_with_factory(positions, create_block_mask)
    if isinstance(inner_attention, MuseGlimmerPackedDocumentSDPA.Config):
        return build_packed_document_attention_masks(positions, window_sizes)
    raise TypeError(
        "AutoParallel Muse Glimmer supports FlexAttention and packed-document "
        f"SDPA, got {type(inner_attention).__name__}"
    )


def _input_constraints(traced_inputs, dense_mesh, global_batch_size):
    flat_inputs, _ = torch.utils._pytree.tree_flatten(
        (tuple(traced_inputs.args), traced_inputs.kwargs)
    )
    constraints = []
    for value in flat_inputs:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "AutoParallel Muse Glimmer requires tensor-only forward input "
                f"leaves, got {type(value).__name__}"
            )
        batch_sharded = value.ndim > 0 and value.shape[0] == global_batch_size
        constraints.append(
            tuple(
                (
                    Shard(0)
                    if batch_sharded and axis in ("dp_replicate", "fsdp")
                    else Replicate()
                )
                for axis in dense_mesh.mesh_dim_names
            )
        )
    return constraints


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

    global_batch_size = training.global_batch_size
    if global_batch_size < 0:
        dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        global_batch_size = training.local_batch_size * dp_degree

    def input_fn():
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        attention_masks = _attention_masks(model, positions, window_sizes)
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

    supported_axes = ("dp_replicate", "fsdp", "tp")
    unsupported_axes = [
        name for name in dense_mesh.mesh_dim_names if name not in supported_axes
    ]
    if unsupported_axes:
        raise ValueError(
            "Unsupported mesh axis for AutoParallel Muse Glimmer: "
            f"{unsupported_axes}. Supported axes: "
            f"{tuple(supported_axes)}"
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
        solver="approx",
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints(
            _input_constraints(autop._traced_inputs, dense_mesh, global_batch_size)
        )
        autop.add_output_constraints([output_sharding])

        start = time.perf_counter()
        sharding_placement = autop.optimize_placement(verbose=False)
        logger.info(
            "AutoParallelGraph placement search took %.2f seconds",
            time.perf_counter() - start,
        )

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
