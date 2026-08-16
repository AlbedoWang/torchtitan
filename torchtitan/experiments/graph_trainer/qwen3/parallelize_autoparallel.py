# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel-based parallelization for Qwen3 MoE."""

import time

import torch
from autoparallel import ForwardInputs
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_moe_ep_regions,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)
from torchtitan.models.common.attention import (
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


def _create_trace_attention_masks(model, positions):
    attention_config = model.config.first_attention
    inner_attention = attention_config.inner_attention
    if not isinstance(inner_attention, FlexAttention.Config):
        raise ValueError("AutoParallel Qwen3 requires FlexAttention")
    batch_size, seq_len = positions.shape
    return create_block_mask(
        and_masks(
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        ),
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=positions.device,
        BLOCK_SIZE=inner_attention.block_size,
        separate_full_blocks=not is_in_batch_invariant_mode(),
    )


def parallelize_autoparallel_qwen3(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph sharding to Qwen3 on an EFSDP+EP mesh."""
    del ac_config
    validate_autoparallel_config(compile_config)

    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Qwen3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Qwen3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Qwen3 does not support PP yet")
    if parallel_dims.tp_enabled:
        raise ValueError("AutoParallel Qwen3 does not support TP yet")

    required_sparse_axes = ("efsdp", "ep")
    missing_sparse_axes = [
        name
        for name in required_sparse_axes
        if parallel_dims.get_optional_mesh(name) is None
    ]
    if missing_sparse_axes:
        raise ValueError(
            "AutoParallel Qwen3 requires EFSDP and EP axes, but missing "
            f"{missing_sparse_axes}"
        )

    sparse_mesh = parallel_dims.get_mesh(list(required_sparse_axes))
    if sparse_mesh.ndim != 2 or sparse_mesh.mesh_dim_names != required_sparse_axes:
        raise ValueError(
            "AutoParallel Qwen3 requires a 2D sparse mesh with EFSDP and EP "
            f"axes, but got mesh axes {sparse_mesh.mesh_dim_names}"
        )

    annotate_moe_ep_regions()
    maybe_register_blockmask_pytree_node()
    x_sharding = (Shard(0), Shard(0))
    input_constraints = None
    global_batch_size = training.global_batch_size
    if global_batch_size < 0:
        dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        global_batch_size = training.local_batch_size * dp_degree

    def input_fn():
        nonlocal input_constraints
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
        attention_masks = _create_trace_attention_masks(model, positions)
        mask_tensors = torch.utils._pytree.tree_leaves(attention_masks)
        input_constraints = [x_sharding] * (2 + len(mask_tensors))
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
    with AutoParallelGraph(
        model,
        input_fn,
        sparse_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        dynamic=True,
    ) as autop:
        if input_constraints is None:
            raise RuntimeError("AutoParallel Qwen3 input tracing did not run")
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints(input_constraints)
        autop.add_output_constraints([x_sharding])

        start = time.time()
        placement = autop.optimize_placement(verbose=False)
        logger.info(
            "AutoParallelGraph Qwen3 placement took %.2f seconds",
            time.time() - start,
        )
        parallel_model = autop.apply_placement_for_fx_module(
            placement,
            compile_config=compile_config,
        )

    return apply_compile(
        parallel_model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
