# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for DeepSeek V3.

Uses AutoParallelGraph to apply solver-based SPMD sharding on AutoParallel's
local_map DSv3 model (whose ops the solver supports), then lets graph_trainer
trace and compile the placed model through its normal `aot_fx_trace` train-step
pipeline. AutoParallel's aligned MoE mesh keeps the axes that form EP explicit
so `local_map` can flatten them into one EP group.

The torchtitan DSv3 model is replaced with AutoParallel's DeepSeekV3Model
because the solver doesn't support torchtitan's token_dispatcher ops
(aten::div.Tensor_mode). The two models share the same hierarchical config
layout via duck typing.
"""

import time

import torch
from autoparallel import ForwardInputs
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


def _load_autoparallel_dsv3_dependency():
    """Load the temporary AutoParallel DSv3 integration dependency."""
    try:
        from autoparallel import build_moe_mesh
        from autoparallel._testing.models.dsv3 import (
            annotate_deepseekv3_for_graph_trainer,
            DeepSeekV3Model,
        )
    except ImportError as exc:
        raise ImportError(
            "AutoParallel graph_trainer DeepSeek V3 currently depends on "
            "autoparallel._testing.models.dsv3. Move that model and annotation "
            "helper into a supported AutoParallel namespace before treating this "
            "route as a stable production dependency."
        ) from exc
    return DeepSeekV3Model, annotate_deepseekv3_for_graph_trainer, build_moe_mesh


def _set_torchtitan_fields(parallel_model):
    if hasattr(parallel_model, "layers") and isinstance(
        parallel_model.layers, torch.nn.ModuleDict
    ):
        for block in parallel_model.layers.values():
            block.moe_enabled = hasattr(block, "moe")


def _preserve_moe_attributes(original_model, parallel_model):
    """Preserve MoE attributes (moe_enabled, load_balance_coeff) from original."""

    def get_moe_modules(model):
        moe_modules = []
        if hasattr(model, "layers"):
            blocks = (
                model.layers.values()
                if isinstance(model.layers, torch.nn.ModuleDict)
                else []
            )
            for block in blocks:
                if hasattr(block, "moe"):
                    moe_modules.append(block.moe)
        return moe_modules

    for orig_moe, par_moe in zip(
        get_moe_modules(original_model), get_moe_modules(parallel_model)
    ):
        if hasattr(orig_moe, "moe_enabled"):
            par_moe.moe_enabled = orig_moe.moe_enabled
        if hasattr(orig_moe, "load_balance_coeff"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff


def parallelize_autoparallel_deepseekv3(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph SPMD sharding to DeepSeek V3.

    Returns a sharded model carrying AutoParallel train-step metadata.
    The TorchTitan batch boundary shards over DP axes only. TP is a regular
    AutoParallel axis, while TorchTitan context parallelism remains unsupported.
    """
    validate_autoparallel_config(compile_config)

    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support PP yet")

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
    (
        APDeepSeekV3Model,
        annotate_deepseekv3_for_graph_trainer,
        build_moe_mesh,
    ) = _load_autoparallel_dsv3_dependency()

    ap_mesh, moe_roles = build_moe_mesh(
        dp_replicate=parallel_dims.dp_replicate,
        dp_shard=parallel_dims.dp_shard,
        cp=parallel_dims.cp,
        tp=parallel_dims.tp,
        ep=parallel_dims.ep,
        device_type=device_type,
    )

    # Use AutoParallel's DSv3 model: torchtitan's token_dispatcher uses
    # aten::div.Tensor_mode which the AP solver doesn't support yet.
    # The AP model accepts torchtitan's config via duck typing (same
    # hierarchical attribute paths).
    with torch.device("meta"):
        ap_model = APDeepSeekV3Model(
            model.config,
            mesh=ap_mesh,
            roles=moe_roles,
            compute_dtype=param_dtype,
        )

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            ap_model.model_args.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        return ForwardInputs(args=(tokens,), kwargs={"positions": positions})

    data_parallel_axes = {
        "dp_replicate",
        "dp_shard_mod_ep",
        "dp_shard_in_ep",
    }
    assert ap_mesh.mesh_dim_names is not None
    x_sharding = tuple(
        Shard(0) if name in data_parallel_axes else Replicate()
        for name in ap_mesh.mesh_dim_names
    )

    autop = AutoParallelGraph(
        ap_model,
        input_fn,
        ap_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        dynamic=True,
        solver=compile_config.autoparallel_solver,
    )

    annotate_deepseekv3_for_graph_trainer(autop.model)

    with autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding, x_sharding])
        autop.add_output_constraints([x_sharding])

        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

        # The output is batch-sharded over DP axes and replicated over TP, so
        # graph_trainer can pair each rank's local logits with local labels.
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
        )

    _set_torchtitan_fields(parallel_mod)
    _preserve_moe_attributes(ap_model, parallel_mod)

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    return model
