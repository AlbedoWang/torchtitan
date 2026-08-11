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
(aten::div.Tensor_mode). This adapter preserves the current TorchTitan model
dimensions and routing settings when constructing AutoParallel's test model.
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
            make_dsv3_config,
        )
    except ImportError as exc:
        raise ImportError(
            "AutoParallel graph_trainer DeepSeek V3 currently depends on "
            "autoparallel._testing.models.dsv3. Move that model and annotation "
            "helper into a supported AutoParallel namespace before treating this "
            "route as a stable production dependency."
        ) from exc
    return (
        DeepSeekV3Model,
        annotate_deepseekv3_for_graph_trainer,
        build_moe_mesh,
        make_dsv3_config,
    )


def _to_autoparallel_dsv3_config(config, make_dsv3_config):
    """Translate the current TorchTitan DSv3 config to AP's test-model config."""
    if hasattr(config, "rope"):
        return config
    if getattr(config, "mtp_layers", None):
        raise ValueError("AutoParallel DeepSeek V3 does not support MTP layers")

    layers = list(config.layers)
    if not layers:
        raise ValueError("AutoParallel DeepSeek V3 requires at least one layer")
    n_dense_layers = sum(layer.moe is None for layer in layers)
    if any(layer.moe is not None for layer in layers[:n_dense_layers]) or any(
        layer.moe is None for layer in layers[n_dense_layers:]
    ):
        raise ValueError(
            "AutoParallel DeepSeek V3 requires dense layers before MoE layers"
        )
    if n_dense_layers == 0 or n_dense_layers == len(layers):
        raise ValueError("AutoParallel DeepSeek V3 requires both dense and MoE layers")

    first_attention = layers[0].attention
    first_moe = layers[n_dense_layers].moe
    assert first_moe is not None
    routed_experts = first_moe.routed_experts
    inner_experts = routed_experts.inner_experts
    shared_experts = first_moe.shared_experts
    if shared_experts is None:
        raise ValueError("AutoParallel DeepSeek V3 requires shared experts")
    shared_hidden_dim = shared_experts.w1.out_features
    if shared_hidden_dim % inner_experts.hidden_dim != 0:
        raise ValueError(
            "DeepSeek V3 shared-expert hidden size must be a multiple of the "
            "routed-expert hidden size"
        )

    rope = first_attention.rope
    ap_config = make_dsv3_config(
        dim=config.dim,
        vocab_size=config.vocab_size,
        n_layers=len(layers),
        n_dense_layers=n_dense_layers,
        n_heads=first_attention.n_heads,
        q_lora_rank=first_attention.q_lora_rank,
        kv_lora_rank=first_attention.kv_lora_rank,
        qk_nope_head_dim=first_attention.qk_nope_head_dim,
        qk_rope_head_dim=first_attention.qk_rope_head_dim,
        v_head_dim=first_attention.v_head_dim,
        mscale=first_attention.mscale,
        dense_hidden_dim=layers[0].feed_forward.w1.out_features,
        moe_hidden_dim=inner_experts.hidden_dim,
        num_experts=first_moe.num_experts,
        num_shared_experts=shared_hidden_dim // inner_experts.hidden_dim,
        top_k=first_moe.router.top_k,
        score_func=first_moe.router.score_func,
        route_norm=first_moe.router.route_norm,
        score_before_experts=getattr(
            routed_experts.token_dispatcher, "score_before_experts", False
        ),
        max_seq_len=rope.max_seq_len,
        rope_theta=rope.theta,
        rope_factor=rope.rope_factor,
        beta_fast=rope.beta_fast,
        beta_slow=rope.beta_slow,
        original_seq_len=rope.original_seq_len,
        load_balance_coeff=first_moe.load_balance_coeff,
    )
    ap_config.norm.eps = config.norm.eps
    for source_layer, ap_layer in zip(layers, ap_config.layers, strict=True):
        ap_layer.attention_norm.eps = source_layer.attention_norm.eps
        ap_layer.ffn_norm.eps = source_layer.ffn_norm.eps
        if ap_layer.moe is not None:
            assert source_layer.moe is not None
            ap_layer.moe.router.route_scale = source_layer.moe.router.route_scale
    return ap_config


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
    The TorchTitan batch boundary shards over DP axes only. CP and TP are
    regular AutoParallel axes here; TorchTitan's context-parallel attention
    preprocessing is not used on this path.
    """
    validate_autoparallel_config(compile_config)

    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support DDP yet")
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
        make_dsv3_config,
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
    ap_model_config = _to_autoparallel_dsv3_config(model.config, make_dsv3_config)
    with torch.device("meta"):
        ap_model = APDeepSeekV3Model(
            ap_model_config,
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
        return tokens

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
        strategy_radius=(0 if compile_config.autoparallel_placements_load_path else 2),
    )

    annotate_deepseekv3_for_graph_trainer(autop.model)

    with autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

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
