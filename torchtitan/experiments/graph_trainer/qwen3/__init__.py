# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerQwen3Model
from .parallelize import parallelize_qwen3


def _parallelize_fn(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from .parallelize_autoparallel import parallelize_autoparallel_qwen3

        return parallelize_autoparallel_qwen3(
            model, compile_config=compile_config, **kwargs
        )
    return parallelize_qwen3(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> ModelSpec:
    kwargs = {}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    base = build_decoder_config_for_backend(
        qwen3_configs[flavor], attn_backend, **kwargs
    )
    config = GraphTrainerQwen3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
