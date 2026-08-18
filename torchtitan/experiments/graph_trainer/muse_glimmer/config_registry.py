# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, replace
from typing import Literal

from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import (
    model_registry as muse_glimmer_model_registry,
)
from torchtitan.models.muse_glimmer.config_registry import (
    muse_glimmer_30b,
    muse_glimmer_debugmodel,
)
from torchtitan.tools.profiler import Profiler

from . import model_registry
from .sdpa import set_model_spec_packed_document_sdpa
from .trainer import (
    GraphTrainerMuseGlimmerPackedSDPATrainer,
    MuseGlimmerPackedSDPATrainer,
)


C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_TRAIN_SHARDS = 1024
C4_STAGED_SHARDS = 16
MUSE_GLIMMER_C4_DATASET = "muse_glimmer_c4_pinned_offline"


def _load_pinned_offline_c4(dataset_path: str):
    if dataset_path != C4_REPO:
        raise ValueError(f"Expected C4 path {C4_REPO!r}, got {dataset_path!r}")
    local_files = [
        hf_hub_download(
            repo_id=C4_REPO,
            filename=(f"en/c4-train.{shard:05d}-of-{C4_TRAIN_SHARDS:05d}.json.gz"),
            repo_type="dataset",
            revision=C4_REVISION,
            local_files_only=True,
        )
        for shard in range(C4_STAGED_SHARDS)
    ]
    return load_dataset(
        "json",
        data_files={"train": local_files},
        features=Features(
            {
                "text": Value("string"),
                "timestamp": Value("string"),
                "url": Value("string"),
            }
        ),
        split="train",
        streaming=True,
    )


DATASETS[MUSE_GLIMMER_C4_DATASET] = DatasetConfig(
    path=C4_REPO,
    loader=_load_pinned_offline_c4,
    sample_processor=lambda sample: sample["text"],
)


def _copy_config(config, config_type):
    return config_type(**{f.name: getattr(config, f.name) for f in fields(config)})


def _muse_glimmer_30b_c4_base(
    attention_backend: Literal["flex", "sdpa"],
):
    config = muse_glimmer_30b()
    if attention_backend == "sdpa":
        config.model_spec = set_model_spec_packed_document_sdpa(
            muse_glimmer_model_registry("30B", attn_backend="flex")
        )
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    config.dataloader = replace(
        config.dataloader,
        dataset=MUSE_GLIMMER_C4_DATASET,
    )
    config.metrics = MetricsProcessor.Config(
        log_freq=1,
        enable_tensorboard=True,
    )
    config.training = TrainingConfig(
        local_batch_size=1,
        global_batch_size=4,
        seq_len=256,
        steps=10,
    )
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=4,
        tensor_parallel_degree=2,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    config.checkpoint = replace(config.checkpoint, enable=False)
    config.activation_checkpoint = SelectiveAC.Config()
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=10,
        profiler_repeat=1,
        profiler_warmup=1,
        profiler_active=1,
    )
    config.debug = replace(config.debug, seed=42)
    return config


def _muse_glimmer_30b_sdpa_c4_base():
    return _muse_glimmer_30b_c4_base("sdpa")


def _muse_glimmer_30b_flex_c4_base():
    return _muse_glimmer_30b_c4_base("flex")


def graph_trainer_muse_glimmer_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(muse_glimmer_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def muse_glimmer_30b_sdpa_c4_torchtitan_4x2() -> MuseGlimmerPackedSDPATrainer.Config:
    config = _muse_glimmer_30b_sdpa_c4_base()
    return _copy_config(config, MuseGlimmerPackedSDPATrainer.Config)


def graph_trainer_muse_glimmer_30b_sdpa_c4_4x2() -> (
    GraphTrainerMuseGlimmerPackedSDPATrainer.Config
):
    base = _muse_glimmer_30b_sdpa_c4_base()
    config = to_graph_trainer_config(base, model_registry)
    validated_ap_compile = GraphTrainerCompileConfig(
        enable=True,
        enable_autoparallel=True,
    )
    config.compile = replace(
        validated_ap_compile,
        enable_autoparallel=False,
    )
    return _copy_config(config, GraphTrainerMuseGlimmerPackedSDPATrainer.Config)


def graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2() -> (
    GraphTrainerMuseGlimmerPackedSDPATrainer.Config
):
    base = _muse_glimmer_30b_sdpa_c4_base()
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        enable_autoparallel=True,
    )
    return _copy_config(config, GraphTrainerMuseGlimmerPackedSDPATrainer.Config)


def graph_trainer_muse_glimmer_30b_flex_c4_4x2() -> GraphTrainer.Config:
    base = _muse_glimmer_30b_flex_c4_base()
    config = to_graph_trainer_config(base, model_registry)
    validated_ap_compile = GraphTrainerCompileConfig(
        enable=True,
        enable_autoparallel=True,
    )
    config.compile = replace(
        validated_ap_compile,
        enable_autoparallel=False,
    )
    return config


def graph_trainer_muse_glimmer_30b_flex_c4_autoparallel_4x2() -> GraphTrainer.Config:
    base = _muse_glimmer_30b_flex_c4_base()
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        enable_autoparallel=True,
    )
    return config
