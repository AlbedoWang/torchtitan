# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.muse_glimmer import MuseGlimmerModel
from torchtitan.trainer import Trainer

from .sdpa import build_packed_document_attention_masks


def _add_packed_document_attention_masks(
    model_config: MuseGlimmerModel.Config,
    processed: tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    inputs, labels, extra_kwargs = processed
    positions = extra_kwargs.get("positions")
    if not isinstance(positions, torch.Tensor):
        raise ValueError(
            "Muse Glimmer packed-document SDPA training requires positions"
        )
    window_sizes = {
        layer.attention.window_size
        for layer in model_config.layers
        if layer.attention.window_size is not None
    }
    extra_kwargs["attention_masks"] = build_packed_document_attention_masks(
        positions,
        window_sizes,
    )
    return inputs, labels, extra_kwargs


class MuseGlimmerPackedSDPATrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert isinstance(self.model_config, MuseGlimmerModel.Config)
        return _add_packed_document_attention_masks(
            self.model_config,
            super().post_dataloading_process(input_dict, labels),
        )


class GraphTrainerMuseGlimmerPackedSDPATrainer(GraphTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config):
        pass

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert isinstance(self.model_config, MuseGlimmerModel.Config)
        return _add_packed_document_attention_masks(
            self.model_config,
            super().post_dataloading_process(input_dict, labels),
        )
