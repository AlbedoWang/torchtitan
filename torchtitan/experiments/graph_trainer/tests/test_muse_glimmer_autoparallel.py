# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from types import SimpleNamespace

import torch
from autoparallel import ForwardInputs
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.experiments.graph_trainer.muse_glimmer import model_registry
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    graph_trainer_muse_glimmer_30b_flex_c4_4x2,
    graph_trainer_muse_glimmer_30b_flex_c4_autoparallel_4x2,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.parallelize_autoparallel import (
    _attention_masks,
    _input_constraints,
)
from torchtitan.models.common.attention import FlexAttention


def _debug_flex_model():
    spec = model_registry("debugmodel", attn_backend="flex")
    with torch.device("meta"):
        return spec.model.build()


def test_flex_c4_configs_differ_only_in_autoparallel_enablement():
    manual = graph_trainer_muse_glimmer_30b_flex_c4_4x2()
    autoparallel = graph_trainer_muse_glimmer_30b_flex_c4_autoparallel_4x2()

    assert isinstance(
        manual.model_spec.model.first_attention.inner_attention,
        FlexAttention.Config,
    )
    assert isinstance(
        autoparallel.model_spec.model.first_attention.inner_attention,
        FlexAttention.Config,
    )
    assert manual.compile == replace(
        autoparallel.compile,
        enable_autoparallel=False,
    )
    assert manual.model_spec.flavor == autoparallel.model_spec.flavor
    assert (
        manual.model_spec.model.vocab_size == autoparallel.model_spec.model.vocab_size
    )
    assert len(manual.model_spec.model.layers) == len(
        autoparallel.model_spec.model.layers
    )
    assert [
        layer.attention.window_size for layer in manual.model_spec.model.layers
    ] == [layer.attention.window_size for layer in autoparallel.model_spec.model.layers]
    assert manual.training == autoparallel.training
    assert manual.parallelism == autoparallel.parallelism
    assert manual.loss == autoparallel.loss
    assert manual.dataloader == autoparallel.dataloader
    assert manual.optimizer == autoparallel.optimizer
    assert manual.lr_scheduler == autoparallel.lr_scheduler


def test_flex_masks_and_2d_input_constraints_preserve_batch_sharding():
    positions_cases = (
        torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
        torch.arange(8).repeat(2, 1),
        torch.tensor(
            [
                [0, 1, 2, 0, 1, 2, 3, 4],
                [0, 1, 0, 1, 2, 0, 1, 2],
            ]
        ),
    )
    model = _debug_flex_model()
    window_sizes = {
        layer.attention.window_size
        for layer in model.config.layers
        if layer.attention.window_size is not None
    }
    expected_mask_spec = None
    for positions in positions_cases:
        masks = _attention_masks(model, positions, window_sizes)
        reference_masks = model.get_attention_masks(positions)

        assert set(masks) == {"global", *(f"swa_{size}" for size in window_sizes)}
        assert all(isinstance(mask, BlockMask) for mask in masks.values())
        flat_masks, mask_spec = torch.utils._pytree.tree_flatten(masks)
        flat_reference_masks, reference_mask_spec = torch.utils._pytree.tree_flatten(
            reference_masks
        )
        assert flat_masks
        assert all(isinstance(value, torch.Tensor) for value in flat_masks)
        assert mask_spec == reference_mask_spec
        assert len(flat_masks) == len(flat_reference_masks)
        if expected_mask_spec is None:
            expected_mask_spec = mask_spec
        else:
            assert mask_spec == expected_mask_spec
        for key, mask in masks.items():
            reference = reference_masks[key]
            assert mask.BLOCK_SIZE == reference.BLOCK_SIZE
            assert mask.seq_lengths == reference.seq_lengths
            for attribute in BlockMask._TENSOR_ATTRS:
                value = getattr(mask, attribute)
                reference_value = getattr(reference, attribute)
                if value is None or reference_value is None:
                    assert value is reference_value
                else:
                    torch.testing.assert_close(value, reference_value)

        global_batch_size = positions.shape[0]
        traced_inputs = ForwardInputs(
            args=(torch.zeros_like(positions),),
            kwargs={"positions": positions, "attention_masks": masks},
        )
        constraints = _input_constraints(
            traced_inputs,
            SimpleNamespace(mesh_dim_names=("fsdp", "tp")),
            global_batch_size,
        )

        assert constraints
        assert set(constraints) == {(Shard(0), Replicate())}
