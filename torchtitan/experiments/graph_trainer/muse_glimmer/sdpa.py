# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductAttention,
)
from torchtitan.models.muse_glimmer.model import _window_mask_key
from torchtitan.protocols.model_spec import ModelSpec


class MuseGlimmerPackedDocumentSDPA(ScaledDotProductAttention):
    """SDPA backend for Muse Glimmer packed-document tensor masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductAttention.Config):
        pass

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: AttentionMasksType | torch.Tensor | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is None:
            return super().forward(
                q_BLNH,
                k_BLNH,
                v_BLNH,
                scale=scale,
                enable_gqa=enable_gqa,
                is_causal=is_causal,
                **kwargs,
            )
        if not isinstance(attention_masks, torch.Tensor):
            raise TypeError(
                "MuseGlimmerPackedDocumentSDPA requires a Tensor attention mask, "
                f"got {type(attention_masks).__name__}"
            )
        if attention_masks.dtype is not torch.bool:
            raise ValueError(
                "MuseGlimmerPackedDocumentSDPA requires a boolean attention mask"
            )

        q_BNLH, k_BNLH, v_BNLH = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out_BNLH = F.scaled_dot_product_attention(
                q_BNLH,
                k_BNLH,
                v_BNLH,
                attn_mask=attention_masks,
                scale=scale,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        return out_BNLH.transpose(1, 2)


def build_packed_document_attention_masks(
    positions_BS: torch.Tensor,
    window_sizes: Iterable[int],
) -> dict[str, torch.Tensor]:
    """Build [B, 1, S, S] causal masks from packed-document positions."""
    if positions_BS.ndim != 2:
        raise ValueError(
            "Packed-document positions must have shape [batch, sequence], "
            f"got {tuple(positions_BS.shape)}"
        )

    seq_len = positions_BS.shape[1]
    document_ids_BS = torch.cumsum((positions_BS == 0).int(), dim=1) - 1
    same_document_BSS = document_ids_BS[:, :, None] == document_ids_BS[:, None, :]
    query_index_SS = torch.arange(seq_len, device=positions_BS.device)[:, None]
    key_index_SS = torch.arange(seq_len, device=positions_BS.device)[None, :]
    causal_SS = query_index_SS >= key_index_SS
    global_mask_B1SS = (same_document_BSS & causal_SS).unsqueeze(1)

    masks = {_window_mask_key(None): global_mask_B1SS}
    for window_size in sorted(set(window_sizes)):
        window_SS = query_index_SS - key_index_SS < window_size
        masks[_window_mask_key(window_size)] = global_mask_B1SS & window_SS
    return masks


def set_model_spec_packed_document_sdpa(model_spec: ModelSpec) -> ModelSpec:
    """Replace a fresh Muse Glimmer model spec's inner attention configs."""
    for layer in model_spec.model.layers:
        layer.attention.inner_attention = MuseGlimmerPackedDocumentSDPA.Config()
    return model_spec
