# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel helpers for graph_trainer's ``aot_fx_trace`` path."""

from functools import partial

import torch
import torch.nn as nn
from autoparallel.api import AutoParallel
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config
from autoparallel.graph_passes.auto_bucketing import (
    _runtime_estimation_ms,
    aten_autobucketing_config,
    aten_autobucketing_reordering_pass,
)
from autoparallel.graph_passes.debug_helpers import make_custom_runtime_estimation
from autoparallel.module_construction import make_parallel_module
from torch._functorch._aot_autograd.fx_utils import get_plain_input_and_grad_nodes
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.export._tree_utils import reorder_kwargs

from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


def _autoparallel_inductor_configs(mesh: DeviceMesh) -> dict:
    """Return the Inductor settings used by AutoParallel's Llama example.

    Keep the auto-bucketing state local to this compile instead of mutating
    AutoParallel's process-global config class. Trace emission is disabled here
    because GraphTrainer records compiler and Kineto traces explicitly.
    """
    set_nccl_topo_config(detect_nccl_topo_config(mesh))
    autobucketing_config = aten_autobucketing_config()
    autobucketing_config.custom_runtime_estimation = make_custom_runtime_estimation(
        mesh
    )
    autobucketing_config.save_trace = False

    return {
        "aten_distributed_optimizations.enable_overlap_scheduling": True,
        "aten_distributed_optimizations.collective_bucketing": True,
        "aten_distributed_optimizations.insert_overlap_deps": False,
        "aten_distributed_optimizations.max_compute_pre_fetch": 10,
        # The AP estimates are optimistic on H100 (Muse Glimmer 4x8x2 fsdp
        # reduce-scatter: estimated comm/compute ratio ~0.6x of measured), so
        # only half of each compute estimate counts toward hiding a collective.
        "aten_distributed_optimizations.compute_overlap_multipler": 0.5,
        # Inductor's own overlap pass uses the same AP estimates as the AP pass.
        "aten_distributed_optimizations.custom_runtime_estimation": partial(
            _runtime_estimation_ms, autobucketing_config.custom_runtime_estimation
        ),
        "reorder_for_peak_memory": False,
        "reorder_for_compute_comm_overlap": False,
        "post_grad_custom_post_pass": partial(
            aten_autobucketing_reordering_pass,
            configs=autobucketing_config,
        ),
    }


def _local_tensor_with_autograd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _get_raw_module_tensor(
    module: nn.Module, fqn: str, *, is_buffer: bool
) -> torch.Tensor:
    *prefix, name = fqn.split(".")
    owner = module.get_submodule(".".join(prefix)) if prefix else module
    tensor_dict = owner._buffers if is_buffer else owner._parameters
    tensor = tensor_dict.get(name)
    if tensor is None:
        kind = "buffer" if is_buffer else "parameter"
        raise AttributeError(f"{fqn!r} is not a registered {kind}")
    return tensor


class AutoParallelGraph(AutoParallel):
    """AutoParallel variant for graph_trainer's ``aot_fx_trace`` pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        annotate_module_fqns(self.model)

    def apply_placement_for_fx_module(
        self,
        sharding_placement=None,
        *,
        compile_config: GraphTrainerCompileConfig,
    ) -> nn.Module:
        """Return an AOT-backed parallel module for graph_trainer tracing.

        This keeps loss in graph_trainer's normal train step, which consumes
        the local AutoParallel outputs as plain tensors.
        """
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )
        parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors,
            fw_compiler=self.compiler_fn,
            bw_compiler=self.compiler_fn,
        )

        graph_param_fqns = list(self.joint_with_descriptors.params_spec)
        graph_buffer_fqns = list(self.joint_with_descriptors.buffers_spec)

        # Number of plain (user) input placeholders the traced graph expects,
        # used in forward to detect whether runtime inputs arrived as args alone
        # or split across args + kwargs. This must match the count baked into the
        # graph (boxed_args = params + buffers + flat_args), so it is read from
        # the graph's input nodes rather than via autoparallel's private
        # _compute_expected_inputs (which had an unstable signature across
        # versions and was removed from autoparallel main).
        num_expected_inputs = len(get_plain_input_and_grad_nodes(self.gm.graph))
        trace_in_spec = torch.utils._pytree.tree_flatten(
            (tuple(self._traced_inputs.args), self._traced_inputs.kwargs)
        )[1]
        has_traced_kwargs = bool(self._traced_inputs.kwargs)

        def forward(self, *args, **kwargs):
            if has_traced_kwargs or kwargs:
                if kwargs:
                    kwargs = reorder_kwargs(kwargs, trace_in_spec)
                flat_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            else:
                flat_args, _ = torch.utils._pytree.tree_flatten(args)
                if len(flat_args) != num_expected_inputs:
                    flat_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            params = [
                _local_tensor_with_autograd(
                    _get_raw_module_tensor(self, fqn, is_buffer=False)
                )
                for fqn in graph_param_fqns
            ] + [
                _local_tensor_with_autograd(
                    _get_raw_module_tensor(self, fqn, is_buffer=True)
                )
                for fqn in graph_buffer_fqns
            ]
            if has_traced_kwargs:
                output = parallel_model_fn(*params, *args, **kwargs)
            else:
                output = parallel_model_fn([*params, *flat_args])
            del params
            return output

        return make_parallel_module(
            self.model,
            sharded_param_dict,
            sharded_buffer_dict,
            forward_fn=forward,
        )
