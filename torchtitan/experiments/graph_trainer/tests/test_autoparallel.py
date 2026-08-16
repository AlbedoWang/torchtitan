# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.device_type = "cpu"
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelDims:
    dp_replicate_enabled = False
    cp_enabled = False
    pp_enabled = False
    tp_enabled = True
    dp_replicate = 1
    dp_shard = 2

    def __init__(self, *, sparse: bool = False):
        self.sparse = sparse
        self.tp_enabled = not sparse

    def get_optional_mesh(self, name):
        enabled = {"fsdp", "tp"} if not self.sparse else {"efsdp", "ep"}
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.used_fx_path = False
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_parameter_memory_constraint(self, *, low, high):
        pass

    def add_input_constraints(self, constraints):
        pass

    def add_output_constraints(self, constraints):
        pass

    def optimize_placement(self, verbose=False):
        return object()

    def apply_placement_for_fx_module(self, *args, **kwargs):
        self.used_fx_path = True
        self.apply_kwargs = kwargs
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        local_batch_size=2,
        seq_len=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def test_autoparallel_integration_matrix():
    from torchtitan.experiments.graph_trainer.tests.integration_tests import (
        build_graph_trainer_autoparallel_h100_test_list,
        build_graph_trainer_autoparallel_test_list,
    )

    suites = {
        "default": build_graph_trainer_autoparallel_test_list(),
        "h100": build_graph_trainer_autoparallel_h100_test_list(),
    }

    assert [test.test_name for test in suites["default"]] == [
        "autoparallel_llama3_fsdp_tp"
    ]
    assert [test.test_name for test in suites["h100"]] == [
        "autoparallel_deepseek_v3_efsdp_ep"
    ]
    assert all(test.ngpu == 4 for tests in suites.values() for test in tests)


def _assert_validated_autoparallel_defaults(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    assert {
        "enable_autoparallel": compile_config.enable_autoparallel,
        "use_autoparallel_defaults": compile_config.use_autoparallel_defaults,
        "mode": compile_config.mode,
        "backend": compile_config.backend,
        "memory_policy": compile_config.memory_policy,
        "pass_pipeline": compile_config.pass_pipeline,
        "inductor_compilation": compile_config.inductor_compilation,
        "numerics_changing_optim": compile_config.numerics_changing_optim,
        "enable_fsdp_ag_rs_overlap": compile_config.enable_fsdp_ag_rs_overlap,
        "enable_fsdp_dense_region_overlap": (
            compile_config.enable_fsdp_dense_region_overlap
        ),
        "disable_passes": compile_config.disable_passes,
    } == {
        "enable_autoparallel": True,
        "use_autoparallel_defaults": True,
        "mode": "aot_fx_trace",
        "backend": "aot_eager",
        "memory_policy": "eager",
        "pass_pipeline": "default",
        "inductor_compilation": "full",
        "numerics_changing_optim": False,
        "enable_fsdp_ag_rs_overlap": False,
        "enable_fsdp_dense_region_overlap": False,
        "disable_passes": ["cudagraph_pass"],
    }


def test_autoparallel_config_validation_and_defaults():
    with pytest.raises(ValueError, match="only supports --compile.mode aot_fx_trace"):
        GraphTrainerCompileConfig(
            mode="jit",
            enable_autoparallel=True,
        )

    regular_compile_config = GraphTrainerCompileConfig()
    assert regular_compile_config.memory_policy == "default"
    assert regular_compile_config.inductor_compilation == "regional"
    assert regular_compile_config.disable_passes == []

    compile_config = GraphTrainerCompileConfig(enable_autoparallel=True)
    _assert_validated_autoparallel_defaults(compile_config)

    already_disabled = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        disable_passes=["cudagraph_pass"],
    )
    assert already_disabled.disable_passes == ["cudagraph_pass"]

    custom_compile_config = GraphTrainerCompileConfig(
        backend="custom",
        memory_policy="default",
        pass_pipeline="custom",
        inductor_compilation="regional",
        numerics_changing_optim=True,
        enable_fsdp_ag_rs_overlap=True,
        enable_fsdp_dense_region_overlap=True,
        enable_autoparallel=True,
        use_autoparallel_defaults=False,
    )
    validate_autoparallel_config(custom_compile_config)
    assert custom_compile_config.backend == "custom"
    assert custom_compile_config.memory_policy == "default"
    assert custom_compile_config.pass_pipeline == "custom"
    assert custom_compile_config.inductor_compilation == "regional"
    assert custom_compile_config.numerics_changing_optim
    assert custom_compile_config.enable_fsdp_ag_rs_overlap
    assert custom_compile_config.enable_fsdp_dense_region_overlap
    assert custom_compile_config.disable_passes == []


@pytest.mark.parametrize(
    ("module", "recipe"),
    [
        ("graph_trainer.llama3", "graph_trainer_llama3_8b"),
        ("graph_trainer.deepseek_v3", "graph_trainer_deepseek_v3_debugmodel"),
    ],
)
def test_autoparallel_cli_uses_validated_defaults(module, recipe):
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    config = cast(
        GraphTrainer.Config,
        ConfigManager().parse_args(
            [
                "--module",
                module,
                "--config",
                recipe,
                "--compile.enable_autoparallel",
            ]
        ),
    )

    _assert_validated_autoparallel_defaults(config.compile)


def test_autoparallel_regional_pass_selection_uses_auto_bucketing():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            use_autoparallel_defaults=False,
            enable_async_tensor_parallel=False,
            disable_passes=["cudagraph_pass"],
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.autobucketing_reordering_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass not in pass_fns


def test_autoparallel_full_pass_selection_injects_backend_inductor_configs():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            enable_async_tensor_parallel=False,
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    graph_passes = passes.construct_default_graph_passes(
        traced_result, config, parallel_dims=_FakeParallelDims()
    )
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert pass_fns == [
        passes.eliminate_dead_code_pass,
        passes.canonicalize_graph_pass,
        passes.deduplicate_fsdp_unshard_chains_pass,
        passes.tag_with_memory_policy_pass,
        passes.apply_cpu_offload_pass,
        passes.selective_activation_remat_pass,
        passes.full_inductor_compilation_pass,
    ]
    assert passes.autobucketing_reordering_pass not in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass not in pass_fns
    full_pass = next(
        pass_fn
        for pass_fn in graph_passes
        if getattr(pass_fn, "func", pass_fn) is passes.full_inductor_compilation_pass
    )
    configs = full_pass.keywords["inductor_configs"]
    assert configs["aten_distributed_optimizations.enable_overlap_scheduling"] is True
    assert configs["aten_distributed_optimizations.collective_bucketing"] is True
    assert configs["aten_distributed_optimizations.insert_overlap_deps"] is True
    assert configs["aten_distributed_optimizations.max_compute_pre_fetch"] == 10
    assert configs["reorder_for_peak_memory"] is False
    assert configs["reorder_for_compute_comm_overlap"] is False
    custom_pass = configs["post_grad_custom_post_pass"]
    assert custom_pass.func.__name__ == "aten_autobucketing_reordering_pass"
    assert custom_pass.keywords["configs"].custom_runtime_estimation is not None


def test_autoparallel_uses_eager_sac_collective_policy():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    def apply_policy(*, enable_autoparallel):
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        all_gather = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(tensor, 2, "group"),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        )
        graph.output(wait)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        config = SimpleNamespace(
            compile=SimpleNamespace(
                enable_autoparallel=enable_autoparallel,
                memory_policy="eager",
            )
        )

        tag_with_memory_policy_pass(gm, config=config)
        return {
            node.target: node.meta.get("recompute")
            for node in gm.graph.nodes
            if node.op == "call_function"
        }

    regular_policy = apply_policy(enable_autoparallel=False)
    autoparallel_policy = apply_policy(enable_autoparallel=True)

    assert autoparallel_policy == regular_policy
    assert (
        autoparallel_policy[torch.ops._c10d_functional.all_gather_into_tensor.default]
        is CheckpointPolicy.PREFER_RECOMPUTE
    )


def test_autoparallel_graph_preserves_copied_model_fqns():
    from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False, device="meta")

        def forward(self, x):
            return self.linear(x).relu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

        def forward(self, x):
            return self.layers[0](x)

    autop = AutoParallelGraph(
        Model(),
        lambda: (torch.randn(2, 4),),
        _FakeMesh(("fsdp",), size=1),
    )
    with autop.fake_mode:
        x = torch.empty(2, 4)
        traced = minimal_fx_tracer(
            lambda input_: autop.model(input_), module=autop.model
        )(x)

    fqns = {
        custom["module_fqn"]
        for node in traced.gm.graph.nodes
        if (custom := node.meta.get("custom")) and "module_fqn" in custom
    }
    assert "layers.0" in fqns
    assert "layers.0.linear" in fqns


@pytest.mark.parametrize(
    (
        "model_name",
        "inductor_compilation",
        "fsdp_reshard_after_forward",
        "expected_reshard_after_forward",
    ),
    [
        ("llama", "regional", "always", True),
        ("deepseek", "full", "never", False),
    ],
)
def test_model_autoparallel_uses_fx_module_path_and_resolved_policy(
    model_name,
    inductor_compilation,
    fsdp_reshard_after_forward,
    expected_reshard_after_forward,
):
    class FakeDeepSeekV3Model(torch.nn.Module):
        def __init__(self, config, *, mesh, compute_dtype):
            super().__init__()
            self.model_args = SimpleNamespace(vocab_size=16)

    _FakeAutoParallelGraph.instances.clear()
    compile_config = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        inductor_compilation=inductor_compilation,
        use_autoparallel_defaults=False,
    )
    parallelism = ParallelismConfig(
        fsdp_reshard_after_forward=fsdp_reshard_after_forward
    )

    if model_name == "llama":
        from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))
        parallel_dims = _FakeParallelDims()
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_llama
        extra_patches = ()
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3 import (
            parallelize_autoparallel,
        )

        model = SimpleNamespace(config=SimpleNamespace())
        parallel_dims = _FakeParallelDims(sparse=True)
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_deepseekv3
        extra_patches = (
            patch.object(
                parallelize_autoparallel,
                "_load_autoparallel_dsv3_dependency",
                return_value=(FakeDeepSeekV3Model, lambda model: None),
            ),
        )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
            )
        )
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "apply_compile", lambda model, **_: model
            )
        )
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)

        call_parallelize(
            model,
            parallel_dims=parallel_dims,
            training=_training_config(),
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
        )

    autop = _FakeAutoParallelGraph.instances[0]
    mp_policy = autop.kwargs["mp_policy"]
    assert mp_policy.param_dtype is torch.bfloat16
    assert mp_policy.reduce_dtype is torch.float32
    assert autop.kwargs["reshard_after_forward"] is expected_reshard_after_forward
    assert autop.kwargs.get("dynamic", False) is (model_name == "deepseek")
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path
