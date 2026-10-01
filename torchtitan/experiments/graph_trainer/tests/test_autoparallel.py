# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.device_type = "cpu"
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelismContext:
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
        enabled = {"dp_shard", "tp"} if not self.sparse else {"edp_shard", "ep"}
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.input_fn = input_fn
        self.mesh = mesh
        self.kwargs = kwargs
        self.used_fx_path = False
        self.optimize_calls = 0
        self.sharding_optimizer = SimpleNamespace(
            load_placements=self._load_placements,
            save_placements=self._save_placements,
        )
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_parameter_memory_constraint(self, *, low, high):
        pass

    def add_input_constraints(self, constraints):
        self.input_constraints = constraints

    def add_output_constraints(self, constraints):
        self.output_constraints = constraints

    def optimize_placement(self, verbose=False):
        self.optimize_calls += 1
        return object()

    def _load_placements(self, path):
        self.loaded_path = path
        return object()

    def _save_placements(self, path):
        self.saved_path = path

    def apply_placement_for_fx_module(self, *args, **kwargs):
        self.used_fx_path = True
        self.apply_kwargs = kwargs
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=2 * 8,
        max_context_length=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def _build_autoparallel_a2a_linear_graph(
    *,
    with_backward=False,
    a2a_as_weight=False,
    a2a_fqn="layers.0.attention.wo",
    linear_op=torch.ops.aten.mm.default,
    with_post_linear_views=True,
):
    """Build the A2A -> WO -> residual shape from the real Llama graph."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")

    embedding_a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(x, 2, 0, "tp"),
    )
    embedding_output_a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(embedding_a2a, 2, 1, "dp"),
    )
    qkv = graph.call_function(
        torch.ops.aten.mm.default, args=(embedding_output_a2a, weight)
    )
    a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(qkv, 2, 1, "tp"),
    )
    pre_wo_unsqueeze = graph.call_function(
        torch.ops.aten.unsqueeze.default,
        args=(a2a, 3),
    )
    pre_wo_permute = graph.call_function(
        torch.ops.aten.permute.default,
        args=(pre_wo_unsqueeze, [0, 1, 3, 2]),
    )
    pre_wo_reshape = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(pre_wo_permute, [1, 1024, 4096]),
    )
    pre_wo = graph.call_function(
        torch.ops.aten.squeeze.dim,
        args=(pre_wo_reshape, 0),
    )
    wo_args = (x, pre_wo) if a2a_as_weight else (pre_wo, weight)
    wo = graph.call_function(linear_op, args=wo_args)
    post_wo_nodes = []
    post_wo = wo
    if with_post_linear_views:
        post_wo_unsqueeze = graph.call_function(
            torch.ops.aten.unsqueeze.default,
            args=(wo, 0),
        )
        post_wo_reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(post_wo_unsqueeze, [2, 512, 1, 4096]),
        )
        post_wo_permute = graph.call_function(
            torch.ops.aten.permute.default,
            args=(post_wo_reshape, [0, 1, 3, 2]),
        )
        post_wo = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(post_wo_permute, [2, 512, 4096]),
        )
        post_wo_nodes = [
            post_wo_unsqueeze,
            post_wo_reshape,
            post_wo_permute,
            post_wo,
        ]
    residual_add = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(embedding_output_a2a, post_wo),
    )
    ffn_w1 = graph.call_function(
        torch.ops.aten.mm.default,
        args=(residual_add, weight),
    )
    ffn_w3 = graph.call_function(torch.ops.aten.mm.default, args=(ffn_w1, weight))
    ffn_w2 = graph.call_function(torch.ops.aten.mm.default, args=(ffn_w3, weight))

    output = ffn_w2
    if with_backward:
        backward = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(residual_add, 2)
        )
        backward.meta["autograd_backward"] = True
        output = (output, backward)
    graph.output(output)

    fqns = {
        embedding_a2a: "tok_embeddings",
        embedding_output_a2a: "tok_embeddings",
        qkv: "layers.0.attention.qkv_linear.wqkv",
        a2a: a2a_fqn,
        pre_wo_unsqueeze: "layers.0.attention.wo",
        pre_wo_permute: "layers.0.attention.wo",
        pre_wo_reshape: "layers.0.attention.wo",
        pre_wo: "layers.0.attention.wo",
        wo: "layers.0.attention.wo",
        residual_add: "layers.0",
        ffn_w1: "layers.0.feed_forward.w1",
        ffn_w3: "layers.0.feed_forward.w3",
        ffn_w2: "layers.0.feed_forward.w2",
    }
    fqns.update({node: "layers.0.attention.wo" for node in post_wo_nodes})
    for node, fqn in fqns.items():
        node.meta["custom"] = {_MODULE_FQN: fqn}

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, SimpleNamespace(
        embedding_a2a=embedding_a2a,
        embedding_output_a2a=embedding_output_a2a,
        qkv=qkv,
        a2a=a2a,
        pre_wo=pre_wo,
        wo=wo,
        post_wo=post_wo,
        residual_add=residual_add,
        ffn_w1=ffn_w1,
        ffn_w3=ffn_w3,
        ffn_w2=ffn_w2,
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
        "autoparallel_deepseek_v3_edp_shard_ep"
    ]
    assert all(test.ngpu == 4 for tests in suites.values() for test in tests)


def _assert_validated_autoparallel_defaults(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    assert {
        "enable_autoparallel": compile_config.enable_autoparallel,
        "use_autoparallel_defaults": compile_config.use_autoparallel_defaults,
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
        "backend": "aot_eager",
        "memory_policy": "eager",
        "pass_pipeline": "default",
        "inductor_compilation": "full",
        "numerics_changing_optim": False,
        "enable_fsdp_ag_rs_overlap": False,
        "enable_fsdp_dense_region_overlap": False,
        "disable_passes": ["cuda_graph_pass"],
    }


def test_autoparallel_config_validation_and_defaults():
    from torchtitan.experiments.graph_trainer.configs import (
        validate_autoparallel_config,
    )

    regular_compile_config = GraphTrainerCompileConfig()
    assert regular_compile_config.memory_policy == "default"
    assert regular_compile_config.inductor_compilation == "regional"
    assert regular_compile_config.disable_passes == []

    compile_config = GraphTrainerCompileConfig(enable_autoparallel=True)
    _assert_validated_autoparallel_defaults(compile_config)

    already_disabled = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        disable_passes=["cuda_graph_pass"],
    )
    assert already_disabled.disable_passes == ["cuda_graph_pass"]

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

    with pytest.raises(ValueError, match="save and load paths are mutually exclusive"):
        GraphTrainerCompileConfig(
            enable_autoparallel=True,
            autoparallel_placements_save_path="save.json",
            autoparallel_placements_load_path="load.json",
        )
    with pytest.raises(ValueError, match="require compile.enable_autoparallel"):
        GraphTrainerCompileConfig(autoparallel_placements_load_path="load.json")


def _pass_selection_config(**compile_kwargs):
    return SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            enable_async_tensor_parallel=False,
            **compile_kwargs,
        ),
        model=SimpleNamespace(layers=[object()]),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )


def _traced_result():
    return SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )


def test_autoparallel_regional_pass_selection_uses_auto_bucketing():
    from torchtitan.experiments.graph_trainer import passes

    config = _pass_selection_config(
        use_autoparallel_defaults=False,
        inductor_compilation="regional",
        disable_passes=["cuda_graph_pass"],
    )

    graph_passes = passes.construct_default_graph_passes(_traced_result(), config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.autobucketing_reordering_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass not in pass_fns


def test_autoparallel_full_pass_selection_injects_backend_inductor_configs():
    from torchtitan.experiments.graph_trainer import passes

    config = _pass_selection_config()

    graph_passes = passes.construct_default_graph_passes(
        _traced_result(), config, parallelism_context=_FakeParallelismContext()
    )
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.autobucketing_reordering_pass not in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass not in pass_fns
    assert pass_fns[-1] is passes.full_inductor_compilation_pass
    configs = graph_passes[-1].keywords["inductor_configs"]
    assert configs["aten_distributed_optimizations.enable_overlap_scheduling"] is True
    assert configs["aten_distributed_optimizations.collective_bucketing"] is True
    assert configs["aten_distributed_optimizations.insert_overlap_deps"] is False
    assert configs["aten_distributed_optimizations.max_compute_pre_fetch"] == 10
    assert configs["aten_distributed_optimizations.compute_overlap_multipler"] == 0.5
    assert configs["reorder_for_peak_memory"] is False
    assert configs["reorder_for_compute_comm_overlap"] is False
    custom_pass = configs["post_grad_custom_post_pass"]
    assert custom_pass.func.__name__ == "aten_autobucketing_reordering_pass"
    assert custom_pass.keywords["configs"].custom_runtime_estimation is not None


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
        _FakeMesh(("dp_shard",), size=1),
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

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=16), layers={})
        parallelism_context = _FakeParallelismContext()
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_llama
        extra_patches = ()
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3 import (
            parallelize_autoparallel,
        )

        model = SimpleNamespace(config=SimpleNamespace())
        parallelism_context = _FakeParallelismContext(sparse=True)
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
            parallelism_context=parallelism_context,
            training=_training_config(),
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
            max_num_documents=1,
        )

    autop = _FakeAutoParallelGraph.instances[0]
    mp_policy = autop.kwargs["mp_policy"]
    assert mp_policy.param_dtype is torch.bfloat16
    assert mp_policy.reduce_dtype is torch.float32
    assert autop.kwargs["reshard_after_forward"] is expected_reshard_after_forward
    assert autop.kwargs.get("dynamic", False) is (model_name == "deepseek")
    assert autop.kwargs["solver"] == compile_config.autoparallel_solver
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path


@pytest.mark.parametrize("use_saved_placements", [False, True])
def test_llama_autoparallel_placements_load_or_save(tmp_path, use_saved_placements):
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

    _FakeAutoParallelGraph.instances.clear()
    placement_path = tmp_path / "placements.json"
    compile_config = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        autoparallel_solver="approx",
        autoparallel_placements_load_path=(
            str(placement_path) if use_saved_placements else ""
        ),
        autoparallel_placements_save_path=(
            "" if use_saved_placements else str(placement_path)
        ),
    )

    with (
        patch.object(
            parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
        ),
        patch.object(
            parallelize_autoparallel, "apply_compile", lambda model, **_: model
        ),
    ):
        parallelize_autoparallel.parallelize_autoparallel_llama(
            SimpleNamespace(config=SimpleNamespace(vocab_size=16), layers={}),
            parallelism_context=_FakeParallelismContext(),
            training=_training_config(),
            parallelism=ParallelismConfig(),
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
            max_num_documents=1,
        )

    autop = _FakeAutoParallelGraph.instances[0]
    assert autop.kwargs["solver"] == "approx"
    assert autop.kwargs["strategy_radius"] == (0 if use_saved_placements else 2)
    if use_saved_placements:
        assert autop.optimize_calls == 0
        assert autop.loaded_path == str(placement_path)
    else:
        assert autop.optimize_calls == 1
        assert autop.saved_path == placement_path


@pytest.mark.parametrize(
    "linear_op", [torch.ops.aten.mm.default, torch.ops.aten.linear.default]
)
@pytest.mark.parametrize("a2a_fqn", ["layers.0.attention.wo", "layers.0.attention"])
def test_autoparallel_eager_sac_saves_a2a_linear_boundaries(linear_op, a2a_fqn):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(
        linear_op=linear_op,
        a2a_fqn=a2a_fqn,
    )
    config = SimpleNamespace(
        compile=SimpleNamespace(enable_autoparallel=True, memory_policy="eager")
    )

    tag_with_memory_policy_pass(gm, config=config)

    assert nodes.a2a.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.post_wo.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.embedding_a2a.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert nodes.embedding_output_a2a.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.pre_wo.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert [
        node.meta["recompute"]
        for node in (nodes.qkv, nodes.wo, nodes.ffn_w1, nodes.ffn_w3, nodes.ffn_w2)
    ] == [
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
    ]


@pytest.mark.parametrize("a2a_fqn", ["layers.0.attention.wo", "layers.0.attention"])
def test_autoparallel_eager_sac_requires_a2a_as_linear_activation(a2a_fqn):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(
        a2a_as_weight=True,
        a2a_fqn=a2a_fqn,
    )
    config = SimpleNamespace(
        compile=SimpleNamespace(enable_autoparallel=True, memory_policy="eager")
    )

    tag_with_memory_policy_pass(gm, config=config)

    assert nodes.a2a.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert nodes.post_wo.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE


@pytest.mark.parametrize("node_name", ["a2a", "wo", "residual_add"])
def test_autoparallel_eager_sac_rejects_missing_layer_metadata(node_name):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_fqn="layers.0.attention")
    getattr(nodes, node_name).meta["custom"].pop(_MODULE_FQN)

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == set()


def test_autoparallel_eager_sac_allows_missing_view_metadata():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_fqn="layers.0.attention")
    nodes.post_wo.meta["custom"].pop(_MODULE_FQN)

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == {
        nodes.a2a,
        nodes.post_wo,
    }


@pytest.mark.parametrize("node_name", ["a2a", "wo", "residual_add"])
def test_autoparallel_eager_sac_rejects_cross_layer_boundary(node_name):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_fqn="layers.0.attention")
    getattr(nodes, node_name).meta["custom"][_MODULE_FQN] = "layers.1.attention"

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == set()


@pytest.mark.parametrize("node_name", ["a2a", "post_wo"])
def test_autoparallel_eager_sac_ignores_nonmatching_forward_fanout(node_name):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_fqn="layers.0.attention")
    output = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output):
        gm.graph.call_function(
            torch.ops.aten.alias.default,
            args=(getattr(nodes, node_name),),
        )

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == {
        nodes.a2a,
        nodes.post_wo,
    }


def test_autoparallel_eager_sac_requires_residual_add_consumer():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_fqn="layers.0.attention")
    nodes.residual_add.target = torch.ops.aten.mul.Tensor

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == set()


def test_autoparallel_eager_sac_allows_direct_residual_consumer():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        _find_autoparallel_a2a_linear_save_nodes,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(
        a2a_fqn="layers.0.attention",
        with_post_linear_views=False,
    )

    assert _find_autoparallel_a2a_linear_save_nodes(gm) == {nodes.a2a, nodes.wo}


@pytest.mark.parametrize("a2a_fqn", ["layers.0.attention.wo", "layers.0.attention"])
def test_autoparallel_eager_sac_does_not_rematerialize_a2a_linear(a2a_fqn):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    def apply_passes(*, enable_autoparallel):
        gm, nodes = _build_autoparallel_a2a_linear_graph(
            with_backward=True,
            a2a_fqn=a2a_fqn,
        )
        config = SimpleNamespace(
            compile=SimpleNamespace(
                enable_autoparallel=enable_autoparallel,
                memory_policy="eager",
            )
        )
        tag_with_memory_policy_pass(gm, config=config)
        selective_activation_remat_pass(gm)
        gm.graph.lint()
        return gm, nodes

    regular_gm, regular_nodes = apply_passes(enable_autoparallel=False)
    autoparallel_gm, autoparallel_nodes = apply_passes(enable_autoparallel=True)

    regular_names = {node.name for node in regular_gm.graph.nodes}
    autoparallel_names = {node.name for node in autoparallel_gm.graph.nodes}
    assert regular_nodes.a2a.name + "_recomputed" in regular_names
    assert regular_nodes.wo.name + "_recomputed" in regular_names
    assert regular_nodes.post_wo.name + "_recomputed" in regular_names
    assert autoparallel_nodes.a2a.name + "_recomputed" not in autoparallel_names
    assert autoparallel_nodes.wo.name + "_recomputed" not in autoparallel_names
    assert autoparallel_nodes.post_wo.name + "_recomputed" not in autoparallel_names
    assert (
        sum(
            node.target is torch.ops._dtensor.shard_dim_alltoall.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == a2a_fqn
            for node in regular_gm.graph.nodes
        )
        == 2
    )
    assert (
        sum(
            node.target is torch.ops._dtensor.shard_dim_alltoall.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == a2a_fqn
            for node in autoparallel_gm.graph.nodes
        )
        == 1
    )
    assert (
        sum(
            node.target is torch.ops.aten.mm.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in regular_gm.graph.nodes
        )
        == 2
    )
    assert (
        sum(
            node.target is torch.ops.aten.mm.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in autoparallel_gm.graph.nodes
        )
        == 1
    )
