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
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
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
    pp_enabled = False

    def __init__(
        self,
        *,
        sparse: bool = False,
        generic_3d: bool = False,
        dp_shard_3d: bool = False,
    ):
        self.sparse = sparse
        self.generic_3d = generic_3d
        self.dp_shard_3d = dp_shard_3d
        self.tp_enabled = not sparse
        self.dp_replicate_enabled = generic_3d
        self.cp_enabled = generic_3d or dp_shard_3d
        self.dp_replicate = 2 if generic_3d else 1
        self.dp_shard = 1 if generic_3d else 2
        self.cp = 2 if self.cp_enabled else 1
        self.tp = 2 if self.tp_enabled else 1
        self.world_size = self.dp_replicate * self.dp_shard * self.cp * self.tp

    def get_optional_mesh(self, name):
        enabled = {"dp_replicate", "fsdp", "tp"} if self.generic_3d else {"fsdp", "tp"}
        if self.sparse:
            enabled = {"efsdp", "ep"}
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
        local_batch_size=2,
        seq_len=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def _build_autoparallel_a2a_linear_graph(
    *,
    with_backward=False,
    a2a_as_weight=False,
    linear_op=torch.ops.aten.mm.default,
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
        a2a: "layers.0.attention.wo",
        pre_wo_unsqueeze: "layers.0.attention.wo",
        pre_wo_permute: "layers.0.attention.wo",
        pre_wo_reshape: "layers.0.attention.wo",
        pre_wo: "layers.0.attention.wo",
        wo: "layers.0.attention.wo",
        post_wo_unsqueeze: "layers.0.attention.wo",
        post_wo_reshape: "layers.0.attention.wo",
        post_wo_permute: "layers.0.attention.wo",
        post_wo: "layers.0.attention.wo",
        residual_add: "layers.0",
        ffn_w1: "layers.0.feed_forward.w1",
        ffn_w3: "layers.0.feed_forward.w3",
        ffn_w2: "layers.0.feed_forward.w2",
    }
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

    with pytest.raises(ValueError, match="save and load paths are mutually exclusive"):
        GraphTrainerCompileConfig(
            enable_autoparallel=True,
            autoparallel_placements_save_path="save.json",
            autoparallel_placements_load_path="load.json",
        )
    with pytest.raises(ValueError, match="require --compile.enable_autoparallel"):
        GraphTrainerCompileConfig(autoparallel_placements_load_path="load.json")


def test_muse_manual_uses_graphtrainer_defaults_with_eager_memory_policy():
    pytest.importorskip("torchvision")
    from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
        graph_trainer_muse_glimmer_30b_sdpa_c4_4x2,
    )

    config = graph_trainer_muse_glimmer_30b_sdpa_c4_4x2()
    assert config.compile.enable_autoparallel is False
    assert config.compile.memory_policy == "eager"
    assert config.compile.inductor_compilation == "regional"
    assert config.compile.disable_passes == []


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


def _normalize_deepseek_parameter_name(name):
    return (
        name.replace(
            ".moe.routed_experts.inner_experts.w1_EFD",
            ".moe.experts.w1",
        )
        .replace(
            ".moe.routed_experts.inner_experts.w2_EDF",
            ".moe.experts.w2",
        )
        .replace(
            ".moe.routed_experts.inner_experts.w3_EFD",
            ".moe.experts.w3",
        )
    )


@pytest.mark.parametrize("flavor", ["debugmodel", "16B"])
def test_autoparallel_deepseek_accepts_current_torchtitan_config(flavor):
    pytest.importorskip("autoparallel")
    from autoparallel._testing.models.dsv3 import DeepSeekV3Model

    from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry

    config = model_registry(flavor, attn_backend="sdpa").model
    with torch.device("meta"):
        native_model = config.build()
        autoparallel_model = DeepSeekV3Model(
            config,
            mesh=None,
            compute_dtype=torch.bfloat16,
        )

    native_parameters = {
        _normalize_deepseek_parameter_name(name): (parameter.shape, parameter.dtype)
        for name, parameter in native_model.named_parameters()
    }
    autoparallel_parameters = {
        name: (parameter.shape, parameter.dtype)
        for name, parameter in autoparallel_model.named_parameters()
    }
    assert autoparallel_parameters == native_parameters
    assert sum(p.numel() for p in autoparallel_model.parameters()) == sum(
        p.numel() for p in native_model.parameters()
    )
    if flavor == "16B":
        assert sum(p.numel() for p in autoparallel_model.parameters()) == 15_706_484_224

    moe = autoparallel_model.layers["1"].moe
    assert moe.tokens_per_expert_E is moe.tokens_per_expert
    assert moe.expert_bias_E is moe.expert_bias


def test_autoparallel_deepseek_rope_uses_runtime_positions():
    pytest.importorskip("autoparallel")
    from autoparallel._testing.models.dsv3 import apply_rotary_emb, precompute_freqs_cis

    from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry

    config = model_registry("debugmodel", attn_backend="sdpa").model
    rope_config = config.first_attention.rope
    rope = rope_config.build()
    batch_size = 8
    seq_len = 2048
    positions = torch.arange(seq_len).repeat(batch_size, 1)
    positions[:, seq_len // 2 :] -= seq_len // 2
    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, 1, rope_config.dim)

    expected, _ = rope(query, query, positions)
    freqs_cis = precompute_freqs_cis(config)
    actual = apply_rotary_emb(query, freqs_cis[positions])

    torch.testing.assert_close(actual, expected)


def test_autoparallel_deepseek_accepts_flex_attention():
    pytest.importorskip("autoparallel")
    from autoparallel._testing.models.dsv3 import DeepSeekV3Model

    from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry

    config = model_registry("debugmodel", attn_backend="flex").model
    with torch.device("meta"):
        model = DeepSeekV3Model(config, mesh=None, compute_dtype=torch.bfloat16)
    assert len(model.layers) == len(config.layers)


@pytest.mark.parametrize(
    "linear_op", [torch.ops.aten.mm.default, torch.ops.aten.linear.default]
)
def test_autoparallel_eager_sac_saves_a2a_linear_boundaries(linear_op):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(linear_op=linear_op)
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


def test_autoparallel_eager_sac_requires_a2a_as_linear_activation():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_as_weight=True)
    config = SimpleNamespace(
        compile=SimpleNamespace(enable_autoparallel=True, memory_policy="eager")
    )

    tag_with_memory_policy_pass(gm, config=config)

    assert nodes.a2a.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert nodes.post_wo.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE


def test_autoparallel_eager_sac_does_not_rematerialize_a2a_linear():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    def apply_passes(*, enable_autoparallel):
        gm, nodes = _build_autoparallel_a2a_linear_graph(with_backward=True)
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
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in regular_gm.graph.nodes
        )
        == 2
    )
    assert (
        sum(
            node.target is torch.ops._dtensor.shard_dim_alltoall.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
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
    assert len(autop.input_constraints) == 2
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path


@pytest.mark.parametrize("use_saved_placements", [False, True])
def test_llama_dp_shard_cp_tp_autoparallel_contract(tmp_path, use_saved_placements):
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

    _FakeAutoParallelGraph.instances.clear()
    ap_mesh = _FakeMesh(("dp_shard", "cp", "tp"), size=8)
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
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))

    with (
        patch.object(
            parallelize_autoparallel,
            "init_device_mesh",
            return_value=ap_mesh,
        ) as mesh_builder,
        patch.object(
            parallelize_autoparallel,
            "AutoParallelGraph",
            _FakeAutoParallelGraph,
        ),
        patch.object(
            parallelize_autoparallel,
            "apply_compile",
            lambda model, **_: model,
        ),
        patch.object(
            parallelize_autoparallel,
            "_apply_autoparallel_context_parallel_attention",
        ) as apply_cp,
        patch.object(parallelize_autoparallel, "device_type", "cpu"),
    ):
        parallelize_autoparallel.parallelize_autoparallel_llama(
            model,
            parallel_dims=_FakeParallelDims(dp_shard_3d=True),
            training=_training_config(),
            parallelism=ParallelismConfig(),
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
        )

    mesh_builder.assert_called_once_with(
        "cpu",
        (2, 2, 2),
        mesh_dim_names=("dp_shard", "cp", "tp"),
    )
    autop = _FakeAutoParallelGraph.instances[0]
    input_sharding = (
        torch.distributed.tensor.Shard(0),
        torch.distributed.tensor.Shard(1),
        torch.distributed.tensor.Replicate(),
    )
    output_sharding = (
        torch.distributed.tensor.Shard(0),
        torch.distributed.tensor.Shard(1),
        torch.distributed.tensor.Shard(2),
    )
    assert autop.mesh is ap_mesh
    assert autop.kwargs["solver"] == "approx"
    assert autop.kwargs["strategy_radius"] == (0 if use_saved_placements else 2)
    assert autop.input_constraints == [input_sharding, input_sharding]
    assert autop.output_constraints == [output_sharding]
    model_output = autop.apply_kwargs["model_output"]
    assert model_output.output_placements == (torch.distributed.tensor.Shard(2),)
    assert model_output.sharded_output_axis == 2
    assert autop.apply_kwargs["manages_context_parallel_input"] is False
    apply_cp.assert_called_once_with(model, ap_mesh)
    if use_saved_placements:
        assert autop.optimize_calls == 0
        assert autop.loaded_path == str(placement_path)
    else:
        assert autop.optimize_calls == 1
        assert autop.saved_path == placement_path


def test_llama_cp_rejects_simultaneous_dp_replicate_and_dp_shard_axes():
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

    parallel_dims = _FakeParallelDims(generic_3d=True)
    parallel_dims.dp_shard = 2
    with pytest.raises(
        ValueError,
        match="simultaneous dp_replicate and dp_shard axes",
    ):
        parallelize_autoparallel._build_autoparallel_mesh(parallel_dims)


def test_llama_autoparallel_context_parallel_wraps_sdpa():
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel
    from torchtitan.models.common.attention import ScaledDotProductAttention

    inner_attention = ScaledDotProductAttention(ScaledDotProductAttention.Config())
    attention = SimpleNamespace(
        inner_attention=inner_attention,
        scaling=None,
        enable_gqa=True,
    )
    model = SimpleNamespace(layers={"0": SimpleNamespace(attention=attention)})
    mesh = object()
    calls = []

    def fake_make_context_parallel(actual_mesh, **kwargs):
        assert actual_mesh is mesh
        assert kwargs == {
            "kind": "sdpa",
            "is_causal": True,
            "scale": None,
            "enable_gqa": True,
        }

        def cp_attention(q, k, v):
            calls.append((q.shape, k.shape, v.shape))
            return q

        return cp_attention

    with patch.object(
        parallelize_autoparallel,
        "make_context_parallel",
        side_effect=fake_make_context_parallel,
    ):
        parallelize_autoparallel._apply_autoparallel_context_parallel_attention(
            model, mesh
        )

    q = torch.randn(2, 8, 4, 16)
    k = torch.randn(2, 8, 2, 16)
    v = torch.randn(2, 8, 2, 16)
    actual = inner_attention(
        q,
        k,
        v,
        attention_masks=None,
        scale=None,
        enable_gqa=True,
    )
    torch.testing.assert_close(actual, q)
    assert calls == [
        (
            torch.Size((2, 4, 8, 16)),
            torch.Size((2, 2, 8, 16)),
            torch.Size((2, 2, 8, 16)),
        )
    ]


def test_llama_autoparallel_context_parallel_rejects_non_sdpa():
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

    model = SimpleNamespace(
        layers={
            "0": SimpleNamespace(
                attention=SimpleNamespace(inner_attention=torch.nn.Identity())
            )
        }
    )
    with pytest.raises(ValueError, match="currently requires SDPA"):
        parallelize_autoparallel._apply_autoparallel_context_parallel_attention(
            model, object()
        )


def test_autoparallel_cp_input_contract_defaults_to_model_managed():
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        autoparallel_manages_context_parallel_input,
    )

    model = torch.nn.Module()
    assert autoparallel_manages_context_parallel_input(model) is True
    model._torchtitan_autoparallel_manages_context_parallel_input = False
    assert autoparallel_manages_context_parallel_input(model) is False


@pytest.mark.parametrize(
    ("model_manages_cp", "expected_input_enabled", "expected_mesh_name"),
    [(True, False, "batch"), (False, True, "loss")],
)
def test_graph_trainer_respects_autoparallel_cp_input_contract(
    model_manages_cp, expected_input_enabled, expected_mesh_name
):
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    batch_mesh = object()
    loss_mesh = object()
    model = torch.nn.Module()
    model._torchtitan_autoparallel_manages_context_parallel_input = model_manages_cp
    trainer = object.__new__(GraphTrainer)
    trainer.config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(enable_autoparallel=True)
    )
    trainer.model_parts = [model]
    trainer.parallel_dims = SimpleNamespace(
        cp_enabled=True,
        get_optional_mesh=lambda name: {"batch": batch_mesh, "loss": loss_mesh}[name],
    )

    assert trainer._context_parallel_input_enabled() is expected_input_enabled
    expected_mesh = batch_mesh if expected_mesh_name == "batch" else loss_mesh
    assert trainer._metrics_loss_mesh() is expected_mesh
