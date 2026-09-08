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

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
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
        moe_3d: bool = False,
    ):
        self.sparse = sparse
        self.generic_3d = generic_3d
        self.dp_shard_3d = dp_shard_3d
        self.moe_3d = moe_3d
        self.tp_enabled = not sparse or moe_3d
        self.dp_replicate_enabled = generic_3d
        self.cp_enabled = generic_3d or dp_shard_3d or moe_3d
        self.dp_replicate = 2 if generic_3d else 1
        self.dp_shard = 1 if generic_3d else 2
        self.cp = 2 if self.cp_enabled else 1
        self.tp = 2 if self.tp_enabled else 1
        self.ep = 4 if moe_3d else 2
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
        "autoparallel_deepseek_v3_dp_cp_tp_folded_ep"
    ]
    assert suites["default"][0].ngpu == 4
    assert suites["h100"][0].ngpu == 8
    assert suites["h100"][0].disabled is False


def test_deepseek_v3_autoparallel_config_uses_sdpa_and_standard_loss():
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel_sdpa_cross_entropy_loss,
    )
    from torchtitan.models.common.attention import ScaledDotProductAttention

    config = graph_trainer_deepseek_v3_debugmodel_sdpa_cross_entropy_loss()

    assert isinstance(config.loss, CrossEntropyLoss.Config)
    assert {
        type(layer.attention.inner_attention)
        for layer in config.model_spec.model.layers
    } == {ScaledDotProductAttention.Config}


def test_deepseek_v3_autoparallel_config_preserves_flex_attention():
    from autoparallel._testing.models.dsv3 import (
        FlexAttentionConfig,
        make_dsv3_config,
    )
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel,
    )
    from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize_autoparallel import (
        _to_autoparallel_dsv3_config,
    )

    config = graph_trainer_deepseek_v3_debugmodel()
    source = config.model_spec.model.layers[0].attention.inner_attention
    ap_config = _to_autoparallel_dsv3_config(
        config.model_spec.model,
        make_dsv3_config,
        FlexAttentionConfig,
    )

    converted = ap_config.layers[0].attention.inner_attention
    assert isinstance(converted, FlexAttentionConfig)
    assert converted.block_size == source.block_size
    assert converted.kernel_options == source.kernel_options


def test_deepseek_v3_flex_cp32k_config_uses_flex_and_32k_rope():
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel_flex_cp32k,
    )
    from torchtitan.models.common.attention import FlexAttention

    config = graph_trainer_deepseek_v3_debugmodel_flex_cp32k()

    assert isinstance(config.loss, CrossEntropyLoss.Config)
    assert {
        type(layer.attention.inner_attention)
        for layer in config.model_spec.model.layers
    } == {FlexAttention.Config}
    assert {
        layer.attention.rope.max_seq_len for layer in config.model_spec.model.layers
    } == {32768}
    assert (config.training.local_batch_size, config.training.seq_len) == (1, 32768)
    assert (
        config.parallelism.context_parallel_degree,
        config.parallelism.tensor_parallel_degree,
        config.parallelism.expert_parallel_degree,
    ) == (4, 2, 8)
    assert config.parallelism.context_parallel_load_balancer is None


def test_deepseek_v3_ap_moe_implements_optimizer_hook_contract():
    from autoparallel.cast_parametrization import apply_dtype_cast
    from autoparallel._testing.models.dsv3 import (
        DeepSeekV3Model,
        FlexAttentionConfig,
        make_dsv3_config,
    )
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel_sdpa_cross_entropy_loss,
    )
    from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize_autoparallel import (
        _preserve_moe_attributes,
        _to_autoparallel_dsv3_config,
    )

    config = graph_trainer_deepseek_v3_debugmodel_sdpa_cross_entropy_loss()
    with torch.device("meta"):
        original_model = config.model_spec.model.build()
        ap_config = _to_autoparallel_dsv3_config(
            original_model.config,
            make_dsv3_config,
            FlexAttentionConfig,
        )
        ap_model = DeepSeekV3Model(ap_config)
    ap_model = apply_dtype_cast(
        ap_model,
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )

    _preserve_moe_attributes(original_model, ap_model)
    ap_moes = [block.moe for block in ap_model.layers.values() if block.moe_enabled]

    assert ap_moes
    assert {type(moe).__name__ for moe in ap_moes} == {"DTypeCastMoE"}
    for moe in ap_moes:
        assert moe.tokens_per_expert_E is moe.tokens_per_expert
        assert moe.expert_bias_E is moe.expert_bias


def test_autoparallel_config_validation():
    with pytest.raises(ValueError, match="only supports --compile.mode aot_fx_trace"):
        validate_autoparallel_config(
            GraphTrainerCompileConfig(
                mode="jit",
                enable_autoparallel=True,
            )
        )

    compile_config = GraphTrainerCompileConfig(
        inductor_compilation="regional",
        enable_autoparallel=True,
    )
    validate_autoparallel_config(compile_config)

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_autoparallel_config(
            GraphTrainerCompileConfig(
                enable_autoparallel=True,
                autoparallel_placements_save_path="save.json",
                autoparallel_placements_load_path="load.json",
            )
        )

    with pytest.raises(ValueError, match="require --compile.enable_autoparallel"):
        validate_autoparallel_config(
            GraphTrainerCompileConfig(
                autoparallel_placements_load_path="load.json",
            )
        )


def test_autoparallel_graph_pass_selection_uses_regular_memory_policy():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            disable_passes=["cudagraph_pass"],
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            enable_async_tensor_parallel=False,
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass in pass_fns


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
        def __init__(self, config, *, mesh, roles, compute_dtype):
            super().__init__()
            self.model_args = SimpleNamespace(vocab_size=16)

        def get_attention_masks(self, positions):
            return None

    _FakeAutoParallelGraph.instances.clear()
    compile_config = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        inductor_compilation=inductor_compilation,
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
            patch.object(parallelize_autoparallel, "device_type", "cpu"),
            patch.object(
                parallelize_autoparallel,
                "_load_autoparallel_dsv3_dependency",
                return_value=(
                    FakeDeepSeekV3Model,
                    lambda model: None,
                    lambda **kwargs: (
                        _FakeMesh(("dp_shard_in_ep",)),
                        SimpleNamespace(
                            ep_axis_names=("dp_shard_in_ep",),
                            ep_group_name="dp_shard_in_ep",
                        ),
                    ),
                    object,
                    lambda **kwargs: None,
                ),
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

        if model_name == "deepseek":
            model.config.rope = object()
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


@pytest.mark.parametrize("use_saved_placements", [False, True])
def test_llama_3d_autoparallel_constraints_and_placement_io(
    tmp_path, use_saved_placements
):
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

    model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))
    with (
        patch.object(
            parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
        ),
        patch.object(
            parallelize_autoparallel, "apply_compile", lambda model, **_: model
        ),
        patch.object(
            parallelize_autoparallel,
            "_apply_autoparallel_context_parallel_attention",
        ) as apply_cp,
    ):
        parallelize_autoparallel.parallelize_autoparallel_llama(
            model,
            parallel_dims=_FakeParallelDims(generic_3d=True),
            training=_training_config(),
            parallelism=ParallelismConfig(),
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
        )

    autop = _FakeAutoParallelGraph.instances[0]
    expected = (torch.distributed.tensor.Shard(0),) + (
        torch.distributed.tensor.Replicate(),
    ) * 2
    assert autop.kwargs["solver"] == "approx"
    assert autop.kwargs["strategy_radius"] == (0 if use_saved_placements else 2)
    assert autop.input_constraints == [expected, expected]
    assert autop.output_constraints == [expected]
    assert autop.apply_kwargs["model_output"] is None
    assert autop.apply_kwargs["manages_context_parallel_input"] is True
    apply_cp.assert_called_once_with(model, autop.mesh)
    if use_saved_placements:
        assert autop.optimize_calls == 0
        assert autop.loaded_path == str(placement_path)
    else:
        assert autop.optimize_calls == 1
        assert autop.saved_path == placement_path


@pytest.mark.parametrize("use_saved_placements", [False, True])
def test_llama_dp_shard_cp_tp_autoparallel_constraints_and_placement_io(
    tmp_path, use_saved_placements
):
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
            parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
        ),
        patch.object(
            parallelize_autoparallel, "apply_compile", lambda model, **_: model
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
    _, positions = autop.input_fn()
    assert positions.dtype is torch.int64
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
    model = SimpleNamespace(
        layers={"0": SimpleNamespace(attention=attention)},
    )
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


@pytest.mark.parametrize("use_saved_placements", [False, True])
def test_deepseek_v3_3d_folded_ep_constraints_and_placement_io(
    tmp_path, use_saved_placements
):
    from torchtitan.experiments.graph_trainer.deepseek_v3 import (
        parallelize_autoparallel,
    )

    model_init = {}
    mesh_build = {}
    moe_roles = SimpleNamespace(ep_axis_names=("cp", "tp"), ep_group_name="ep")
    ap_mesh = _FakeMesh(("dp_shard_mod_ep", "cp", "tp"), size=8)

    class FakeDeepSeekV3Model(torch.nn.Module):
        def __init__(self, config, *, mesh, roles, compute_dtype):
            super().__init__()
            self.model_args = SimpleNamespace(vocab_size=16)
            model_init.update(mesh=mesh, roles=roles, compute_dtype=compute_dtype)

        def get_attention_masks(self, positions):
            return None

    def fake_build_moe_mesh(**kwargs):
        mesh_build.update(kwargs)
        return ap_mesh, moe_roles

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
        patch.object(parallelize_autoparallel, "device_type", "cpu"),
        patch.object(
            parallelize_autoparallel,
            "_load_autoparallel_dsv3_dependency",
            return_value=(
                FakeDeepSeekV3Model,
                lambda model: None,
                fake_build_moe_mesh,
                object,
                lambda **kwargs: None,
            ),
        ),
        patch.object(
            parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
        ),
        patch.object(
            parallelize_autoparallel, "apply_compile", lambda model, **_: model
        ),
    ):
        parallelize_autoparallel.parallelize_autoparallel_deepseekv3(
            SimpleNamespace(config=SimpleNamespace(rope=object())),
            parallel_dims=_FakeParallelDims(sparse=True, moe_3d=True),
            training=_training_config(),
            parallelism=ParallelismConfig(),
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
        )

    assert {
        k: mesh_build[k] for k in ("dp_replicate", "dp_shard", "cp", "tp", "ep")
    } == {
        "dp_replicate": 1,
        "dp_shard": 2,
        "cp": 2,
        "tp": 2,
        "ep": 4,
    }
    assert model_init["mesh"] is ap_mesh
    assert model_init["roles"] is moe_roles

    autop = _FakeAutoParallelGraph.instances[0]
    expected = (
        torch.distributed.tensor.Shard(0),
        torch.distributed.tensor.Replicate(),
        torch.distributed.tensor.Replicate(),
    )
    assert autop.mesh is ap_mesh
    assert autop.kwargs["solver"] == "approx"
    assert autop.kwargs["strategy_radius"] == (0 if use_saved_placements else 2)
    assert autop.kwargs["dynamic"] is True
    assert autop.input_constraints == [expected, expected]
    assert autop.output_constraints == [expected]
    assert autop.apply_kwargs["manages_context_parallel_input"] is True
    if use_saved_placements:
        assert autop.optimize_calls == 0
        assert autop.loaded_path == str(placement_path)
    else:
        assert autop.optimize_calls == 1
        assert autop.saved_path == placement_path


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
    setattr(
        model,
        "_torchtitan_autoparallel_manages_context_parallel_input",
        model_manages_cp,
    )
    trainer = object.__new__(GraphTrainer)
    trainer.config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(enable_autoparallel=True)
    )
    trainer.model_parts = [model]
    trainer.parallel_dims = SimpleNamespace(
        cp_enabled=True,
        get_optional_mesh=lambda name: {
            "batch": batch_mesh,
            "loss": loss_mesh,
        }[name],
    )

    assert trainer._context_parallel_input_enabled() is expected_input_enabled
    expected_mesh = batch_mesh if expected_mesh_name == "batch" else loss_mesh
    assert trainer._metrics_loss_mesh() is expected_mesh


def test_autoparallel_cp_input_contract_defaults_to_model_managed():
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        autoparallel_manages_context_parallel_input,
    )

    model = torch.nn.Module()
    assert autoparallel_manages_context_parallel_input(model) is True
    model._torchtitan_autoparallel_manages_context_parallel_input = False
    assert autoparallel_manages_context_parallel_input(model) is False


def test_autoparallel_graph_preserves_traced_kwargs_at_runtime():
    from autoparallel import ForwardInputs
    from torchtitan.experiments.graph_trainer import autoparallel_api

    tokens = torch.ones(2, 4)
    positions = torch.arange(4).expand(2, -1)
    attention_masks = object()
    calls = []

    def compiled(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0]

    def make_module(model, _params, _buffers, *, forward_fn):
        model.forward = forward_fn.__get__(model, type(model))
        return model

    autop = object.__new__(autoparallel_api.AutoParallelGraph)
    autop.model = torch.nn.Module()
    autop.gm = SimpleNamespace(graph=object())
    autop.joint_with_descriptors = SimpleNamespace(
        params_spec=[],
        buffers_spec=[],
    )
    autop.compiler_fn = object()
    autop._traced_inputs = ForwardInputs(
        args=(tokens,),
        kwargs={"positions": positions, "attention_masks": attention_masks},
    )
    autop._apply_placement_common = lambda _placement: ({}, {})

    with (
        patch.object(
            autoparallel_api,
            "aot_compile_joint_with_descriptors",
            return_value=compiled,
        ),
        patch.object(
            autoparallel_api,
            "get_plain_input_and_grad_nodes",
            return_value={},
        ),
        patch.object(autoparallel_api, "make_parallel_module", make_module),
    ):
        model = autop.apply_placement_for_fx_module(
            compile_config=GraphTrainerCompileConfig(),
        )
        output = model(
            tokens,
            positions=positions,
            attention_masks=attention_masks,
        )

    assert output is tokens
    assert calls == [
        (
            (tokens,),
            {"positions": positions, "attention_masks": attention_masks},
        )
    ]
