# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from typing import cast
from unittest.mock import call, Mock, patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import init_distributed


def test_fake_pg_uses_requested_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "6")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torchtitan.distributed.utils.init_fake_mode") as init_fake_mode,
    ):
        assert init_distributed(CommConfig(mode="fake_backend")) == 8
    init_fake_mode.assert_called_once_with(8, "fake_backend", rank=6)


def test_fake_pg_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "8")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(ValueError, match=r"RANK must be in \[0, 8\)"),
    ):
        init_distributed(CommConfig(mode="fake_backend"))


def test_dist_sum_tensor_keeps_local_result_as_tensor():
    value = torch.tensor(3, dtype=torch.int64)

    result = dist_utils.dist_sum_tensor(value)

    assert result is value


def test_dist_sum_tensor_waits_for_distributed_result():
    value = torch.tensor(3, dtype=torch.int64)
    reduced = torch.tensor(8, dtype=torch.int64)
    mesh = cast(DeviceMesh, object())

    with (
        patch.object(dist_utils.funcol, "all_reduce", return_value=reduced) as reduce,
        patch.object(dist_utils.funcol, "wait_tensor", return_value=reduced) as wait,
    ):
        result = dist_utils.dist_sum_tensor(value, mesh)

    assert result is reduced
    reduce.assert_called_once_with(value, reduceOp="SUM", group=mesh)
    wait.assert_called_once_with(reduced)


def test_set_pg_timeouts_includes_spmd_mesh_groups_once():
    timeout = timedelta(minutes=30)
    one_dimensional_group = Mock(spec=torch.distributed.ProcessGroup)
    spmd_only_group = Mock(spec=torch.distributed.ProcessGroup)
    one_dimensional_mesh = Mock(spec=DeviceMesh)
    one_dimensional_mesh.get_group.return_value = one_dimensional_group
    spmd_mesh = Mock(spec=DeviceMesh)
    spmd_mesh.get_all_groups.return_value = [
        one_dimensional_group,
        spmd_only_group,
    ]
    parallel_dims = Mock(spec=ParallelDims)
    parallel_dims.get_all_one_dimensional_meshes.return_value = {
        "tp": one_dimensional_mesh
    }
    parallel_dims.spmd_meshes.return_value = [spmd_mesh]

    with (
        patch("torchtitan.distributed.utils.torch.distributed.barrier"),
        patch("torchtitan.distributed.utils.device_module.current_device"),
        patch("torchtitan.distributed.utils.device_module.synchronize"),
        patch(
            "torchtitan.distributed.utils.torch.distributed.set_timeout"
        ) as set_timeout,
    ):
        dist_utils.set_pg_timeouts(timeout, parallel_dims)

    assert set_timeout.call_args_list == [
        call(timeout, one_dimensional_group),
        call(timeout, spmd_only_group),
        call(timeout, None),
    ]
