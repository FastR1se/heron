"""A run on disk (ADR-0038): what the executor read and sent, on both clocks, and how it ended.

Records are read back with the platform's own reader here; the experiment scripts read
them with ``mcap`` directly, so a reader that agreed with a wrong writer would not go
unnoticed there.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_profile, write_profile, write_urdf
from heron.adapters.dummy import DummyAdapter
from heron.cli.main import EXIT_OK, app
from heron.cli.run import RunError, execute
from heron.core.enums import Status
from heron.core.models import AdapterSpec, JointRange, Limits, ModelDescription, RobotProfile
from heron.core.types import Command, RobotState
from heron.registry.registry import Registry
from heron.scenario.loader import load_scenario
from heron.sdk import record as recording
from heron.sdk.executor import Executor, ManualClock
from heron.sdk.record import INTERRUPTED, McapRecorder, RecordError, read_record
from heron.skills.move_joints import MoveJointsSkill

pytest.importorskip("mcap")

runner = CliRunner()
JOINTS = ["joint0", "joint1", "joint2"]
GOAL = {"target": [0.5, 0.0, -0.5], "duration": 0.1}


class NarrowAdapter(DummyAdapter):
    """Guards ±1 while the profile allows ±2: the skill's first command is refused."""

    def connect(self, profile: RobotProfile, config: dict[str, object]) -> None:
        dof = profile.contract.dof
        narrow = Limits(joint_position=JointRange(min=[-1.0] * dof, max=[1.0] * dof))
        super().connect(profile.model_copy(update={"limits": narrow}), config)


class BreakingAdapter(DummyAdapter):
    """A link that goes down after two commands: not a refusal, a defect."""

    def send_command(self, cmd: Command) -> None:
        self.accepted = getattr(self, "accepted", 0) + 1
        if self.accepted > 2:
            raise RuntimeError("связь оборвалась")
        super().send_command(cmd)


class WatchedAdapter(DummyAdapter):
    connected: ClassVar[list[str]] = []

    def connect(self, profile: RobotProfile, config: dict[str, object]) -> None:
        type(self).connected.append(profile.name)
        super().connect(profile, config)


def _recorded_run(path: Path, adapter: DummyAdapter | None = None) -> tuple[object, object]:
    profile = make_profile()
    link = adapter if adapter is not None else DummyAdapter()
    link.connect(profile, {"initial_position": [0.0, 0.0, 1.5]})
    recorder = McapRecorder(path, {"robot": "arm1"}, origin_ns=0)
    result = Executor(link, profile, clock=ManualClock(), recorder=recorder).run(
        MoveJointsSkill(), GOAL
    )
    recorder.close(result)
    return result, read_record(path)


def _bench(tmp_path: Path, plugin: str = "dummy") -> Path:
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters={"sim.dummy": AdapterSpec(plugin=plugin)},
    )
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)
    scene = {
        "schema_version": 1,
        "name": "scene",
        "world": {"backend": "dummy"},
        "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        "task": {"skill": "move_joints", **GOAL},
    }
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene, allow_unicode=True), encoding="utf-8")
    return path


# -- what is in a record -------------------------------------------------------


def test_every_state_read_and_every_command_sent_is_in_the_record(tmp_path: Path) -> None:
    result, record = _recorded_run(tmp_path / "run.mcap")

    assert result.status is Status.SUCCESS  # type: ignore[attr-defined]
    assert [item.value for item in record.states] == result.states  # type: ignore[attr-defined]
    assert [item.value for item in record.commands] == result.commands  # type: ignore[attr-defined]
    assert record.run["robot"] == "arm1"  # type: ignore[attr-defined]
    assert record.result == {  # type: ignore[attr-defined]
        "status": "success",
        "ticks": str(result.ticks),  # type: ignore[attr-defined]
        "commands": str(len(result.commands)),  # type: ignore[attr-defined]
        "reason": "",
    }


def test_each_message_carries_the_executors_clock_and_the_worlds(tmp_path: Path) -> None:
    """A command's world time is the stamp of the state it was computed from, and it is
    written right after that state; the executor's clock moves by the command period."""
    _, record = _recorded_run(tmp_path / "run.mcap")
    states, commands = record.states, record.commands  # type: ignore[attr-defined]

    for state, command in zip(states, commands, strict=True):
        # MCAP keeps time in whole nanoseconds; a world clock in ms loses nothing.
        assert command.world == state.world == pytest.approx(state.value.timestamp, abs=1e-9)
        assert command.sequence == state.sequence + 1
        assert command.wall == state.wall
    assert [s.wall for s in states] == pytest.approx([i * 0.02 for i in range(len(states))])


def test_a_refused_command_is_not_recorded_as_sent(tmp_path: Path) -> None:
    result, record = _recorded_run(tmp_path / "run.mcap", NarrowAdapter())

    assert result.status is Status.FAILURE  # type: ignore[attr-defined]
    assert record.commands == []  # type: ignore[attr-defined]
    assert len(record.states) == 1  # type: ignore[attr-defined]
    assert record.result["status"] == "failure"  # type: ignore[attr-defined]
    assert "joint2 = 1.5" in record.result["reason"]  # type: ignore[attr-defined]


def test_a_value_that_is_not_a_number_is_written_as_null_and_read_back_as_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nan.mcap"
    recorder = McapRecorder(path, {}, origin_ns=0)
    recorder.state(0.0, RobotState(timestamp=1.0, joint_position=(math.nan, 1.0)))
    recorder.close(None)

    (state,) = read_record(path).states
    assert math.isnan(state.value.joint_position[0])
    assert state.value.joint_position[1] == 1.0


def test_a_file_that_is_not_a_record_is_refused(tmp_path: Path) -> None:
    from mcap.writer import Writer

    path = tmp_path / "other.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        writer.finish()
    with pytest.raises(RecordError, match="не запись прогона"):
        read_record(path)


# -- run --record --------------------------------------------------------------


def test_run_record_writes_the_run_and_its_contract(tmp_path: Path) -> None:
    scene = _bench(tmp_path)
    out = tmp_path / "out.mcap"

    result = runner.invoke(app, ["run", str(scene), "-R", str(tmp_path), "--record", str(out)])

    assert result.exit_code == EXIT_OK, result.output
    assert f"записано   {out}" in result.output
    record = read_record(out)
    profile = Registry([tmp_path]).load("arm")
    assert record.run["contract_id"] == profile.contract.contract_id
    assert record.run["contract"] == profile.contract.canonical_json()
    assert record.run["skill"] == "move_joints"
    assert record.result is not None and record.result["status"] == "success"


def test_an_interrupted_run_leaves_what_it_had_and_says_so(tmp_path: Path) -> None:
    scene = _bench(tmp_path, plugin="test_record:BreakingAdapter")
    out = tmp_path / "out.mcap"

    with pytest.raises(RuntimeError, match="связь оборвалась"):
        execute(load_scenario(scene), Registry([tmp_path]), record=out, environ={})

    record = read_record(out)
    assert record.result is not None
    assert record.result["status"] == INTERRUPTED
    assert "связь оборвалась" in record.result["reason"]
    assert len(record.commands) == 2
    assert len(record.states) == 3


def test_without_mcap_the_run_is_refused_before_it_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _bench(tmp_path, plugin="test_record:WatchedAdapter")
    out = tmp_path / "out.mcap"
    WatchedAdapter.connected.clear()

    def absent(part: str) -> object:
        raise RecordError("запись прогона требует пакет mcap")

    monkeypatch.setattr(recording, "_mcap", absent)
    with pytest.raises(RunError, match="mcap"):
        execute(load_scenario(scene), Registry([tmp_path]), record=out, environ={})

    assert WatchedAdapter.connected == []
    assert not out.exists()


def test_a_dry_run_names_the_record_and_writes_nothing(tmp_path: Path) -> None:
    scene = _bench(tmp_path)
    out = tmp_path / "out.mcap"

    lines, result = execute(
        load_scenario(scene), Registry([tmp_path]), record=out, dry_run=True, environ={}
    )

    assert result is None
    assert f"запись    {out}" in lines
    assert not out.exists()
