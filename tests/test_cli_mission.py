"""``mission``: the commands, and a mission with real robot processes (ADR-0068).

The coordinator starts one process per robot, as ``mission run`` does by default:
the same interpreter, ``python -m`` of the command line, a socket on the loopback.
The robots are made of numbers, so this runs anywhere -- Windows included -- and
it is the whole path: plan on disk, processes, protocol, executor, record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_command_profile, make_contract, make_profile, write_profile, write_urdf
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, EXIT_UNREADABLE, app
from heron.cli.mission import prepare_robot
from heron.cli.run import RunError
from heron.coordinator.journal import read_events
from heron.coordinator.mission import load_mission
from heron.coordinator.plan import PLAN_FILE, RESULT_FILE, MissionPlan, make_plan
from heron.core.models import ModelDescription
from heron.registry.registry import Registry
from heron.scenario.loader import load_scenario
from heron.sdk.record import read_record
from heron.store.policies import PolicyStore
from heron.validator.mission import check_mission

runner = CliRunner()

JOINTS = ["joint0", "joint1", "joint2"]


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """A registry of two robots of the two classes, fast enough to run for real."""
    write_urdf(tmp_path, "arm", JOINTS)
    write_urdf(tmp_path, "cmd", JOINTS)
    write_profile(tmp_path, make_profile("arm", description=ModelDescription(urdf="urdf/arm.urdf")))
    write_profile(
        tmp_path,
        make_command_profile(
            "cmd",
            description=ModelDescription(urdf="urdf/cmd.urdf"),
            contract=make_contract(3, state_rate_hz=20.0, command_rate_hz=10.0),
        ),
    )
    return tmp_path


def write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def scene(bench: Path) -> Path:
    return write(
        bench / "scene.yaml",
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": "dummy"},
            "robots": [
                {"id": "arm1", "profile": "arm", "role": "scripted"},
                {"id": "cmd1", "profile": "cmd", "role": "scripted"},
            ],
        },
    )


def mission(bench: Path, *tasks: dict[str, Any]) -> Path:
    document = {"schema_version": 1, "name": "demo", "tasks": list(tasks)}
    return write(bench / "mission.yaml", document)


REACH = {
    "id": "reach",
    "skill": "move_joints",
    "goal": {"target": [0.3, -0.2, 0.1], "duration": 0.2},
}
GO = {"id": "go", "skill": "goto_joints", "after": [],
      "goal": {"target": [0.1, 0.0, -0.1], "tolerance": 0.01}}
KEEP = {"id": "keep", "skill": "hold", "after": ["reach", "go"], "goal": {"duration": 0.1}}


def invoke(*args: str) -> Any:
    return runner.invoke(app, ["mission", *args])


# -- validate ----------------------------------------------------------------------


def test_validate_prints_who_does_what(bench: Path) -> None:
    result = invoke("validate", str(scene(bench)), str(mission(bench, REACH, GO, KEEP)),
                    "-R", str(bench))
    assert result.exit_code == EXIT_OK, result.output
    assert "reach  -> arm1" in result.output
    assert "go     -> cmd1" in result.output
    assert "после reach, go" in result.output


def test_validate_fails_on_an_ambiguous_task(bench: Path) -> None:
    path = write(
        bench / "twins.yaml",
        {
            "schema_version": 1,
            "name": "twins",
            "world": {"backend": "dummy"},
            "robots": [
                {"id": "left", "profile": "arm", "role": "scripted"},
                {"id": "right", "profile": "arm", "role": "scripted"},
            ],
        },
    )
    result = invoke("validate", str(path), str(mission(bench, REACH)), "-R", str(bench))
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "[mission.task_ambiguous]" in result.output


def test_an_unreadable_mission_is_exit_two(bench: Path) -> None:
    broken = write(bench / "broken.yaml", {"schema_version": 1, "name": "x", "tasks": []})
    assert invoke("validate", str(scene(bench)), str(broken), "-R", str(bench)).exit_code == (
        EXIT_UNREADABLE
    )


# -- run -----------------------------------------------------------------------------


def test_a_dry_run_prints_the_plan_and_the_robot_processes(bench: Path) -> None:
    into = bench / "run"
    result = invoke("run", str(scene(bench)), str(mission(bench, REACH, GO, KEEP)),
                    "-R", str(bench), "--into", str(into), "--dry-run")
    assert result.exit_code == EXIT_OK, result.output
    assert "arm1     профиль arm  связь sim.dummy -> dummy" in result.output
    assert f"mission robot {into.resolve()} --robot cmd1" in result.output
    assert not into.exists(), "a dry run writes nothing"


def test_a_mission_that_fails_validation_does_not_start(bench: Path) -> None:
    bad = dict(REACH, requires={"dof": "==7"})
    into = bench / "run"
    result = invoke("run", str(scene(bench)), str(mission(bench, bad)), "-R", str(bench),
                    "--into", str(into))
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert not into.exists()


def test_a_mission_runs_with_one_process_per_robot(bench: Path) -> None:
    into = bench / "run"
    result = invoke("run", str(scene(bench)), str(mission(bench, REACH, GO, KEEP)),
                    "-R", str(bench), "--into", str(into), "--record")
    assert result.exit_code == EXIT_OK, result.output
    assert "миссия выполнена" in result.output

    outcome = json.loads((into / RESULT_FILE).read_text(encoding="utf-8"))
    assert outcome["status"] == "success"
    assert [t["state"] for t in outcome["tasks"]] == ["succeeded"] * 3
    events = [e["event"] for e in read_events(into)]
    assert events.count("robot_ready") == 2
    assert events[-1] == "mission_finished"

    record = read_record(into / "go.mcap")
    assert record.run["mission"] == "demo"
    assert record.run["task"] == "go"
    assert record.run["plan"] == MissionPlan.read(into).plan_id
    assert record.result is not None and record.result["status"] == "success"


def test_a_failed_task_stops_the_mission_and_names_it(bench: Path) -> None:
    long_hold = {"id": "long", "skill": "hold", "goal": {"duration": 60.0}}
    bad_go = dict(GO, goal={"target": [0.1, 0.1, 0.1, 0.1], "tolerance": 0.01})
    into = bench / "run"
    result = invoke("run", str(scene(bench)), str(mission(bench, long_hold, bad_go)),
                    "-R", str(bench), "--into", str(into))
    assert result.exit_code == EXIT_FOUND_ERRORS, result.output
    outcome = json.loads((into / RESULT_FILE).read_text(encoding="utf-8"))
    assert outcome["status"] == "failure"
    assert outcome["reason"].startswith("задача go (cmd1")
    assert {t["task"]: t["state"] for t in outcome["tasks"]} == {
        "long": "cancelled",
        "go": "failed",
    }


def test_a_run_directory_is_not_reused(bench: Path) -> None:
    into = bench / "run"
    into.mkdir()
    (into / PLAN_FILE).write_text("{}", encoding="utf-8")
    result = invoke("run", str(scene(bench)), str(mission(bench, REACH)), "-R", str(bench),
                    "--into", str(into))
    assert result.exit_code == EXIT_FOUND_ERRORS


# -- a robot's process refuses before it connects ----------------------------------------


def a_plan(bench: Path) -> MissionPlan:
    """The plan ``mission run`` would write, with a world environment to apply."""
    registry = Registry([bench])
    s = load_scenario(scene(bench))
    m = load_mission(mission(bench, REACH, GO))
    checked = check_mission(m, s, registry, store=PolicyStore(bench / "store"))
    profiles = {robot.id: registry.load(robot.profile) for robot in s.robots}
    return make_plan(
        m,
        s,
        checked.assignment,
        profiles,
        scenario_path=bench / "scene.yaml",
        registry_roots=(bench,),
        store=None,
        environment={"ROS_DOMAIN_ID": "1"},
        record=False,
    )


def test_a_robot_refuses_a_process_in_another_domain(bench: Path) -> None:
    with pytest.raises(RunError, match="ROS_DOMAIN_ID"):
        prepare_robot(a_plan(bench), "arm1", bench, environ={"ROS_DOMAIN_ID": "0"})


def test_a_robot_refuses_a_profile_changed_after_the_plan(bench: Path) -> None:
    the_plan = a_plan(bench)
    changed = the_plan.model_copy(
        update={"robots": tuple(r.model_copy(update={"contract_id": "f" * 16})
                                for r in the_plan.robots)}
    )
    with pytest.raises(RunError, match="изменился после плана"):
        prepare_robot(changed, "arm1", bench, environ={})


def test_a_prepared_robot_is_connected_with_its_skills_built(bench: Path) -> None:
    environ: dict[str, str] = {}
    prepared = prepare_robot(a_plan(bench), "cmd1", bench, environ=environ)
    try:
        assert list(prepared.tasks) == ["go"]
        assert environ == {"ROS_DOMAIN_ID": "1"}, "the world's environment is applied (ADR-0039)"
    finally:
        prepared.executor.adapter.disconnect()
