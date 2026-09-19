"""A mission against a scenario: the schema, and who does each task (ADR-0065 - ADR-0067).

Profiles are built in code and written to a temporary registry, as elsewhere in the
suite, so that deleting ``examples/`` leaves these tests standing (invariant 1).
The examples are checked at the end, and skip themselves without the directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import make_command_profile, make_profile, write_profile
from heron.coordinator.mission import Mission, load_mission
from heron.core.capabilities import Capability
from heron.core.enums import Severity
from heron.core.loading import LoadError, parse_model
from heron.core.models import Scenario
from heron.registry.registry import Registry
from heron.scenario.loader import load_scenario
from heron.store.policies import PolicyStore
from heron.validator.mission import MissionCheck, check_mission

POLICY = "reach@0123456789abcdef"


def mission(*tasks: dict[str, Any], name: str = "m") -> Mission:
    return parse_model({"schema_version": 1, "name": name, "tasks": list(tasks)}, Mission)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    """Robots that differ in exactly the ways a coordinator has to tell apart."""
    gripper = frozenset({Capability("arm.joint_position"), Capability("gripper.binary")})
    for profile in (
        make_profile("arm", dof=3, capabilities=gripper),
        make_profile("big", dof=5),
        make_command_profile("cmd", dof=4),
        # Declares the command-class skill on a streaming link: equality refuses it.
        make_profile("streamer", dof=4, skills=("goto_joints",)),
        make_profile("trained", dof=3, skills=({"move_joints": POLICY},)),
    ):
        write_profile(tmp_path, profile)
    return Registry([tmp_path])


def scenario(*robots: tuple[str, str, str]) -> Scenario:
    return parse_model(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": "dummy"},
            "robots": [{"id": rid, "profile": prof, "role": role} for rid, prof, role in robots],
        },
        Scenario,
    )


COMPLEX = (("a1", "arm", "scripted"), ("b1", "big", "scripted"), ("c1", "cmd", "scripted"),
           ("wall", "arm", "static"))


def check(m: Mission, s: Scenario, registry: Registry, tmp_path: Path) -> MissionCheck:
    return check_mission(m, s, registry, store=PolicyStore(tmp_path / "store"))


def errors(result: MissionCheck) -> list[str]:
    return [f.code for f in result.report.errors]


# -- schema --------------------------------------------------------------------------


def test_a_task_names_a_skill_a_goal_and_what_it_needs() -> None:
    m = mission(
        {
            "id": "pick",
            "skill": "move_joints",
            "goal": {"target": [0, 0, 0]},
            "required_capabilities": ["gripper.binary"],
            "requires": {"dof": ">=3"},
        }
    )
    task = m.tasks[0]
    assert task.required_capabilities == {Capability("gripper.binary")}
    assert str(task.requires["dof"]) == ">=3"
    assert task.goal == {"target": [0, 0, 0]}


def test_goal_fields_beside_the_task_are_refused() -> None:
    """A misspelt key must not become a goal field nobody reads (ADR-0065)."""
    with pytest.raises(LoadError, match="target"):
        mission({"id": "t", "skill": "hold", "target": [0, 0, 0]})


def test_unknown_quantities_and_capabilities_are_refused_on_reading() -> None:
    with pytest.raises(LoadError):
        mission({"id": "t", "skill": "hold", "requires": {"reach_m": ">=1"}})
    with pytest.raises(LoadError):
        mission({"id": "t", "skill": "hold", "required_capabilities": ["legs.walk"]})


def test_task_ids_are_unique() -> None:
    with pytest.raises(LoadError, match="повтор"):
        mission({"id": "t", "skill": "hold"}, {"id": "t", "skill": "hold"})


def test_a_task_waits_for_the_one_above_unless_told_otherwise() -> None:
    m = mission(
        {"id": "a", "skill": "hold"},
        {"id": "b", "skill": "hold"},
        {"id": "c", "skill": "hold", "after": []},
        {"id": "d", "skill": "hold", "after": ["a", "c"]},
    )
    assert [m.waits_for(i) for i in range(4)] == [(), ("a",), (), ("a", "c")]


def test_the_mission_schema_carries_a_version_and_no_platform_name() -> None:
    from heron.core.branding import APP_NAME

    assert "schema_version" in Mission.model_fields
    assert not any(APP_NAME in name for name in Mission.model_fields)


def test_a_mission_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("schema_version: 1\nname: m\ntasks:\n  - {id: t, skill: hold}\n", "utf-8")
    assert load_mission(path).tasks[0].id == "t"


# -- assignment (ADR-0066) -------------------------------------------------------------


def test_the_only_robot_declaring_a_skill_gets_the_task(registry: Registry, tmp_path: Path) -> None:
    result = check(mission({"id": "t", "skill": "goto_joints"}), scenario(*COMPLEX), registry,
                   tmp_path)
    assert result.ok, errors(result)
    assert result.assignment == {"t": "c1"}


def test_two_capable_robots_are_an_error_not_a_guess(registry: Registry, tmp_path: Path) -> None:
    result = check(mission({"id": "t", "skill": "move_joints"}), scenario(*COMPLEX), registry,
                   tmp_path)
    assert errors(result) == ["mission.task_ambiguous"]
    assert result.report.errors[0].detail["candidates"] == ["a1", "b1"]
    assert "t" not in result.assignment


def test_a_quantity_of_the_task_narrows_the_candidates(registry: Registry, tmp_path: Path) -> None:
    m = mission({"id": "t", "skill": "move_joints", "requires": {"dof": "==5"}})
    assert check(m, scenario(*COMPLEX), registry, tmp_path).assignment == {"t": "b1"}


def test_a_capability_of_the_task_narrows_the_candidates(
    registry: Registry, tmp_path: Path
) -> None:
    m = mission({"id": "t", "skill": "hold", "required_capabilities": ["gripper.binary"]})
    assert check(m, scenario(*COMPLEX), registry, tmp_path).assignment == {"t": "a1"}


def test_no_capable_robot_names_every_robots_reason(registry: Registry, tmp_path: Path) -> None:
    m = mission({"id": "t", "skill": "move_joints", "requires": {"dof": "==7"}})
    result = check(m, scenario(*COMPLEX), registry, tmp_path)

    assert errors(result) == ["mission.task_unassigned"]
    reasons = {
        robot: [item["code"] for item in items]
        for robot, items in result.report.errors[0].detail["reasons"].items()
    }
    assert reasons == {
        "a1": ["task.requirement_unmet"],
        "b1": ["task.requirement_unmet"],
        "c1": ["scenario.skill_not_in_profile"],
        "wall": ["mission.robot_static"],
    }


def test_the_control_class_is_compared_by_equality(registry: Registry, tmp_path: Path) -> None:
    """A streaming robot declaring a command-class skill is not a candidate (ADR-0049)."""
    result = check(
        mission({"id": "t", "skill": "goto_joints"}),
        scenario(("s1", "streamer", "scripted")),
        registry,
        tmp_path,
    )
    assert errors(result) == ["mission.task_unassigned"]
    codes = [r.code for r in result.reasons["t"]["s1"]]
    assert "skill.control_mode_incompatible" in codes


def test_a_pinned_robot_resolves_an_ambiguity(registry: Registry, tmp_path: Path) -> None:
    m = mission({"id": "t", "skill": "move_joints", "robot": "b1"})
    assert check(m, scenario(*COMPLEX), registry, tmp_path).assignment == {"t": "b1"}


def test_a_pin_is_checked_like_any_candidate(registry: Registry, tmp_path: Path) -> None:
    pinned_to_wall = mission({"id": "t", "skill": "hold", "robot": "wall"})
    assert errors(check(pinned_to_wall, scenario(*COMPLEX), registry, tmp_path)) == [
        "mission.task_unassigned"
    ]
    pinned_to_nobody = mission({"id": "t", "skill": "hold", "robot": "ghost"})
    assert errors(check(pinned_to_nobody, scenario(*COMPLEX), registry, tmp_path)) == [
        "mission.robot_unknown"
    ]


def test_a_policy_absent_here_assigns_with_a_warning(registry: Registry, tmp_path: Path) -> None:
    """As ``validate`` treats it (ADR-0046): the profile is right, this machine is not."""
    result = check(
        mission({"id": "t", "skill": "move_joints"}),
        scenario(("r1", "trained", "scripted")),
        registry,
        tmp_path,
    )
    assert result.ok
    assert result.assignment == {"t": "r1"}
    assert [(f.code, f.severity) for f in result.report] == [
        ("mission.skill_unverified", Severity.WARNING)
    ]


# -- order (ADR-0067) --------------------------------------------------------------------


def test_a_task_may_wait_only_for_tasks_written_above(registry: Registry, tmp_path: Path) -> None:
    m = mission(
        {"id": "a", "skill": "goto_joints", "after": ["b"]},
        {"id": "b", "skill": "goto_joints", "after": ["b"]},
        {"id": "c", "skill": "goto_joints", "after": ["zzz"]},
    )
    assert errors(check(m, scenario(*COMPLEX), registry, tmp_path)) == [
        "mission.dependency_forward",
        "mission.dependency_forward",
        "mission.dependency_unknown",
    ]


# -- examples --------------------------------------------------------------------------


def test_the_mixed_bench_mission_gives_each_class_its_task(examples_dir: Path) -> None:
    registry = Registry([examples_dir / "robots"])
    result = check_mission(
        load_mission(examples_dir / "missions" / "mixed_bench.yaml"),
        load_scenario(examples_dir / "scenarios" / "mixed_bench.yaml"),
        registry,
    )
    assert result.ok, errors(result)
    assert result.assignment == {"arm_ready": "arm1", "rest_ready": "arm2", "arm_hold": "arm1"}


@pytest.mark.parametrize("scene", ["complex.yaml", "complex_bench.yaml"])
def test_one_complex_mission_is_assigned_alike_in_both_worlds(
    examples_dir: Path, scene: str
) -> None:
    """The same mission, a world of numbers or Gazebo: only the scenario differs (ADR-0065)."""
    registry = Registry([examples_dir / "robots"])
    result = check_mission(
        load_mission(examples_dir / "missions" / "complex.yaml"),
        load_scenario(examples_dir / "scenarios" / scene),
        registry,
    )
    assert result.ok, errors(result)
    assert result.assignment == {
        "arm_aside": "arm1",
        "g1_arms": "g1",
        "arm_hold": "arm1",
        "arm_home": "arm1",
    }


def test_the_complex_missions_goals_are_as_wide_as_their_robots(examples_dir: Path) -> None:
    """``requires.dof`` and the width of ``target`` are one quantity written twice
    (ADR-0066): held together here, since the core does not read the goal."""
    registry = Registry([examples_dir / "robots"])
    m = load_mission(examples_dir / "missions" / "complex.yaml")
    s = load_scenario(examples_dir / "scenarios" / "complex.yaml")
    assignment = check_mission(m, s, registry).assignment
    profiles = {robot.id: registry.load(robot.profile) for robot in s.robots}
    for task in m.tasks:
        if "target" in task.goal:
            assert len(task.goal["target"]) == profiles[assignment[task.id]].contract.dof
