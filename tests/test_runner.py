"""The chain profile -> validator -> skill -> adapter, end to end.

No simulator, no ROS 2, no hardware: this is what phase 1 has to demonstrate,
and it is what CI runs on every commit.
"""

from __future__ import annotations

import pytest

from conftest import make_profile
from heron.adapters.dummy import DummyAdapter
from heron.core.capabilities import Capability
from heron.core.enums import ActionType, ControlMode, Status
from heron.sdk.runner import IncompatibleSkillError, check_runnable, run_skill
from heron.skills.hold import HoldSkill
from heron.skills.library import SkillLibrary, SkillNotFoundError


def test_hold_runs_to_success() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {"initial_position": [0.1, 0.2, 0.3]})

    status, seen = run_skill(HoldSkill(), adapter, profile, {"duration": 0.1})

    assert status is Status.SUCCESS
    assert len(seen) > 1
    assert adapter.read_state().joint_position == (0.1, 0.2, 0.3)
    assert all(cmd.type is ActionType.JOINT_POSITION for cmd in adapter.commands)


def test_skill_refuses_a_robot_it_cannot_drive() -> None:
    """Invariant 6: a stop before the first tick, not a failure mid-motion."""
    profile = make_profile(control_mode=ControlMode.COMMAND)
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(IncompatibleSkillError) as exc:
        run_skill(HoldSkill(), adapter, profile)

    assert "skill.control_mode_incompatible" in exc.value.report.codes()
    assert not adapter.commands  # nothing was sent


def test_link_poorer_than_the_profile_is_caught() -> None:
    """The profile says the robot can; the adapter says this link cannot."""
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {"capabilities": ["arm.joint_torque"]})

    report = check_runnable(HoldSkill(), profile, adapter)
    assert not report.ok
    assert "adapter.capabilities_missing" in report.codes()


def test_missing_capability_in_profile_is_caught() -> None:
    profile = make_profile(capabilities=frozenset({Capability("base.velocity")}))
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    report = check_runnable(HoldSkill(), profile, adapter)
    assert "skill.capabilities_missing" in report.codes()


def test_skill_without_declaration_cannot_be_checked() -> None:
    class Undeclared:
        def start(self, goal: dict[str, object], ctx: object) -> None: ...

        def tick(self, state: object) -> Status:
            return Status.SUCCESS

        def cancel(self) -> None: ...

    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    report = check_runnable(Undeclared(), profile, adapter)  # type: ignore[arg-type]
    assert "skill.undeclared" in report.codes()


def test_cancelled_skill_fails() -> None:
    skill = HoldSkill()
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    skill.start({}, object())  # type: ignore[arg-type]
    skill.cancel()
    assert skill.tick(adapter.read_state()) is Status.FAILURE


# -- library ---------------------------------------------------------------


def test_library_lists_builtin_skills() -> None:
    assert "hold" in SkillLibrary().names()


def test_library_returns_a_declaration() -> None:
    spec = SkillLibrary().spec("hold")
    assert spec.name == "hold"
    assert Capability("arm.joint_position") in spec.required_capabilities


def test_unknown_skill_is_an_error() -> None:
    with pytest.raises(SkillNotFoundError):
        SkillLibrary().spec("no_such_skill")


def test_library_creates_instances() -> None:
    first = SkillLibrary().create("hold")
    second = SkillLibrary().create("hold")
    assert first is not second
