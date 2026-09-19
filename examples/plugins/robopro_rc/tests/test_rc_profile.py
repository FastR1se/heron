"""The RC5 profile of the examples, through the command line a user has.

Phase 5, criterion 3: the validator refuses a streaming skill on the real profile,
before anything is connected. And the rest of the way to the robot short of the
robot: the profile and its scenario validate, the mission runs in the world of
numbers, the run on hardware is planned only with a speed scale.

Through the command line, not the validator's functions: a robot package imports the
platform's public API and nothing else (ADR-0021), and the command line is what a
user runs. Skipped when the examples are not beside the package.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from heron.core.models import RobotProfile

EXAMPLES = Path(__file__).resolve().parents[3]
REGISTRY = EXAMPLES / "robots"
PROFILE = REGISTRY / "rc5" / "profile.yaml"
SCENARIO = EXAMPLES / "scenarios" / "rc5_bench.yaml"
CLI = [sys.executable, "-m", RobotProfile.__module__.split(".")[0] + ".cli.main"]
"""The platform's command line, named through the module of a public class."""

pytestmark = pytest.mark.skipif(not PROFILE.is_file(), reason="профиля RC5 в examples нет")


def cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        cwd=EXAMPLES.parent,
    )


def test_the_profile_and_its_scenario_validate_clean() -> None:
    result = cli("validate", PROFILE, SCENARIO, "--registry", REGISTRY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "замечаний нет" in result.stdout


def test_a_streaming_skill_is_refused_on_the_real_profile_before_connecting() -> None:
    """Criterion 3: the rule of the validator, on the profile the robot runs with."""
    result = cli("skill", "check", "move_joints", PROFILE)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[skill.control_mode_incompatible]" in result.stdout
    assert cli("skill", "check", "goto_joints", PROFILE).returncode == 0


def test_a_scenario_asking_the_robot_for_a_streaming_skill_does_not_validate() -> None:
    result = cli("run", SCENARIO, "--registry", REGISTRY, "--skill", "move_joints",
                 "-g", "duration=4", "--target", "real", "--speed-scale", "0.1", "--dry-run")
    assert result.returncode != 0
    assert "move_joints" in result.stdout + result.stderr


def test_the_mission_runs_in_the_world_of_numbers() -> None:
    result = cli("run", SCENARIO, "--registry", REGISTRY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "выполнено" in result.stdout


def test_the_run_on_hardware_is_planned_only_with_a_scale() -> None:
    planned = cli("run", SCENARIO, "--registry", REGISTRY, "--target", "real",
                  "--speed-scale", "0.1", "--dry-run")
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert "real -> robopro_rc" in planned.stdout
    assert "скорости 0.1" in planned.stdout

    unscaled = cli("run", SCENARIO, "--registry", REGISTRY, "--target", "real", "--dry-run")
    assert unscaled.returncode != 0
