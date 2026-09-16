"""The commands around training, and the scaffold that proves a task needs no core edit.

Nothing here starts a process. A dry run is the whole point: what a training run
will do has to be readable on a machine that cannot do it (ADR-0044).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_policy_contract, make_profile, write_profile, write_urdf
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, EXIT_UNREADABLE, app
from heron.core.contract import ActionSpec
from heron.core.enums import ActionOffset, ActionType, ControlMode
from heron.core.models import Completion, Scenario, SkillSpec

runner = CliRunner()

TASK = "bench_reach"
JOINTS = ["joint0", "joint1", "joint2"]


class BenchTask:
    backend = "isaaclab"
    spec = SkillSpec(
        name=TASK,
        required_capabilities=frozenset(make_profile().capabilities),
        required_control_mode=ControlMode.STREAMING,
    )
    completion = Completion(tolerance=0.05, hold_s=0.2, horizon_s=4.0)


@pytest.fixture
def tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("heron.train.tasks.load", lambda kind, name: BenchTask)
    installed = {TASK: "tests:BenchTask"}
    monkeypatch.setattr("heron.train.tasks.installed_tasks", lambda: installed)
    # The command imported the name, so the command's reference is the one a test
    # of the command has to replace.
    monkeypatch.setattr("heron.cli.main.installed_tasks", lambda: installed)


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    root.mkdir()
    profile = make_profile(
        description={"urdf": "urdf/test_arm.urdf"},
        contract=make_policy_contract(
            3,
            action=ActionSpec(
                type=ActionType.JOINT_POSITION,
                clip=(-1.0, 1.0),
                scale=[0.5, 0.5, 0.5],
                offset=ActionOffset.DEFAULT_POSE,
                kp=[10.0, 20.0, 30.0],
                kd=[1.0, 2.0, 3.0],
            ),
        ),
    )
    write_profile(root, profile)
    write_urdf(root, profile.name, JOINTS)
    return root


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "bench_train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "bench_train",
                "world": {"backend": "isaaclab"},
                "robots": [
                    {"id": "arm1", "profile": "test_arm", "role": "learner", "task": TASK}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# -- train run --dry-run ----------------------------------------------------


def test_a_dry_run_prints_the_derivation_and_starts_nothing(
    scenario_file: Path, registry: Path, tasks: None, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["train", "run", str(scenario_file), "--registry", str(registry),
         "--into", str(tmp_path / "run"), "--dry-run"],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "decimation 2" in result.output
    assert "joint_pos_rel" in result.output
    assert "будет выполнено" in result.output
    assert not (tmp_path / "run").exists()


def test_a_dry_run_says_it_did_not_check_the_machine(
    scenario_file: Path, registry: Path, tasks: None, tmp_path: Path
) -> None:
    """A plan is about the data; whether this machine can train is another question."""
    result = runner.invoke(
        app,
        ["train", "run", str(scenario_file), "--registry", str(registry),
         "--into", str(tmp_path / "run"), "--dry-run"],
    )
    assert "train env" in result.output


def test_a_scenario_of_an_unknown_backend_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "other",
                "world": {"backend": "no_such_trainer"},
                "robots": [{"id": "arm1", "profile": "test_arm", "role": "static"}],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "run", str(path), "--dry-run"])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "не начинается" in result.output


def test_an_unreadable_scenario_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "run", str(path), "--dry-run"])
    assert result.exit_code == EXIT_UNREADABLE


def test_a_learner_without_a_task_is_not_even_a_scenario(tmp_path: Path) -> None:
    """The payload of a role is required by the schema, as ``policy`` is for ``frozen``."""
    path = tmp_path / "no_task.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "no_task",
                "world": {"backend": "isaaclab"},
                "robots": [{"id": "arm1", "profile": "test_arm", "role": "learner"}],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "run", str(path), "--dry-run"])
    assert result.exit_code == EXIT_UNREADABLE
    assert "task" in result.output


# -- validate, on a training scenario ---------------------------------------


def test_validate_reports_a_task_nobody_installed(
    scenario_file: Path, registry: Path
) -> None:
    result = runner.invoke(app, ["validate", str(scenario_file), "--registry", str(registry)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "train.task_not_found" in result.output


def test_validate_does_not_ask_a_learner_for_an_adapter(
    scenario_file: Path, registry: Path, tasks: None
) -> None:
    """Nobody drives a learner, so the rule of ADR-0002 about adapters does not apply."""
    result = runner.invoke(app, ["validate", str(scenario_file), "--registry", str(registry)])
    assert "scenario.backend_adapter_missing" not in result.output
    assert result.exit_code == EXIT_OK, result.output


def test_a_neighbour_in_a_training_scene_is_refused(
    tmp_path: Path, registry: Path, tasks: None
) -> None:
    path = tmp_path / "crowd.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "crowd",
                "world": {"backend": "isaaclab"},
                "robots": [
                    {"id": "arm1", "profile": "test_arm", "role": "learner", "task": TASK},
                    {"id": "arm2", "profile": "test_arm", "role": "static"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(path), "--registry", str(registry)])
    assert "train.neighbours_unsupported" in result.output


def test_a_training_world_without_a_learner_is_refused(
    tmp_path: Path, registry: Path
) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "empty",
                "world": {"backend": "isaaclab"},
                "robots": [{"id": "arm1", "profile": "test_arm", "role": "static"}],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(path), "--registry", str(registry)])
    assert "train.learner_missing" in result.output


# -- task new ---------------------------------------------------------------


def test_the_stub_is_a_package_and_a_scenario_and_nothing_else(tmp_path: Path) -> None:
    """Criterion 10: somebody's own task is installed, never written into the core."""
    result = runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    assert result.exit_code == EXIT_OK, result.output

    created = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert created == [
        "my_task/my_task.yaml",
        "my_task/pyproject.toml",
        "my_task/src/my_task/__init__.py",
        "my_task/src/my_task/task.py",
    ]


def test_the_stub_declares_the_entry_point_of_the_sixth_extension_point(
    tmp_path: Path,
) -> None:
    from heron.core.branding import ENTRY_POINT_GROUPS

    runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    text = (tmp_path / "my_task" / "pyproject.toml").read_text(encoding="utf-8")
    assert ENTRY_POINT_GROUPS["tasks"] in text
    assert "my_task = \"my_task.task:MyTaskTask\"" in text


def test_the_stub_scenario_is_a_scenario(tmp_path: Path) -> None:
    runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    data = yaml.safe_load((tmp_path / "my_task" / "my_task.yaml").read_text(encoding="utf-8"))
    scenario = Scenario.model_validate({**data, "robots": [
        {**data["robots"][0], "profile": "test_arm"}
    ]})
    assert scenario.learners[0].task == "my_task"


def test_the_stub_imports_nothing_private(tmp_path: Path) -> None:
    from test_public_api import private_imports

    runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    assert not private_imports(tmp_path)


def test_the_stub_does_not_overwrite(tmp_path: Path) -> None:
    runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    result = runner.invoke(app, ["task", "new", "my_task", "--into", str(tmp_path)])
    assert result.exit_code == EXIT_UNREADABLE
    assert "уже есть" in result.output


def test_a_bad_task_name_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(app, ["task", "new", "MyTask", "--into", str(tmp_path)])
    assert result.exit_code == EXIT_UNREADABLE


def test_task_list_shows_what_is_installed(tasks: None) -> None:
    result = runner.invoke(app, ["task", "list"])
    assert result.exit_code == EXIT_OK
    assert TASK in result.output
