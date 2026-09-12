"""CLI behaviour, including the exit codes CI depends on.

0 -- nothing found, 1 -- errors found, 2 -- could not read the input. A command
that printed an error and still exited 0 would make the validator useless in a
pipeline, which is where it matters most.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import make_profile
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, EXIT_UNREADABLE, app

runner = CliRunner()


@pytest.fixture
def profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_arm.yaml"
    path.write_text(make_profile().model_dump_json(indent=2), encoding="utf-8")
    return path


def test_version_reports_paths() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == EXIT_OK
    assert "_HOME" in result.stdout  # the env var, built from the prefix


def test_capabilities_lists_the_closed_registry() -> None:
    result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == EXIT_OK
    assert "arm.joint_position" in result.stdout
    assert "закрыт" in result.stdout


def test_validate_clean_profile_exits_zero(profile_file: Path) -> None:
    result = runner.invoke(app, ["validate", str(profile_file)])
    assert result.exit_code == EXIT_OK


def test_validate_broken_profile_exits_one(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(
        make_profile(image="heron/arm:latest").model_dump_json(indent=2), encoding="utf-8"
    )
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "profile.image_has_namespace" in result.stdout


def test_unreadable_file_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "not_yaml.yaml"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == EXIT_UNREADABLE


def test_contract_id_prints_the_identifier(profile_file: Path) -> None:
    result = runner.invoke(app, ["contract", "id", str(profile_file)])
    assert result.exit_code == EXIT_OK
    assert result.stdout.strip() == make_profile().contract.contract_id


def test_contract_diff_reports_fields(tmp_path: Path, profile_file: Path) -> None:
    other = tmp_path / "other.yaml"
    changed = make_profile(name="test_arm")
    data = changed.model_dump(mode="json")
    data["contract"]["joint_order"] = ["joint1", "joint0", "joint2"]
    import json

    other.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["contract", "diff", str(profile_file), str(other)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "joint_order" in result.stdout


def test_contract_lock_then_validate(tmp_path: Path, profile_file: Path) -> None:
    assert runner.invoke(app, ["contract", "lock", str(profile_file)]).exit_code == EXIT_OK
    assert (tmp_path / "contracts.lock.json").is_file()
    assert runner.invoke(app, ["validate", str(profile_file)]).exit_code == EXIT_OK


def test_silent_contract_change_fails_validation(tmp_path: Path, profile_file: Path) -> None:
    """The end-to-end version of ADR-0004's rule."""
    runner.invoke(app, ["contract", "lock", str(profile_file)])

    import json

    data = json.loads(profile_file.read_text(encoding="utf-8"))
    data["contract"]["joint_order"] = ["joint1", "joint0", "joint2"]
    profile_file.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(profile_file)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "contract.undeclared_change" in result.stdout


def test_profile_new_creates_a_validating_stub(tmp_path: Path) -> None:
    assert runner.invoke(app, ["profile", "new", "my_bot", "--into", str(tmp_path)]).exit_code == 0
    stub = tmp_path / "my_bot" / "profile.yaml"
    assert stub.is_file()
    assert (tmp_path / "my_bot" / "adapter.py").is_file()
    assert runner.invoke(app, ["validate", str(stub)]).exit_code == EXIT_OK


def test_profile_new_refuses_to_overwrite(tmp_path: Path) -> None:
    runner.invoke(app, ["profile", "new", "my_bot", "--into", str(tmp_path)])
    result = runner.invoke(app, ["profile", "new", "my_bot", "--into", str(tmp_path)])
    assert result.exit_code == EXIT_UNREADABLE


def test_skill_check_reports_incompatibility(tmp_path: Path) -> None:
    from heron.core.enums import ControlMode

    path = tmp_path / "cmd_arm.yaml"
    path.write_text(
        make_profile(control_mode=ControlMode.COMMAND).model_dump_json(), encoding="utf-8"
    )
    result = runner.invoke(app, ["skill", "check", "hold", str(path)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "skill.control_mode_incompatible" in result.stdout


def test_skill_list_includes_builtin() -> None:
    result = runner.invoke(app, ["skill", "list"])
    assert result.exit_code == EXIT_OK
    assert "hold" in result.stdout
