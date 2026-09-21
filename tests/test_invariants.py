"""Invariants checked mechanically against the source tree.

CLAUDE.md phrases invariant 1 as a procedure: "delete ``robots/`` -- the core
still builds and passes its tests". A rule stated only in prose drifts, so the
procedure is executed here instead of being trusted.

The same applies to the naming rule of section 17. It is the kind of rule that
is obeyed on the day it is written and quietly broken three phases later, by a
single convenient literal in an error message. Grepping for it costs nothing.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

SRC = REPO_ROOT / "src"
PACKAGE = SRC / "heron"
BRANDING = PACKAGE / "core" / "branding.py"

APP_NAME = "heron"
"""Spelled here on purpose: this file is the one that hunts for the literal."""


def python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


# -- section 17: one source of truth for the name --------------------------


def test_name_literal_appears_only_in_branding() -> None:
    pattern = re.compile(APP_NAME, re.IGNORECASE)
    offenders: list[str] = []
    for path in python_files(PACKAGE):
        if path == BRANDING:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, "имя платформы написано литералом:\n" + "\n".join(offenders)


def test_package_uses_relative_imports() -> None:
    """An absolute self-import spells the name once per module (section 17)."""
    offenders: list[str] = []
    for path in python_files(PACKAGE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if (node.module or "").split(".")[0] == APP_NAME:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == APP_NAME:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, "абсолютный импорт внутри пакета:\n" + "\n".join(offenders)


def test_schemas_carry_no_platform_name() -> None:
    """Data schemas version themselves, they do not name the platform."""
    from heron.core.models import RobotProfile, Scenario

    for model in (RobotProfile, Scenario):
        fields = set(model.model_fields)
        assert "schema_version" in fields
        assert not any(APP_NAME in name.lower() for name in fields)


def test_environment_variables_are_built_from_the_prefix() -> None:
    from heron.core.branding import ENV_PREFIX, env_var

    assert env_var("HOME") == f"{ENV_PREFIX}_HOME"
    assert env_var("store") == f"{ENV_PREFIX}_STORE"


# -- invariant 1: no robot names in the core -------------------------------


VENDOR_WORDS = [
    "unitree",
    "franka",
    "universal_robots",
    "kuka",
    "fanuc",
    "boston",
    "pulse_75",
    "g1",
]
TYPE_WORDS = ["humanoid", "manipulator", "гуманоид", "манипулятор"]


@pytest.mark.parametrize("word", VENDOR_WORDS)
def test_core_mentions_no_vendor(word: str) -> None:
    pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in python_files(PACKAGE)
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"вендорское имя {word!r} в ядре: {offenders}"


@pytest.mark.parametrize("word", TYPE_WORDS)
def test_core_knows_capabilities_not_types(word: str) -> None:
    """Invariant 7: the core does not know the words "humanoid" or "arm type"."""
    pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
    offenders: list[str] = []
    for path in python_files(PACKAGE):
        source = path.read_text(encoding="utf-8")
        # Docstrings may quote the invariant itself; code may not use the word.
        stripped = re.sub(r'"""[\s\S]*?"""', "", source)
        stripped = re.sub(r"#.*", "", stripped)
        if pattern.search(stripped):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"тип робота {word!r} в коде ядра: {offenders}"


def test_core_imports_without_examples() -> None:
    """The check CLAUDE.md states: remove the robots, the core still works.

    Run in a subprocess with ``examples`` hidden from the path, so an accidental
    import of an example would fail here rather than in phase 4.
    """
    code = (
        "import sys; "
        "sys.path = [p for p in sys.path if 'examples' not in p]; "
        "import heron, heron.core, heron.validator, heron.sdk, heron.cli.main; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_the_platform_imports_no_ros2() -> None:
    """Section 10, at the one place it is easiest to break.

    ``ros2_control`` is a core adapter and the CLI imports it on every command,
    including ``validate``. If it pulled in rclpy at import time, the platform
    would stop installing on a machine without a ROS 2 distribution -- which is
    the machine it is developed on. The import has to stay inside ``connect``
    (ADR-0014), and a subprocess is the only honest way to check that, because
    this process has already imported everything.
    """
    code = (
        "import sys; "
        "import heron.cli.main, heron.ros2, heron.adapters.ros2_control; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'rclpy', 'sensor_msgs', 'std_msgs', 'trajectory_msgs'}); "
        "print('leaked:', leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "leaked: []" in result.stdout, result.stdout


def test_importing_the_platform_imports_no_training_stack() -> None:
    """Section 10 again, at the place phase 6 makes it easy to break.

    The training backend ships in the core (ADR-0044), so every command imports
    it. If it pulled in Isaac, torch or the learning library at import time, the
    platform would stop installing on the machine it is developed on and on every
    machine that only executes policies. Isaac lives in the training process and
    nowhere else, and a subprocess is the only honest way to check that.
    """
    code = (
        "import sys; "
        "import heron.cli.main, heron.train.isaaclab, heron.train.isaaclab_task; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'isaaclab', 'isaacsim', 'isaaclab_rl', 'rsl_rl', 'torch'}); "
        "print('leaked:', leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "leaked: []" in result.stdout, result.stdout


def test_the_core_imports_no_task_of_the_examples() -> None:
    """A training task is found through an entry point, never imported by path."""
    pattern = re.compile(r"(arm6_reach|g1_upper_reach|g1_stand|examples)")
    offenders: list[str] = []
    for path in python_files(PACKAGE):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and pattern.search(stripped):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {stripped}")
    assert not offenders, "ядро импортирует пример:\n" + "\n".join(offenders)


def test_core_has_no_vendor_dependencies() -> None:
    """Section 10: the core installs without a simulator or an SDK."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = " ".join(data["project"]["dependencies"]).lower()
    for forbidden in ("mujoco", "isaac", "onnx", "rclpy", "torch", "gz-", "rsl"):
        assert forbidden not in required, f"{forbidden} в обязательных зависимостях ядра"


# -- invariant 5: a profile is data ----------------------------------------


def test_adding_a_robot_touches_no_core_file() -> None:
    """A generated stub is data plus an adapter package, and nothing else.

    The description joins the list in phase 2: it is an input the stub needs to
    stand in a world at all (ADR-0009), and it is still not a core file. The adapter
    became a package with its contract test in phase 5 (ADR-0056).
    """
    import tempfile

    from heron.cli.new_robot import write_stub

    with tempfile.TemporaryDirectory() as tmp:
        created = write_stub("some_robot", Path(tmp)).paths
        assert [p.relative_to(Path(tmp, "some_robot")).as_posix() for p in created] == [
            "profile.yaml",
            "pyproject.toml",
            "src/some_robot/__init__.py",
            "src/some_robot/adapter.py",
            "src/some_robot/testing.py",
            "tests/test_some_robot_adapter.py",
            "urdf/some_robot.urdf",
        ]


def test_a_generated_stub_validates_against_its_own_description() -> None:
    """The stub's two halves agree with each other on the day they are made."""
    import tempfile

    from heron.cli.new_robot import write_stub
    from heron.core.enums import ControlMode
    from heron.core.loading import load_model
    from heron.core.models import RobotProfile
    from heron.validator.rules import check_profile

    with tempfile.TemporaryDirectory() as tmp:
        for mode in ControlMode:
            profile_path = write_stub(f"robot_{mode}", Path(tmp), dof=4, control_mode=mode).profile
            profile = load_model(profile_path, RobotProfile)
            report = check_profile(profile, path=profile_path)
            assert report.ok, report.codes()
            assert profile.control_mode is mode


# -- the decision log is an index, not prose -------------------------------


DECISIONS = REPO_ROOT / "docs" / "decisions"
GUIDE = REPO_ROOT / "CLAUDE.md"
CARD = "[0-9][0-9][0-9][0-9]-*.md"
REFERENCE = re.compile(r"ADR-(\d{4})")
SEARCHED = ("src", "tests", "examples", "scripts", ".github", "docs/plans")


def card_numbers() -> set[str]:
    return {path.name[:4] for path in DECISIONS.glob(CARD)}


def indexed_numbers() -> set[str]:
    """Numbers CLAUDE.md names, including the grouped entries.

    The index writes one line per decision, but related decisions share a line:
    ``**0035-0037**`` as a range, ``**0052, 0057**`` as a list. Both forms are
    read here so that the guide stays free to group.
    """
    numbers: set[str] = set()
    for entry in re.findall(r"^- \*\*([0-9–—,\- ]+)\*\*", GUIDE.read_text("utf-8"), re.M):
        for part in entry.split(","):
            bounds = re.findall(r"\d{4}", part)
            if len(bounds) == 2:
                numbers.update(f"{n:04d}" for n in range(int(bounds[0]), int(bounds[1]) + 1))
            elif bounds:
                numbers.add(bounds[0])
    return numbers


def test_every_decision_is_named_in_the_guide() -> None:
    """CLAUDE.md is the index of the decision log, and an index with holes is worse
    than none: a decision nobody is pointed at is a decision nobody reads before
    working in its area. The rule "новое решение — одна строка сюда" is checked
    rather than trusted, in both directions.
    """
    cards, indexed = card_numbers(), indexed_numbers()
    missing = sorted(cards - indexed)
    assert not missing, "карточки есть, строки в CLAUDE.md нет: " + ", ".join(missing)
    invented = sorted(indexed - cards)
    assert not invented, "CLAUDE.md называет решения без карточек: " + ", ".join(invented)


def test_decision_references_resolve() -> None:
    """Five hundred references to seventy cards, and renumbering has happened once
    already (the phase cards ADR-P8A..P8F became ADR-0065..0070). A reference to a
    number that no longer exists points the reader at nothing.
    """
    dangling: list[str] = []
    cards = card_numbers()
    for root in SEARCHED:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number in sorted(set(REFERENCE.findall(text)) - cards):
                dangling.append(f"{path.relative_to(REPO_ROOT).as_posix()}: ADR-{number}")
    assert not dangling, "ссылка на решение без карточки:\n" + "\n".join(dangling)
