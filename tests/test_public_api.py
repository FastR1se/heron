"""A robot package speaks to the platform through the public API only (ADR-0021).

The package of a vendor's robot is the proof that a robot joins through the
extension points. That proof is false the moment the package imports the
validator or the command line, and nobody would notice by reading it -- so the
imports are read by a test.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from conftest import EXAMPLES
from heron.cli.new_robot import write_stub
from heron.core.enums import ControlMode
from heron.public import PUBLIC_MODULES

APP_NAME = "heron"
"""Spelled here on purpose, like in ``test_invariants``: this file hunts for it."""

PLUGINS = EXAMPLES / "plugins"


def platform_imports(path: Path) -> list[tuple[int, str]]:
    """Platform modules a file imports, relative to the package root.

    ``from <pkg>.core import protocols`` names the module ``core.protocols``;
    ``from <pkg> import core`` names the package ``core`` itself, which opens
    everything under it and is therefore not public.
    """
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head, _, rest = alias.name.partition(".")
                if head == APP_NAME:
                    found.append((node.lineno, rest))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            head, _, rest = node.module.partition(".")
            if head != APP_NAME:
                continue
            for alias in node.names:
                submodule = f"{rest}.{alias.name}" if rest else alias.name
                found.append((node.lineno, submodule if submodule in PUBLIC_MODULES else rest))
    return found


def private_imports(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for line, module in platform_imports(path):
            if module not in PUBLIC_MODULES:
                offenders.append(f"{path.relative_to(root)}:{line}: {module or '<пакет>'}")
    return offenders


@pytest.mark.parametrize("module", sorted(PUBLIC_MODULES))
def test_every_public_module_exists(module: str) -> None:
    importlib.import_module(f"{APP_NAME}.{module}")


def test_a_private_import_is_caught(tmp_path: Path) -> None:
    """The check itself, on a package made to break it."""
    package = tmp_path / "vendor_robot"
    package.mkdir()
    (package / "adapter.py").write_text(
        f"from {APP_NAME}.core.protocols import ControlAdapter\n"
        f"from {APP_NAME}.core import models\n"
        f"import {APP_NAME}.generators.twin\n"
        f"from {APP_NAME}.validator.rules import check_scenario\n"
        f"from {APP_NAME} import cli\n"
        f"import {APP_NAME}\n",
        encoding="utf-8",
    )

    assert private_imports(tmp_path) == [
        f"{Path('vendor_robot', 'adapter.py')}:4: validator.rules",
        f"{Path('vendor_robot', 'adapter.py')}:5: <пакет>",
        f"{Path('vendor_robot', 'adapter.py')}:6: <пакет>",
    ]


def test_robot_packages_import_only_the_public_api() -> None:
    if not PLUGINS.is_dir():
        pytest.skip("пакетов роботов в examples/plugins нет")
    offenders = private_imports(PLUGINS)
    assert not offenders, "пакет робота обходит публичный API:\n" + "\n".join(offenders)


@pytest.mark.parametrize("mode", list(ControlMode))
def test_a_generated_robot_package_imports_only_the_public_api(
    tmp_path: Path, mode: ControlMode
) -> None:
    """The path a user takes starts inside the promise, not beside it (ADR-0056)."""
    write_stub("my_robot", tmp_path, control_mode=mode)
    offenders = private_imports(tmp_path)
    assert not offenders, "заготовка обходит публичный API:\n" + "\n".join(offenders)


def test_platform_tests_do_not_name_the_robot_packages() -> None:
    """A package holds itself to the suite; the platform does not import it (ADR-0056).

    The platform's tests once reached the G1 package by importing its harness. A name
    spelled as data -- a policy called like its task package -- is not a reach.
    """
    if not PLUGINS.is_dir():
        pytest.skip("пакетов роботов в examples/plugins нет")
    names = {
        module.name
        for root in PLUGINS.iterdir()
        for module in (root / "src").glob("*")
        if module.is_dir()
    }
    offenders: list[str] = []
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                spelled = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                spelled = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [f"{path.name}:{node.lineno}: {name}" for name in spelled if name in names]
    assert not offenders, "тесты платформы называют пакеты роботов:\n" + "\n".join(offenders)
