"""Whether this machine has what the sides of a link declare (ADR-0025).

Everything else in the validator gives one answer on every machine: a contract
either matches or it does not. This module answers a question about the machine
it runs on, and that is why ``validate`` never calls it -- CI validates the
examples on a machine that has neither ROS 2 nor any robot's SDK, and is right
to. ``env check`` asks it about a whole scene; ``sim run`` about the twins before
a world starts; ``run`` about the adapter before a robot is connected.

"Found" here means found the way the process would find it, not loaded. A module
is located without being imported: importing a DDS binding to check it would
load, in the checking process, the very library whose coexistence with another
one ADR-0022 had to rule on. What that leaves unproven is written in the card.

The machine is handed in, as the launcher is (ADR-0011): what a check concludes
is testable on a machine that has none of what is checked.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol
from xml.etree import ElementTree as ET

from ..core.enums import Role
from ..core.environment import Requirement, RequirementKind
from ..core.models import RobotProfile, Scenario, sim_target
from ..core.plugins import PluginError, available, load
from ..core.protocols import Twin
from ..core.twins import TwinError, find_twin
from ..registry.registry import ProfileNotFoundError, Registry
from .report import Report, error

ADR: Final = "ADR-0025"

KIND_WORDS: Final = {
    RequirementKind.PYTHON: "модуль Python",
    RequirementKind.ROS2: "пакет ROS 2",
    RequirementKind.EXECUTABLE: "программа",
    RequirementKind.VARIABLE: "переменная окружения",
}

SOURCE_HINT: Final = "подключите окружение: source /opt/ros/<дистрибутив>/setup.bash"


@dataclass(frozen=True, slots=True)
class Found:
    """Where a machine found something, and which version it is when it can say."""

    location: str
    version: str | None = None


class Machine(Protocol):
    """What a check may ask of the machine it runs on."""

    def find(self, requirement: Requirement) -> Found | None: ...

    def ros2_sourced(self) -> bool:
        """Whether a ROS 2 environment is sourced in this process at all."""


class LocalMachine:
    """The machine this process runs on, asked the way the process itself would ask."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ
        self._distributions: Mapping[str, list[str]] | None = None

    def find(self, requirement: Requirement) -> Found | None:
        kind = RequirementKind(requirement.kind)
        if kind is RequirementKind.PYTHON:
            return self._python(requirement.name)
        if kind is RequirementKind.ROS2:
            return self._ros2(requirement.name)
        if kind is RequirementKind.EXECUTABLE:
            path = shutil.which(requirement.name, path=self._environ.get("PATH"))
            return Found(location=path) if path is not None else None
        value = self._environ.get(requirement.name, "")
        return Found(location=f"{requirement.name}={value}") if value else None

    def ros2_sourced(self) -> bool:
        return bool(self._environ.get("AMENT_PREFIX_PATH"))

    def _python(self, name: str) -> Found | None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return None
        if spec is None:
            return None
        locations = list(spec.submodule_search_locations or [])
        location = locations[0] if locations else (spec.origin or name)
        return Found(location=str(location), version=self._python_version(name))

    def _python_version(self, name: str) -> str | None:
        """Version of the distribution that provides a top-level module, if one does."""
        if self._distributions is None:
            self._distributions = importlib.metadata.packages_distributions()
        for distribution in self._distributions.get(name, []):
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    def _ros2(self, name: str) -> Found | None:
        """The ament index, searched in the order the sourced prefixes give.

        The same lookup ``ros2 pkg prefix`` makes, done on files: nothing of ROS 2
        is imported, so the check runs where ROS 2 is absent and says so.
        """
        for prefix in self._environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
            if not prefix:
                continue
            marker = Path(prefix, "share", "ament_index", "resource_index", "packages", name)
            if marker.is_file():
                return Found(
                    location=prefix,
                    version=_package_version(Path(prefix, "share", name, "package.xml")),
                )
        return None


def _package_version(manifest: Path) -> str | None:
    try:
        version = ET.parse(manifest).getroot().findtext("version")
    except (OSError, ET.ParseError):
        return None
    return version.strip() if version and version.strip() else None


class UndeclaredRequirementsError(TypeError):
    """A side of a link that does not say what it needs."""


def declared_requirements(owner: object) -> tuple[Requirement, ...]:
    """A side's requirements, refusing a side that declares none.

    There is no default. A side that says nothing is not a side that needs
    nothing -- it is one nobody has asked, and the first machine to find out is
    the one it fails on (the reasoning of ADR-0008).
    """
    declared = getattr(owner, "requirements", None)
    if declared is None or isinstance(declared, str | bytes):
        raise UndeclaredRequirementsError("не объявляет requirements")
    try:
        items = tuple(declared)
    except TypeError:
        raise UndeclaredRequirementsError("requirements — не последовательность") from None
    strangers = [item for item in items if not isinstance(item, Requirement)]
    if strangers:
        raise UndeclaredRequirementsError(
            f"в requirements не требования: {', '.join(repr(item) for item in strangers[:3])}"
        )
    return items


def check_requirements(
    requirements: Iterable[Requirement], *, where: str, machine: Machine
) -> Report:
    """Each requirement against the machine: needs met, prohibitions respected."""
    report = Report()
    for requirement in requirements:
        found = machine.find(requirement)
        word = KIND_WORDS[RequirementKind(requirement.kind)]
        detail = {
            "kind": str(requirement.kind),
            "name": requirement.name,
            "version": requirement.version,
            "forbidden": requirement.forbidden,
            "found": found.version if found is not None else None,
            "location": found.location if found is not None else None,
        }

        if requirement.forbidden:
            if found is None:
                continue
            if found.version is not None and not requirement.admits(found.version):
                continue
            version = found.version or "версия не определяется"
            condition = f", запрещены {requirement.version}" if requirement.version else ""
            report.add(
                error(
                    "env.forbidden",
                    where,
                    f"запрещено: {word} {requirement.name} ({version}{condition}; "
                    f"{found.location}) — {requirement.because}",
                    hint=requirement.hint,
                    adr=ADR,
                    detail=detail,
                )
            )
            continue

        if found is None:
            message = f"нет: {word} {requirement.name}"
            if requirement.version is not None:
                message += f" {requirement.version}"
            hint = requirement.hint
            if requirement.kind is RequirementKind.ROS2 and not machine.ros2_sourced():
                message += " — окружение ROS 2 не подключено"
                hint = f"{hint}; {SOURCE_HINT}"
            if requirement.because:
                message += f" ({requirement.because})"
            report.add(error("env.missing", where, message, hint=hint, adr=ADR, detail=detail))
            continue

        if requirement.version is None:
            continue
        if found.version is None:
            message = (
                f"{word} {requirement.name} есть ({found.location}), но версия не "
                f"определяется, а требуется {requirement.version}"
            )
        elif not requirement.admits(found.version):
            message = (
                f"{word} {requirement.name} {found.version} ({found.location}), "
                f"требуется {requirement.version}"
            )
        else:
            continue
        if requirement.because:
            message += f" ({requirement.because})"
        report.add(
            error("env.version_unmet", where, message, hint=requirement.hint, adr=ADR,
                  detail=detail)
        )
    return report


# --------------------------------------------------------------------------
# Sides of a link.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Checked:
    """A side whose requirements were read and checked."""

    where: str
    requirements: int


@dataclass(slots=True)
class EnvironmentCheck:
    report: Report = field(default_factory=Report)
    checked: list[Checked] = field(default_factory=list)

    def add(self, other: EnvironmentCheck) -> None:
        self.report.extend(other.report.findings)
        self.checked.extend(other.checked)


def _side(owner: object, where: str, machine: Machine, fix: str) -> EnvironmentCheck:
    result = EnvironmentCheck()
    try:
        requirements = declared_requirements(owner)
    except UndeclaredRequirementsError as exc:
        result.report.add(
            error(
                "env.unreadable",
                where,
                f"требования не прочитать: {exc}",
                hint=fix,
                adr=ADR,
            )
        )
        return result
    result.report = check_requirements(requirements, where=where, machine=machine)
    result.checked.append(Checked(where=where, requirements=len(requirements)))
    return result


def check_adapter(plugin: str, *, machine: Machine | None = None) -> EnvironmentCheck:
    """The robot's side: what the process driving it needs (``run``)."""
    where = f"adapter:{plugin}"
    try:
        factory = load("adapters", plugin)
    except PluginError as exc:
        result = EnvironmentCheck()
        result.report.add(
            error(
                "env.unreadable",
                where,
                f"адаптер не загружается на этой машине, и его требований не прочитать — {exc}",
                hint="поставьте пакет робота, который объявляет этот адаптер",
                adr=ADR,
                detail={"plugin": plugin},
            )
        )
        return result
    return _side(
        factory,
        where,
        machine or LocalMachine(),
        fix="адаптер объявляет requirements: последовательность Requirement, пустую, "
        "если связи ничего не нужно",
    )


def check_twin(twin: Twin, *, machine: Machine | None = None) -> EnvironmentCheck:
    """The world's side: what the process starting a scene needs for this twin."""
    return _side(
        twin,
        f"twin:{twin.adapter}@{twin.backend}",
        machine or LocalMachine(),
        fix="двойник объявляет requirements: последовательность Requirement",
    )


def _driven(
    scenario: Scenario, registry: Registry, robot_id: str | None
) -> Sequence[tuple[str, RobotProfile]]:
    """Driven robots and their profiles. What cannot be read is the validator's to say."""
    driven: list[tuple[str, RobotProfile]] = []
    for robot in scenario.robots:
        if robot.role is Role.STATIC or (robot_id is not None and robot.id != robot_id):
            continue
        try:
            driven.append((robot.id, registry.source(robot.profile).profile))
        except ProfileNotFoundError:
            continue
    return driven


def check_environment(
    scenario: Scenario,
    registry: Registry,
    *,
    world: bool = True,
    robots: bool = True,
    robot_id: str | None = None,
    machine: Machine | None = None,
) -> EnvironmentCheck:
    """Both sides of every driven robot's link in a scene, each side checked once.

    The world's side only in a world the platform starts -- a backend installed
    as a plugin -- for the reason ``twin.*`` rules give (ADR-0024). A twin that
    cannot be found is a validator finding (``twin.missing``, ``twin.unloadable``)
    and is not repeated here. An adapter that cannot be loaded is only a warning
    to the validator, which answers for every machine; on this one it means the
    robot cannot be driven, so here it is an error.
    """
    machine = machine or LocalMachine()
    backend = scenario.world.backend
    target = sim_target(backend)
    world_started = world and backend in available("sim")
    result = EnvironmentCheck()
    seen: set[str] = set()

    for _, profile in _driven(scenario, registry, robot_id):
        spec = profile.adapters.get(target)
        if spec is None:
            continue
        if robots and f"adapter:{spec.plugin}" not in seen:
            seen.add(f"adapter:{spec.plugin}")
            result.add(check_adapter(spec.plugin, machine=machine))
        if not world_started:
            continue
        try:
            twin = find_twin(spec.plugin, backend)
        except TwinError:
            continue
        if twin is None or f"twin:{twin.adapter}@{twin.backend}" in seen:
            continue
        seen.add(f"twin:{twin.adapter}@{twin.backend}")
        result.add(check_twin(twin, machine=machine))
    return result
