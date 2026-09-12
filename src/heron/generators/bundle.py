"""Assembling everything one scenario generates.

The generators each know one format; this module knows the scenario. It is the
only place that reads the registry, resolves paths and decides what a robot's
role means for what gets written, so that adding a format later is adding a
module rather than threading the scenario through one more layer.

The manifest written alongside is what makes the bundle answerable. Artifacts on
a disk otherwise carry no statement about where they came from, and "were these
generated from the current profile?" becomes a question answered by looking at
modification times. With a manifest and deterministic generation it is answered
by regenerating and comparing (ADR-0010).

It also carries, for each driven robot, the names of the control plane the
bundle creates -- its controller, its topics, its message types. Those are what
an adapter has to find at run time, and writing them next to the configuration
that creates them turns "does the adapter look in the right place?" from a
question about two modules agreeing into a line to read off a disk and compare
with ``ros2 topic list`` (ADR-0014).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..core.enums import Role
from ..core.models import RobotProfile, Scenario, ScenarioRobot, World, sim_target
from ..core.urdf import parse_urdf
from ..registry.registry import Registry
from ..ros2 import names
from .common import (
    Bundle,
    GeneratedFile,
    GeneratorError,
    Instance,
    namespace,
    resolve_description,
    sorted_files,
)
from .launch import launch_file
from .ros2_control import robot_files
from .rviz import rviz_file
from .world import WORLD_FILE, world_file

MANIFEST_FILE: Final = "manifest.json"
MANIFEST_SCHEMA_VERSION: Final = 2
"""Bumped when the manifest gains or loses a field. Version 2 added the
``control`` block of every driven robot (ADR-0014)."""
SOLO_ROLE: Final = Role.SCRIPTED
"""Role a lone profile is generated under: driven, but not by a learner."""

GENERATED_BACKENDS: Final = frozenset({"gazebo"})
"""Backends these generators can actually produce artifacts for.

Everything written here -- the hardware component, the system plugin, the world
format, the spawn command -- belongs to one simulator. Generating it under
another backend's name would hand out a bundle that looks right and describes a
world nobody asked for, so an unsupported backend is refused instead."""


def scenario_instances(
    scenario: Scenario, registry: Registry
) -> list[Instance]:
    """One instance per scenario robot, with its description resolved."""
    instances: list[Instance] = []
    for robot in scenario.robots:
        source = registry.source(robot.profile)
        instances.append(
            Instance(
                id=robot.id,
                profile=source.profile,
                description=resolve_description(source.profile, source.path),
                pose=robot.pose,
                role=robot.role,
            )
        )
    return instances


def scenario_for_profile(profile: RobotProfile, backend: str | None = None) -> Scenario:
    """A one-robot scenario, so that a lone profile can be generated for.

    The backend is not invented: it is the simulation adapter the profile
    declares, and a profile declaring several has to say which (ADR-0002 leaves
    exactly one owner of the physics).
    """
    backends = sorted(profile.sim_backends)
    if backend is None:
        if len(backends) != 1:
            raise GeneratorError(
                f"{profile.name}: адаптеров симуляции {len(backends)} "
                f"({', '.join(backends) or '—'}), выберите бэкенд явно — "
                f"в сцене владелец физики один (ADR-0002)"
            )
        backend = backends[0]
    if sim_target(backend) not in profile.adapters:
        raise GeneratorError(
            f"{profile.name}: нет адаптера {sim_target(backend)!r}; "
            f"есть: {', '.join(sorted(profile.adapters)) or '—'}"
        )
    return Scenario(
        schema_version=1,
        name=profile.name,
        world=World(backend=backend),
        robots=(ScenarioRobot(id=profile.name, profile=profile.name, role=SOLO_ROLE),),
    )


def _root_links(instances: Sequence[Instance]) -> dict[str, str]:
    frames: dict[str, str] = {}
    for instance in instances:
        described = parse_urdf(instance.description)
        root = described.root_link
        if root is None:
            raise GeneratorError(
                f"{instance.id}: в описании {instance.description.name} не видно "
                f"единственного корневого звена; дерево звеньев разомкнуто или "
                f"замкнуто в цикл"
            )
        frames[instance.id] = root
    return frames


def _manifest_robot(item: Instance) -> dict[str, Any]:
    """One robot's line in the manifest.

    An obstacle gets no control block, and that is a statement rather than an
    omission: nothing drives it, so there is no topic to name (ADR-0012).
    """
    entry: dict[str, Any] = {
        "id": item.id,
        "profile": item.profile.name,
        "role": str(item.role),
        "pose": list(item.pose),
        "contract_id": item.profile.contract.contract_id,
        "revision": item.profile.contract.revision,
        "description": item.profile.description.urdf,
    }
    if item.controlled:
        entry["control"] = names.control_plane(namespace(item), item.profile)
    return entry


def _manifest(
    scenario: Scenario, instances: Sequence[Instance], files: Sequence[GeneratedFile]
) -> GeneratedFile:
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario": scenario.name,
        "world": {"backend": scenario.world.backend, "file": scenario.world.file},
        "robots": [_manifest_robot(item) for item in instances],
        "files": sorted(str(item.path) for item in files),
    }
    return GeneratedFile(
        path=PurePosixPath(MANIFEST_FILE),
        content=json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def build(
    scenario: Scenario,
    registry: Registry,
    *,
    into: Path,
    scenario_path: Path | None = None,
) -> Bundle:
    """Everything this scenario generates, ready to be written or compared."""
    return _assemble(
        scenario,
        scenario_instances(scenario, registry),
        root=Path(into).resolve(),
        scenario_path=scenario_path,
    )


def build_profile(
    profile: RobotProfile,
    profile_path: Path,
    *,
    into: Path,
    backend: str | None = None,
) -> Bundle:
    """Everything a lone profile generates, under a one-robot scenario.

    The profile is already in hand, so it is not looked up by name again: a
    registry search here would make generation depend on where the file sits
    rather than on what was asked for.
    """
    scenario = scenario_for_profile(profile, backend)
    instance = Instance(
        id=profile.name,
        profile=profile,
        description=resolve_description(profile, profile_path),
        role=SOLO_ROLE,
    )
    return _assemble(scenario, [instance], root=Path(into).resolve())


def _assemble(
    scenario: Scenario,
    instances: list[Instance],
    *,
    root: Path,
    scenario_path: Path | None = None,
) -> Bundle:
    if scenario.world.backend not in GENERATED_BACKENDS:
        raise GeneratorError(
            f"сцена {scenario.name!r}: генераторов для бэкенда "
            f"{scenario.world.backend!r} нет. Порождаются среды только для: "
            f"{', '.join(sorted(GENERATED_BACKENDS))}. Робот с адаптером под "
            f"другой бэкенд проверяется валидатором, но сред для него пока не "
            f"строится"
        )

    files: list[GeneratedFile] = []

    for instance in instances:
        files.extend(robot_files(instance, root))

    if scenario.world.file is None:
        files.append(world_file(scenario, instances))
        world_reference = f"$(var bundle)/{WORLD_FILE}"
    else:
        stated = Path(scenario.world.file)
        if not stated.is_absolute() and scenario_path is not None:
            stated = (scenario_path.parent / stated).resolve()
        world_reference = stated.as_posix()

    files.append(rviz_file(instances, _root_links(instances)))
    files.append(launch_file(scenario, instances, world=world_reference))

    ordered = sorted_files(files)
    return Bundle(root=root, files=(*ordered, _manifest(scenario, instances, ordered)))
