"""What every generator shares: a file it produced, and a bundle of such files.

ADR-0010 fixes three properties, and they all live here rather than in each
generator, because a property held by four modules separately is a property held
by none:

* a generated file is a value -- a relative path and text -- until something
  explicitly writes it, so generating and writing can be told apart, and
  ``--check`` is the same code path as ``--write``;
* every file opens with a header naming what produced it and from what, so that
  a file found on a disk months later can be traced back without guessing;
* generation is deterministic, so a comparison with what is already on disk is
  meaningful.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from ..core.branding import APP_TITLE
from ..core.enums import Role
from ..core.models import RobotProfile

if TYPE_CHECKING:
    from .twin import GazeboTwin

XML_COMMENT = ("<!--", "-->")
HASH_COMMENT = ("#", "")


class GeneratorError(RuntimeError):
    """A value in the profile has no expression in the target format.

    Not a validator finding: a finding describes input a human can fix and the
    validator keeps going. This is the generator refusing to invent something it
    was not given.
    """


@dataclass(frozen=True, slots=True)
class Instance:
    """One robot about to be brought up in a world.

    A profile describes a kind of robot; an instance is that kind placed in a
    scenario, with an identifier and a pose. Two instances may share a profile,
    which is why everything named after the robot is named after ``id`` and
    never after ``profile.name``: topics, namespaces and generated directories
    have to stay distinct for twins.
    """

    id: str
    profile: RobotProfile
    description: Path
    """Input description, already resolved against the profile's directory."""

    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    role: Role = Role.STATIC
    twin: GazeboTwin | None = None
    """The other side of the robot's link in the world (ADR-0021).

    Attached when the scene is assembled, and only to a driven robot: an
    obstacle has no link, so nothing speaks for it."""

    @property
    def directory(self) -> PurePosixPath:
        """Where this instance's artifacts live inside the bundle."""
        return PurePosixPath(self.id)

    @property
    def controlled(self) -> bool:
        """Whether anything drives this robot's joints.

        An obstacle has a description and a pose and nothing else: generating
        control interfaces for it would claim a controller that never runs
        (ARCHITECTURE.md, section 7).
        """
        return self.role is not Role.STATIC


def namespace(instance: Instance) -> str:
    """ROS 2 namespace of one robot: its scenario id, and nothing else.

    Named here rather than spelled out at each use, because four generators
    depend on the three of them agreeing -- the plugin that starts the controller
    manager, the launch file that pushes the namespace, the parameter file that
    has to reach that node, and the frames the visualization looks for.
    """
    return instance.id


def node_name(instance: Instance, node: str) -> str:
    """Fully qualified name of a node belonging to this robot.

    Parameter files are matched against a node's *fully qualified* name, so a
    file written for ``controller_manager`` never reaches
    ``/arm1/controller_manager`` and every parameter in it is silently ignored --
    including the ones that say which controllers exist. The namespace is known,
    so the name is derived instead of being left to a wildcard.
    """
    return f"/{namespace(instance)}/{node}"


def resolve_description(profile: RobotProfile, profile_path: Path) -> Path:
    """Where a profile's description file is, relative to the profile itself.

    Relative and not to the working directory: a profile plus its description
    move together, and a path that only works from one directory would make a
    robot depend on where its owner stood (ADR-0009).
    """
    stated = Path(profile.description.urdf)
    if stated.is_absolute():
        return stated
    return (profile_path.parent / stated).resolve()


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    """One artifact, as a value. ``path`` is relative to the bundle root."""

    path: PurePosixPath
    content: str

    def resolve(self, root: Path) -> Path:
        return root / Path(*self.path.parts)


def header(
    what: str,
    source: str,
    *,
    contract_id: str | None = None,
    comment: tuple[str, str] = HASH_COMMENT,
) -> str:
    """The block every generated file opens with (ADR-0010)."""
    lines = [
        f"{what}. Файл порождён платформой {APP_TITLE}; правки будут потеряны.",
        f"Источник: {source}.",
    ]
    if contract_id is not None:
        lines.append(f"Контракт на момент порождения: {contract_id}.")
    lines.append("Кинематику платформа не сочиняет: описание — вход (ADR-0009).")

    opening, closing = comment
    if closing:
        body = "\n".join(f"  {line}" for line in lines)
        return f"{opening}\n{body}\n{closing}\n"
    return "\n".join(f"{opening} {line}" for line in lines) + "\n"


def xml_text(root: ET.Element, *, head: str) -> str:
    """Serialize an XML tree the same way every time (ADR-0010)."""
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return f'<?xml version="1.0"?>\n{head}{body}\n'


@dataclass(frozen=True, slots=True)
class Difference:
    """One file that is missing or differs from what would be generated now."""

    path: PurePosixPath
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class Bundle:
    """Everything one scenario or profile generates, plus where it goes."""

    root: Path
    files: tuple[GeneratedFile, ...]

    def write(self) -> list[Path]:
        """Write every file, creating directories. Returns what was written."""
        written: list[Path] = []
        for item in self.files:
            target = item.resolve(self.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8", newline="\n")
            written.append(target)
        return written

    def differences(self) -> list[Difference]:
        """How what is on disk differs from what would be generated now.

        Empty means the artifacts are current. This is the whole point of
        determinism: staleness becomes a checkable fact (ADR-0010).
        """
        found: list[Difference] = []
        for item in self.files:
            target = item.resolve(self.root)
            if not target.is_file():
                found.append(Difference(item.path, "файла нет"))
                continue
            if target.read_text(encoding="utf-8") != item.content:
                found.append(Difference(item.path, "содержимое отличается"))
        return found

    def paths(self) -> list[PurePosixPath]:
        return [item.path for item in self.files]

    def merged(self, other: Iterable[GeneratedFile]) -> Bundle:
        return Bundle(root=self.root, files=(*self.files, *other))


def sorted_files(files: Sequence[GeneratedFile]) -> tuple[GeneratedFile, ...]:
    """Files in a stable order, so a manifest lists them the same way twice."""
    return tuple(sorted(files, key=lambda item: str(item.path)))
