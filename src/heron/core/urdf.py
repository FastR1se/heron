"""Reading a kinematic description.

ADR-0009: the description is an input of the platform, never its output. The
core therefore has to read URDF, and reads it with the standard library --
section 10 promises an install without a simulator or a robot SDK, and a parser
from the ROS ecosystem would break that promise for the sake of an XML document.

Only what the platform compares against a profile is extracted: joint names,
kinds, limits and the link tree. Visual materials, transmissions, sensors and
vendor extensions are left in the tree untouched, because the generator merges
its own blocks into a copy of that tree and everything it does not understand
has to survive the trip.

Units are not converted here. URDF fixes radians by specification; a contract
may declare degrees, and the conversion belongs where the two are compared, next
to the message explaining what was compared with what.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

from .loading import LoadError, SchemaProblem

XACRO_SUFFIX: Final = ".xacro"
"""A template, not a description. The core does not expand it (ADR-0009)."""

COMMANDABLE_TYPES: Final = frozenset({"revolute", "continuous", "prismatic"})
"""Joint kinds a controller can drive through a single command interface."""

ANGULAR_TYPES: Final = frozenset({"revolute", "continuous"})
"""Joint kinds whose limits URDF expresses in radians."""

JOINT_TYPES: Final = COMMANDABLE_TYPES | frozenset({"fixed", "floating", "planar"})


@dataclass(frozen=True, slots=True)
class JointLimit:
    """Bounds a description states for one joint, in URDF units."""

    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None

    @property
    def bounded(self) -> bool:
        return self.lower is not None and self.upper is not None


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    """One joint of a description."""

    name: str
    type: str
    parent: str = ""
    child: str = ""
    limit: JointLimit = field(default_factory=JointLimit)

    @property
    def commandable(self) -> bool:
        """Whether a controller can drive it with one command interface."""
        return self.type in COMMANDABLE_TYPES

    @property
    def angular(self) -> bool:
        """Whether its limits are angles (radians) rather than metres."""
        return self.type in ANGULAR_TYPES


@dataclass(frozen=True, slots=True)
class UrdfModel:
    """A parsed description: what the platform compares, plus the tree itself."""

    name: str
    path: Path
    joints: tuple[UrdfJoint, ...]
    links: tuple[str, ...]
    meshes: tuple[str, ...]
    """Every ``filename`` a ``<mesh>`` refers to, in document order."""

    tree: ET.Element = field(compare=False, repr=False, default_factory=lambda: ET.Element("robot"))
    """The document as read. The generator merges into a copy of it."""

    def joint(self, name: str) -> UrdfJoint | None:
        for item in self.joints:
            if item.name == name:
                return item
        return None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.joints)

    @property
    def commandable_joints(self) -> tuple[UrdfJoint, ...]:
        return tuple(item for item in self.joints if item.commandable)

    @property
    def root_link(self) -> str | None:
        """The link no joint names as a child -- the frame everything hangs on.

        ``None`` when the description has no links, or when every link is some
        joint's child, which means the tree is cyclic and not a robot.
        """
        children = {item.child for item in self.joints}
        roots = [link for link in self.links if link not in children]
        return roots[0] if len(roots) == 1 else None


def is_template(path: Path | str) -> bool:
    """Whether a description path names a template rather than a document."""
    return str(path).endswith(XACRO_SUFFIX)


def _number(
    element: ET.Element, attribute: str, problems: list[SchemaProblem], where: str
) -> float | None:
    raw = element.get(attribute)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        problems.append(SchemaProblem(f"{where}.{attribute}", f"не число: {raw!r}"))
        return None


def parse_urdf_text(text: str, path: Path) -> UrdfModel:
    """Parse a description already in memory, reporting problems like a schema."""
    problems: list[SchemaProblem] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise LoadError(path, [SchemaProblem("<XML>", str(exc))]) from exc

    if root.tag != "robot":
        raise LoadError(
            path, [SchemaProblem("<корень>", f"ожидался <robot>, получен <{root.tag}>")]
        )

    joints: list[UrdfJoint] = []
    for element in root.findall("joint"):
        name = element.get("name")
        kind = element.get("type")
        where = f"joint[{name or '<без имени>'}]"
        if not name:
            problems.append(SchemaProblem(where, "у сустава нет атрибута name"))
            continue
        if kind is None:
            problems.append(SchemaProblem(where, "у сустава нет атрибута type"))
            continue
        if kind not in JOINT_TYPES:
            problems.append(
                SchemaProblem(where, f"неизвестный тип {kind!r}; известны: "
                                     f"{', '.join(sorted(JOINT_TYPES))}")
            )
            continue
        limit_element = element.find("limit")
        limit = JointLimit()
        if limit_element is not None:
            limit = JointLimit(
                lower=_number(limit_element, "lower", problems, f"{where}.limit"),
                upper=_number(limit_element, "upper", problems, f"{where}.limit"),
                effort=_number(limit_element, "effort", problems, f"{where}.limit"),
                velocity=_number(limit_element, "velocity", problems, f"{where}.limit"),
            )
        parent = element.find("parent")
        child = element.find("child")
        joints.append(
            UrdfJoint(
                name=name,
                type=kind,
                parent=(parent.get("link", "") if parent is not None else ""),
                child=(child.get("link", "") if child is not None else ""),
                limit=limit,
            )
        )

    links: list[str] = []
    for element in root.findall("link"):
        name = element.get("name")
        if not name:
            problems.append(SchemaProblem("link", "у звена нет атрибута name"))
            continue
        links.append(name)

    duplicates = sorted({n for n in links if links.count(n) > 1})
    if duplicates:
        problems.append(SchemaProblem("link", f"повторы имён звеньев: {', '.join(duplicates)}"))
    joint_names = [item.name for item in joints]
    duplicates = sorted({n for n in joint_names if joint_names.count(n) > 1})
    if duplicates:
        problems.append(SchemaProblem("joint", f"повторы имён суставов: {', '.join(duplicates)}"))

    if problems:
        raise LoadError(path, problems)

    meshes = [
        filename
        for mesh in root.iter("mesh")
        if (filename := mesh.get("filename")) is not None
    ]
    return UrdfModel(
        name=root.get("name", ""),
        path=path,
        joints=tuple(joints),
        links=tuple(links),
        meshes=tuple(meshes),
        tree=root,
    )


def parse_urdf(path: Path) -> UrdfModel:
    """Read a description from disk.

    A template is refused rather than guessed at: expanding it needs the ROS 2
    toolchain, and pretending it parsed would turn every comparison against the
    profile into a check that silently did nothing (ADR-0009).
    """
    if is_template(path):
        raise LoadError(
            path,
            [
                SchemaProblem(
                    "<файл>",
                    f"это шаблон {XACRO_SUFFIX}, а не описание; разверните его "
                    f"шагом генерации в окружении ROS 2",
                )
            ],
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LoadError(path, [SchemaProblem("<файл>", f"не читается: {exc}")]) from exc
    return parse_urdf_text(text, path)
