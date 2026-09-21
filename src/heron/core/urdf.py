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

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Final
from xml.etree import ElementTree as ET

from .loading import LoadError, SchemaProblem

XACRO_SUFFIX: Final = ".xacro"
"""A template, not a description. The core does not expand it (ADR-0009)."""

COMMANDABLE_TYPES: Final = frozenset({"revolute", "continuous", "prismatic"})
"""Joint kinds a controller can drive through a single command interface."""

ANGULAR_TYPES: Final = frozenset({"revolute", "continuous"})
"""Joint kinds whose limits URDF expresses in radians."""

JOINT_TYPES: Final = COMMANDABLE_TYPES | frozenset({"fixed", "floating", "planar"})

UNRESOLVABLE_SCHEMES: Final = ("package://", "model://")
"""Mesh references only a ROS 2 or simulator resource path can resolve."""

FILE_SCHEME: Final = "file://"

WORLD_LINK: Final = "world"
"""The link simulators and ``robot_state_publisher`` treat as the fixed frame."""


STL_HEADER: Final = 84
"""Bytes before the first triangle of a binary STL: 80 of comment, 4 of count."""

STL_TRIANGLE: Final = 50
"""Bytes one triangle of a binary STL occupies: twelve floats and two spare."""

STL_FACET: Final = b"facet normal"
"""How the ASCII form of STL opens each triangle."""


@dataclass(frozen=True, slots=True)
class CollisionMesh:
    """A mesh a link offers the physics engine to collide with.

    Separate from the visual mesh of the same link and never interchangeable with
    it, however often a CAD export writes one file into both: what is drawn costs
    a frame, what collides costs every physics step.
    """

    link: str
    filename: str


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
    mimic: str | None = None
    """The joint this one follows, if the description makes it a ``<mimic>``.

    Read, not yet judged: a follower is left out of a stub's ``joint_order``
    (ADR-0074), and no rule compares it with a profile."""

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

    collision_meshes: tuple[CollisionMesh, ...] = ()
    """Those of ``meshes`` a ``<collision>`` names, with the link that owns each.

    A subset, read in the same pass over the same document, not a second list of
    meshes: what makes it worth naming apart is who pays for it. A visual mesh is
    drawn; a collision mesh is walked by the solver on every step of the world."""

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

    @property
    def fixed_to_world(self) -> bool:
        """Whether the description bolts its tree to the world frame.

        By convention a link named ``world`` is the fixed frame, and a fixed joint
        from it holds the robot in place. Without one the base is free: whatever
        the joints do, gravity decides where the robot ends up.
        """
        return any(item.parent == WORLD_LINK and item.type == "fixed" for item in self.joints)


def resolve_mesh(reference: str, description: Path, mesh_dir: str | None = None) -> Path | None:
    """Where a mesh reference points on disk, or ``None`` if only a resource path can say.

    One resolution for two readers: the rule that checks a mesh exists and the
    generator that rewrites the reference in its copy (ADR-0017). Resolved in two
    places, the validator could approve a file the simulator never loads -- the
    shape every failure of phase 2 had.
    """
    if reference.startswith(UNRESOLVABLE_SCHEMES):
        return None
    target = Path(reference.removeprefix(FILE_SCHEME))
    if not target.is_absolute():
        base = description.parent / mesh_dir if mesh_dir else description.parent
        target = base / target
    return target


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
        mimic = element.find("mimic")
        joints.append(
            UrdfJoint(
                name=name,
                type=kind,
                parent=(parent.get("link", "") if parent is not None else ""),
                child=(child.get("link", "") if child is not None else ""),
                limit=limit,
                mimic=(mimic.get("joint") if mimic is not None else None),
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
    collision_meshes = [
        CollisionMesh(link=element.get("name", ""), filename=filename)
        for element in root.findall("link")
        for collision in element.findall("collision")
        for mesh in collision.iter("mesh")
        if (filename := mesh.get("filename")) is not None
    ]
    return UrdfModel(
        name=root.get("name", ""),
        path=path,
        joints=tuple(joints),
        links=tuple(links),
        meshes=tuple(meshes),
        collision_meshes=tuple(collision_meshes),
        tree=root,
    )


def triangle_count(path: Path) -> int | None:
    """Triangles in a mesh file, or ``None`` when this reader cannot say.

    STL only, and deliberately: it is what CAD writes when a machine is exported
    for simulation, which is where a collision mesh of a hundred thousand
    triangles comes from in the first place. For a binary file the count is a
    field of the format and costs one read of 84 bytes; for the ASCII form it is
    the number of facets. Any other format is answered ``None`` rather than
    guessed at -- a warning carrying a number nobody measured is worse than no
    warning, because it is believed.
    """
    if path.suffix.lower() != ".stl":
        return None
    try:
        with path.open("rb") as handle:
            head = handle.read(STL_HEADER)
            if len(head) < STL_HEADER:
                return None
            declared = int.from_bytes(head[80:STL_HEADER], "little")
            if path.stat().st_size == STL_HEADER + declared * STL_TRIANGLE:
                return declared
            # Not the binary layout: the ASCII form, where one facet is one
            # triangle. Counted on the bytes, because the file is someone else's
            # and need not decode.
            handle.seek(0)
            return sum(chunk.count(STL_FACET) for chunk in _chunks(handle))
    except OSError:
        return None


def _chunks(handle: IO[bytes], size: int = 1 << 20) -> Iterator[bytes]:
    """The file in pieces that overlap by one byte less than the phrase sought.

    One less, exactly: enough that a facet split across a boundary is still
    whole in some piece, and too little for a whole facet to sit inside an
    overlap and be counted twice. Read in pieces because a collision mesh worth
    warning about can be a hundred megabytes, and the reader that measures it
    should not need them in memory.
    """
    tail = b""
    while piece := handle.read(size):
        yield tail + piece
        tail = piece[-(len(STL_FACET) - 1) :]


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
