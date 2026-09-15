"""Reading an MJCF description.

The second language a robot can be described in (ADR-0031). URDF is what a
Gazebo scene is started from; MJCF is what a MuJoCo world loads -- the vendor's
twin loads it itself, and the platform's part is to compare it with the profile
before the twin is started, the way ``urdf.*`` rules compare the URDF.

Read with the standard library, for the reason ``core.urdf`` gives: the core
installs without a simulator (section 10), and MuJoCo's own parser is the
simulator.

Only what the platform compares is extracted: joints and their limits, actuators
and the joints they drive, meshes, the physics step, and every file a document
pulls in. Three things MuJoCo does are mirrored, because reading the raw
attributes would be wrong in each:

* ``<include>`` is replaced by the children of the included document, and its
  path is read from the directory of the main file;
* an attribute an element leaves out comes from its ``<default>`` class -- the
  element's ``class``, else the ``childclass`` of the nearest body, else the
  top-level default -- walking up to the parent classes;
* angles are degrees unless ``<compiler angle="radian">``.

Unlike ``core.urdf``, angles are converted here, to radians. A URDF number has
one unit by specification; an MJCF number has whichever unit a ``<compiler>``
element somewhere in the included files says, and a model handing such numbers
out unconverted would hand out numbers nobody can read without the document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

from .loading import LoadError, SchemaProblem

MUJOCO: Final = "mujoco"
"""The backend whose world is an MJCF scene: ``world.backend`` and ``sim.mujoco``."""

DEFAULT_TIMESTEP: Final = 0.002
"""MuJoCo's physics step when no ``<option timestep>`` says otherwise, seconds."""

JOINT_TYPES: Final = frozenset({"hinge", "slide", "ball", "free"})
COMMANDABLE_TYPES: Final = frozenset({"hinge", "slide"})
"""Joints one actuator drives with one scalar."""

ANGULAR_TYPES: Final = frozenset({"hinge", "ball"})
DEFAULT_JOINT_TYPE: Final = "hinge"

MAIN_CLASS: Final = "main"
"""The class of the top-level ``<default>``."""

ANGLE_UNITS: Final = frozenset({"degree", "radian"})
DEFAULT_ANGLE: Final = "degree"


@dataclass(frozen=True, slots=True)
class MjcfJoint:
    """One named joint of a model."""

    name: str
    type: str
    range: tuple[float, float] | None = None
    """Limits when the joint is limited: radians for angular joints, metres for slides."""

    effort: float | None = None
    """Largest actuator force the joint accepts (``actuatorfrcrange``), N·m or N."""

    @property
    def commandable(self) -> bool:
        return self.type in COMMANDABLE_TYPES

    @property
    def angular(self) -> bool:
        return self.type in ANGULAR_TYPES


@dataclass(frozen=True, slots=True)
class MjcfActuator:
    """One actuator, in the order the model declares it."""

    name: str
    kind: str
    """The element: ``motor``, ``position``, ``general``, ..."""

    joint: str | None
    """The joint it transmits to, when it transmits to a joint."""

    force: float | None = None
    """Largest force it applies, when bounded: ``forcerange``, or ``ctrlrange``
    times the gear for a motor, whose control is the force."""


@dataclass(frozen=True, slots=True)
class MjcfModel:
    """A parsed MJCF document with its includes resolved."""

    name: str
    path: Path
    joints: tuple[MjcfJoint, ...]
    actuators: tuple[MjcfActuator, ...]
    meshes: tuple[tuple[str, Path], ...]
    """Each mesh file as written, and where MuJoCo will look for it."""

    includes: tuple[Path, ...]
    """Every file pulled in by ``<include>``, resolved, in the order met."""

    timestep: float = DEFAULT_TIMESTEP
    tree: ET.Element = field(
        compare=False, repr=False, default_factory=lambda: ET.Element("mujoco")
    )
    """The document with its includes expanded."""

    def joint(self, name: str) -> MjcfJoint | None:
        for item in self.joints:
            if item.name == name:
                return item
        return None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.joints)

    @property
    def files(self) -> tuple[Path, ...]:
        """The document and everything it includes: what loading it reads."""
        return (self.path, *self.includes)

    def loads(self, path: Path) -> bool:
        """Whether loading this document reads that file."""
        wanted = path.resolve()
        return any(item == wanted for item in self.files)

    def actuators_of(self, joint: str) -> tuple[MjcfActuator, ...]:
        return tuple(item for item in self.actuators if item.joint == joint)

    def effort(self, joint: str) -> float | None:
        """The smallest bound on force any part of the model puts on the joint."""
        described = self.joint(joint)
        bounds = [item.force for item in self.actuators_of(joint) if item.force is not None]
        if described is not None and described.effort is not None:
            bounds.append(described.effort)
        return min(bounds) if bounds else None


# --------------------------------------------------------------------------


def _read(path: Path, problems: list[SchemaProblem], where: str) -> ET.Element | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(SchemaProblem(where, f"не читается {path}: {exc}"))
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        problems.append(SchemaProblem(where, f"{path.name}: {exc}"))
        return None
    if root.tag != "mujoco":
        problems.append(
            SchemaProblem(where, f"{path.name}: ожидался <mujoco>, получен <{root.tag}>")
        )
        return None
    return root


def _expand(
    element: ET.Element,
    base: Path,
    chain: tuple[Path, ...],
    includes: list[Path],
    problems: list[SchemaProblem],
) -> None:
    """Replace every ``<include>`` under ``element`` by the included children, in place."""
    for child in list(element):
        if child.tag != "include":
            _expand(child, base, chain, includes, problems)
            continue
        position = list(element).index(child)
        element.remove(child)
        stated = child.get("file")
        if not stated:
            problems.append(SchemaProblem("include", "нет атрибута file"))
            continue
        target = Path(stated)
        target = (target if target.is_absolute() else base / target).resolve()
        if target in chain:
            problems.append(SchemaProblem("include", f"{stated}: включение по кругу"))
            continue
        included = _read(target, problems, f"include[{stated}]")
        if included is None:
            continue
        includes.append(target)
        _expand(included, base, (*chain, target), includes, problems)
        for offset, item in enumerate(list(included)):
            element.insert(position + offset, item)


def _floats(
    raw: str | None, count: int, where: str, problems: list[SchemaProblem]
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    try:
        values = tuple(float(item) for item in raw.split())
    except ValueError:
        problems.append(SchemaProblem(where, f"не числа: {raw!r}"))
        return None
    if len(values) != count:
        problems.append(SchemaProblem(where, f"ожидалось {count} числа, получено {raw!r}"))
        return None
    return values


def _merged(root: ET.Element, tag: str) -> dict[str, str]:
    """Attributes of every top-level ``<tag>``, later ones overriding earlier ones."""
    merged: dict[str, str] = {}
    for item in root.findall(tag):
        merged.update(item.attrib)
    return merged


def _limited(flag: str | None, present: bool, autolimits: bool) -> bool:
    """MuJoCo's ``limited``: ``true``, ``false``, or ``auto`` -- a stated range limits."""
    if flag == "true":
        return present
    if flag == "false":
        return False
    return present and autolimits


class _Defaults:
    """``<default>`` classes and the lookup MuJoCo does through them."""

    def __init__(self, root: ET.Element, problems: list[SchemaProblem]) -> None:
        self.classes: dict[str, list[ET.Element]] = {}
        self.parents: dict[str, str | None] = {}
        for top in root.findall("default"):
            self._collect(top, None, problems)

    def _collect(
        self, element: ET.Element, parent: str | None, problems: list[SchemaProblem]
    ) -> None:
        name = element.get("class", MAIN_CLASS if parent is None else "")
        if not name:
            problems.append(SchemaProblem("default", "вложенный <default> без class"))
            return
        self.classes.setdefault(name, []).append(element)
        self.parents.setdefault(name, parent)
        for child in element.findall("default"):
            self._collect(child, name, problems)

    def known(self, name: str) -> bool:
        return name == MAIN_CLASS or name in self.classes

    def attribute(self, tags: tuple[str, ...], attribute: str, name: str) -> str | None:
        current: str | None = name
        while current is not None:
            for default in self.classes.get(current, []):
                for tag in tags:
                    for item in default.findall(tag):
                        if attribute in item.attrib:
                            return item.attrib[attribute]
            current = self.parents.get(current)
        return None


def parse_mjcf(path: Path) -> MjcfModel:
    """Read a model from disk, following its includes, reporting problems like a schema."""
    problems: list[SchemaProblem] = []
    main = path.resolve()
    root = _read(main, problems, "<файл>")
    if root is None:
        raise LoadError(path, problems)
    includes: list[Path] = []
    _expand(root, main.parent, (main,), includes, problems)

    compiler = _merged(root, "compiler")
    angle = compiler.get("angle", DEFAULT_ANGLE)
    if angle not in ANGLE_UNITS:
        problems.append(SchemaProblem("compiler.angle", f"{angle!r}: известны degree, radian"))
    to_model = math.radians if angle == DEFAULT_ANGLE else float
    autolimits = compiler.get("autolimits", "true") != "false"
    defaults = _Defaults(root, problems)

    def attribute(element: ET.Element, tags: tuple[str, ...], name: str, cls: str) -> str | None:
        stated = element.get(name)
        return stated if stated is not None else defaults.attribute(tags, name, cls)

    joints: list[MjcfJoint] = []

    def joint_of(element: ET.Element, childclass: str | None) -> None:
        name = element.get("name")
        if not name:
            return
        where = f"joint[{name}]"
        cls = element.get("class", childclass or MAIN_CLASS)
        if not defaults.known(cls):
            problems.append(SchemaProblem(where, f"класс {cls!r} не объявлен в <default>"))
        tags = ("joint",)
        kind = "free" if element.tag == "freejoint" else (
            attribute(element, tags, "type", cls) or DEFAULT_JOINT_TYPE
        )
        if kind not in JOINT_TYPES:
            known = ", ".join(sorted(JOINT_TYPES))
            problems.append(SchemaProblem(where, f"тип {kind!r}; известны: {known}"))
            return
        bounds: tuple[float, float] | None = None
        raw = _floats(attribute(element, tags, "range", cls), 2, f"{where}.range", problems)
        if raw is not None and _limited(attribute(element, tags, "limited", cls), True, autolimits):
            low, high = raw
            bounds = (to_model(low), to_model(high)) if kind in ANGULAR_TYPES else (low, high)
        effort: float | None = None
        force = _floats(
            attribute(element, tags, "actuatorfrcrange", cls), 2, f"{where}.actuatorfrcrange",
            problems,
        )
        flag = attribute(element, tags, "actuatorfrclimited", cls)
        if force is not None and _limited(flag, True, autolimits):
            effort = max(abs(force[0]), abs(force[1]))
        joints.append(MjcfJoint(name=name, type=kind, range=bounds, effort=effort))

    def walk(element: ET.Element, childclass: str | None) -> None:
        for child in element:
            if child.tag == "body":
                walk(child, child.get("childclass", childclass))
            elif child.tag in ("joint", "freejoint"):
                joint_of(child, childclass)

    for worldbody in root.findall("worldbody"):
        walk(worldbody, None)

    actuators: list[MjcfActuator] = []
    for section in root.findall("actuator"):
        for element in section:
            cls = element.get("class", MAIN_CLASS)
            where = f"actuator[{element.get('name') or element.tag}]"
            tags = (element.tag, "general")
            force = None
            forcerange = _floats(attribute(element, tags, "forcerange", cls), 2,
                                 f"{where}.forcerange", problems)
            if forcerange is not None and _limited(
                attribute(element, tags, "forcelimited", cls), True, autolimits
            ):
                force = max(abs(forcerange[0]), abs(forcerange[1]))
            elif element.tag == "motor":
                ctrl = _floats(attribute(element, tags, "ctrlrange", cls), 2,
                               f"{where}.ctrlrange", problems)
                gear = (attribute(element, tags, "gear", cls) or "1").split()
                if ctrl is not None and _limited(
                    attribute(element, tags, "ctrllimited", cls), True, autolimits
                ):
                    try:
                        force = max(abs(ctrl[0]), abs(ctrl[1])) * abs(float(gear[0]))
                    except (ValueError, IndexError):
                        problems.append(SchemaProblem(f"{where}.gear", f"не число: {gear!r}"))
            actuators.append(
                MjcfActuator(
                    name=element.get("name", ""),
                    kind=element.tag,
                    joint=element.get("joint"),
                    force=force,
                )
            )

    meshdir = compiler.get("meshdir", compiler.get("assetdir"))
    mesh_base = main.parent / meshdir if meshdir else main.parent
    meshes: list[tuple[str, Path]] = []
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            stated = mesh.get("file")
            if stated is None:
                continue
            target = Path(stated)
            meshes.append((stated, target if target.is_absolute() else mesh_base / target))

    timestep = DEFAULT_TIMESTEP
    option = _merged(root, "option").get("timestep")
    if option is not None:
        try:
            timestep = float(option)
        except ValueError:
            problems.append(SchemaProblem("option.timestep", f"не число: {option!r}"))
        if timestep <= 0:
            problems.append(SchemaProblem("option.timestep", f"шаг не положителен: {option!r}"))

    names = [item.name for item in joints]
    duplicates = sorted({item for item in names if names.count(item) > 1})
    if duplicates:
        problems.append(SchemaProblem("joint", f"повторы имён суставов: {', '.join(duplicates)}"))

    if problems:
        raise LoadError(path, problems)
    return MjcfModel(
        name=root.get("model", ""),
        path=main,
        joints=tuple(joints),
        actuators=tuple(actuators),
        meshes=tuple(meshes),
        includes=tuple(includes),
        timestep=timestep,
        tree=root,
    )
