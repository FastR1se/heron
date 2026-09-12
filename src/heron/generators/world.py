"""The world: ground, light, physics.

Deliberately the smallest world that is a world. Everything in it is either
fixed by the format or derived from the scenario, and nothing is decoration:
a generator that invents furniture produces a scene the scenario does not
describe, and the scenario is what the validator checked.

The one derived quantity worth stating is the physics step. A robot whose
contract promises state at some rate cannot get it from a world that steps more
slowly, so the step comes from the fastest ``state_rate_hz`` in the scenario and
is capped at the usual millisecond. Deriving it removes a whole class of
mismatch instead of adding a rule that detects it -- the same move as ADR-0004,
one step earlier.

A scenario that names its own world file gets that file used as-is: ADR-0009
makes a stated world an input, exactly like a stated description.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Final
from xml.etree import ElementTree as ET

from ..core.models import Scenario
from .common import XML_COMMENT, GeneratedFile, Instance, header, xml_text

SDF_VERSION: Final = "1.10"
WORLD_FILE: Final = "world.sdf"
MAX_STEP_SIZE: Final = 0.001
"""Coarsest step the platform will ask a world to take, in seconds."""

SYSTEM_PLUGINS: Final = (
    ("gz-sim-physics-system", "gz::sim::systems::Physics"),
    ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
    ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
)
"""Minimum set: step the world, accept spawn requests, publish the scene."""


def physics_step(instances: Sequence[Instance]) -> float:
    """Step small enough for the fastest state rate in the scenario."""
    rates = [item.profile.contract.state_rate_hz for item in instances]
    fastest = max(rates) if rates else 1.0 / MAX_STEP_SIZE
    return min(MAX_STEP_SIZE, 1.0 / fastest)


def _ground_plane(parent: ET.Element) -> None:
    model = ET.SubElement(parent, "model", {"name": "ground_plane"})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": "link"})

    for tag, name in (("collision", "collision"), ("visual", "visual")):
        element = ET.SubElement(link, tag, {"name": name})
        geometry = ET.SubElement(element, "geometry")
        plane = ET.SubElement(geometry, "plane")
        ET.SubElement(plane, "normal").text = "0 0 1"
        ET.SubElement(plane, "size").text = "100 100"
        if tag == "visual":
            material = ET.SubElement(element, "material")
            ET.SubElement(material, "ambient").text = "0.8 0.8 0.8 1"
            ET.SubElement(material, "diffuse").text = "0.8 0.8 0.8 1"
            ET.SubElement(material, "specular").text = "0.8 0.8 0.8 1"


def _sun(parent: ET.Element) -> None:
    light = ET.SubElement(parent, "light", {"type": "directional", "name": "sun"})
    ET.SubElement(light, "cast_shadows").text = "true"
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"


def world_tree(scenario: Scenario, instances: Sequence[Instance]) -> ET.Element:
    """The world element, with everything the scenario decides in it."""
    sdf = ET.Element("sdf", {"version": SDF_VERSION})
    world = ET.SubElement(sdf, "world", {"name": scenario.name})

    physics = ET.SubElement(world, "physics", {"name": "derived", "type": "ignored"})
    ET.SubElement(physics, "max_step_size").text = f"{physics_step(instances):g}"
    ET.SubElement(physics, "real_time_factor").text = "1.0"

    for filename, name in SYSTEM_PLUGINS:
        ET.SubElement(world, "plugin", {"filename": filename, "name": name})

    _sun(world)
    _ground_plane(world)
    return sdf


def world_file(scenario: Scenario, instances: Sequence[Instance]) -> GeneratedFile:
    """The generated world. Only called when the scenario names no file."""
    rates = ", ".join(
        f"{item.id} {item.profile.contract.state_rate_hz:g} Гц" for item in instances
    )
    head = header(
        "Мир сцены",
        f"сценарий {scenario.name}; шаг физики выведен из частот состояния ({rates})",
        comment=XML_COMMENT,
    )
    return GeneratedFile(
        path=PurePosixPath(WORLD_FILE), content=xml_text(world_tree(scenario, instances), head=head)
    )
