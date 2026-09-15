"""Bringing the scenario up: one launch file, in XML.

ADR-0010 chooses XML over the Python form of the same thing. What the platform
has to express here is a list -- these processes, with these arguments, in these
namespaces -- and a list is data. The Python form would be a program that builds
the list, and a generated program is a thing someone eventually edits.

The order inside is the order of dependence, and each step waits on its own
predecessor rather than on a timer: the spawner waits for a controller manager
that only exists once the simulator has loaded the model, and the model is only
loaded once the world runs. Waiting is expressed with the timeouts the ROS 2
tools already have, because a generated sleep is a guess that fails on a slower
machine.

Every robot lives in a namespace named after its scenario id, not after its
profile: two instances of one profile must not share topics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Final
from xml.etree import ElementTree as ET

from ..core.environment import RequirementKind, needed
from ..core.models import Scenario
from .common import (
    XML_COMMENT,
    GeneratedFile,
    GeneratorError,
    Instance,
    header,
    namespace,
    xml_text,
)
from .description import description_path
from .rviz import RVIZ_FILE, frame_prefix
from .twin import require_twin

LAUNCH_FILE: Final = "bringup.launch.xml"
BUNDLE_VAR: Final = "bundle"

LAUNCH_PACKAGES: Final[dict[str, str]] = {
    "ros_gz_sim": "ros-gz",
    "ros_gz_bridge": "ros-gz",
    "robot_state_publisher": "robot-state-publisher",
    "joint_state_publisher": "joint-state-publisher",
    "rviz2": "desktop",
}
"""ROS 2 packages the world's part of a generated launch may reference, and what
installs each. What a twin starts comes from packages the twin declares among its
requirements, and the generator refuses a node from any other (ADR-0025).

Declared here because a launch file that names a package the environment does
not have does not degrade -- it fails to parse, and the whole scene dies with
it, including the robots that were fine. A missing node for one obstacle took
down a working world exactly that way (ADR-0013).

The set is checked from both sides by a test: nothing in a generated launch may
fall outside it, and everything in it must be installed by the phase 0 script.
Adding a node from a new package is therefore a change to the environment, made
deliberately, and not a line that works on one machine."""

CLOCK_BRIDGE: Final = "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
"""One bridge, one topic: without simulated time every node timestamps with the
wall clock, and a run recorded that way cannot be replayed (ADR-0003)."""

TEXT_PARAM: Final = "str"
"""Declared type of every parameter whose value is free text.

Without it the launch frontend guesses the type by running the value through a
YAML loader, and a robot description is not YAML. The generated header alone is
enough to break it: a line like "Источник: файл.urdf" reads as a mapping inside
a scalar and the whole launch fails to parse before a single process starts.

So the two decisions collide -- ADR-0010 puts a traceable header on every
generated file, and the launch frontend sniffs types -- and this is where they
are reconciled. Declaring the type is also simply correct: the value is text,
and guessing was never wanted."""


def _argument(parent: ET.Element, name: str, default: str) -> None:
    ET.SubElement(parent, "arg", {"name": name, "default": default})


def _simulator(parent: ET.Element, world: str) -> None:
    """The simulator itself, once with a window and once without."""
    launch_file_of_simulator = "$(find-pkg-share ros_gz_sim)/launch/gz_sim.launch.py"
    for condition, extra in (("unless", ""), ("if", " -s --headless-rendering")):
        include = ET.SubElement(
            parent,
            "include",
            {"file": launch_file_of_simulator, condition: "$(var headless)"},
        )
        ET.SubElement(include, "arg", {"name": "gz_args", "value": f"-r -v 3{extra} {world}"})


def _robot(parent: ET.Element, instance: Instance) -> None:
    group = ET.SubElement(parent, "group")
    ET.SubElement(group, "push-ros-namespace", {"namespace": namespace(instance)})

    description = f"$(var {BUNDLE_VAR})/{description_path(instance)}"
    publisher = ET.SubElement(
        group,
        "node",
        {"pkg": "robot_state_publisher", "exec": "robot_state_publisher",
         "name": "robot_state_publisher"},
    )
    ET.SubElement(publisher, "param", {"name": "use_sim_time", "value": "true"})
    ET.SubElement(
        publisher,
        "param",
        {"name": "frame_prefix", "value": frame_prefix(instance), "type": TEXT_PARAM},
    )
    ET.SubElement(
        publisher,
        "param",
        {
            "name": "robot_description",
            "value": f"$(command 'cat {description}')",
            "type": TEXT_PARAM,
        },
    )

    x, y, z = instance.pose
    ET.SubElement(
        group,
        "node",
        {
            "pkg": "ros_gz_sim",
            "exec": "create",
            "name": "spawn",
            "args": (
                f"-topic robot_description -name {instance.id} "
                f"-x {x:g} -y {y:g} -z {z:g}"
            ),
        },
    )

    if not instance.controlled:
        # An obstacle has no controller manager, so nothing reports its joints
        # and the visualization draws it up to its first movable one. Its physics
        # is frozen, so default values are a fact rather than an assumption
        # (ADR-0012).
        ET.SubElement(
            group,
            "node",
            {
                "pkg": "joint_state_publisher",
                "exec": "joint_state_publisher",
                "name": "joint_state_publisher",
            },
        )
        return
    # What drives the robot is the twin's to start; this module does not know
    # whose controller it is writing down (ADR-0021). It does know that a node
    # from a package nobody declared kills the scene (ADR-0013), and a vendor's
    # twin is held to that exactly as ours is.
    twin = require_twin(instance)
    declared = needed(twin.requirements, RequirementKind.ROS2)
    for node in twin.launch_nodes(instance):
        if node.package not in declared:
            raise GeneratorError(
                f"{instance.id}: двойник {type(twin).__name__} запускает узел {node.name!r} "
                f"из пакета {node.package!r}, которого не объявляет в requirements. "
                f"Файл запуска не деградирует: нет пакета — нет всей сцены "
                f"(ADR-0013, ADR-0025)"
            )
        attributes = {"pkg": node.package, "exec": node.executable, "name": node.name}
        if node.arguments:
            attributes["args"] = " ".join(node.arguments)
        ET.SubElement(group, "node", attributes)


def scene_environment(instances: Sequence[Instance]) -> dict[str, str]:
    """Variables every process of the world starts with, by name (ADR-0034).

    Asked of each robot's twin. A variable is the world's, not a robot's -- the
    controller managers of all robots share the simulator's process -- so two
    twins asking one variable for different values describe no world that can run.
    """
    wanted: dict[str, tuple[str, str]] = {}
    for instance in instances:
        if instance.twin is None:
            continue
        for name, value in instance.twin.environment(instance).items():
            held = wanted.setdefault(name, (value, instance.id))
            if held[0] != value:
                raise GeneratorError(
                    f"{instance.id}: двойник требует {name}={value!r}, а двойник робота "
                    f"{held[1]} — {name}={held[0]!r}. Процессы мира работают в одном "
                    f"окружении, выполнить оба нельзя (ADR-0034)"
                )
    return {name: wanted[name][0] for name in sorted(wanted)}


def launch_tree(instances: Sequence[Instance], world: str) -> ET.Element:
    """The launch description: environment, simulator, clock bridge, robots, visualization."""
    launch = ET.Element("launch")
    _argument(launch, "headless", "false")
    _argument(launch, "rviz", "true")
    _argument(launch, BUNDLE_VAR, "$(dirname)")
    _argument(launch, "world", world)

    # Before the first process: a variable set later reaches only what starts later.
    for name, value in scene_environment(instances).items():
        ET.SubElement(launch, "set_env", {"name": name, "value": value})

    _simulator(launch, "$(var world)")
    ET.SubElement(
        launch,
        "node",
        {
            "pkg": "ros_gz_bridge",
            "exec": "parameter_bridge",
            "name": "clock_bridge",
            "args": CLOCK_BRIDGE,
        },
    )

    for instance in instances:
        _robot(launch, instance)

    viewer = ET.SubElement(
        launch,
        "node",
        {
            "pkg": "rviz2",
            "exec": "rviz2",
            "name": "rviz2",
            "args": f"-d $(var {BUNDLE_VAR})/{RVIZ_FILE}",
            "if": "$(var rviz)",
        },
    )
    ET.SubElement(viewer, "param", {"name": "use_sim_time", "value": "true"})
    return launch


def launch_file(
    scenario: Scenario, instances: Sequence[Instance], *, world: str
) -> GeneratedFile:
    """The bringup file. ``world`` is a launch-time reference, not a disk path."""
    head = header(
        "Запуск сцены",
        f"сценарий {scenario.name}, роботов {len(instances)}",
        comment=XML_COMMENT,
    )
    return GeneratedFile(
        path=PurePosixPath(LAUNCH_FILE), content=xml_text(launch_tree(instances, world), head=head)
    )
