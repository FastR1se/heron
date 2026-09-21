"""The G1's twin in Gazebo, as the platform asks it (ADR-0033, ADR-0034).

Checked here is everything the twin writes; nothing compiles the controller or starts
a world. That the controller builds is the installer's (`scripts/install.sh install unitree_g1`),
that the robot moves -- the Linux runs' (scripts/runs/). What this machine can check about the
C++ side is that it speaks the parameter names, class and package the Python side writes --
read off its source, because a name spelled apart in two languages is how phase 2 lost scenes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest
import yaml
from unitree_g1.bridge import (
    BRIDGE_PACKAGE,
    CONTROLLER,
    CONTROLLER_TYPE,
    IMU_INTERFACES,
    IMU_LINK,
    MESSAGES_PACKAGE,
    PARAMETERS,
    RMW,
    G1GazeboTwin,
)
from unitree_g1.channel import LinkConfig
from unitree_g1.slots import LAYOUT, slots_of
from unitree_g1.testing import (
    JOINTS_23DOF,
    JOINTS_UPPER_BODY_6,
    LINK_PARAMS,
    PLUGIN,
    profile_23dof,
    profile_upper_body,
)

from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionType, AngleUnit, ObservationType, Role, TwinKind
from heron.core.environment import RequirementKind, needed
from heron.core.models import AdapterSpec, RobotProfile
from heron.core.urdf import UrdfModel, parse_urdf_text
from heron.generators import ros2_control
from heron.generators.common import GeneratorError, Instance
from heron.generators.twin import GAZEBO, GazeboTwin

ROS2_DIR = Path(__file__).resolve().parents[1] / "ros2" / BRIDGE_PACKAGE
ROOT = Path("/bundle")


def gazebo_profile(**overrides: Any) -> RobotProfile:
    overrides.setdefault(
        "adapters", {"sim.gazebo": AdapterSpec(plugin=PLUGIN, params=dict(LINK_PARAMS))}
    )
    return profile_23dof(**overrides)


def contract(joints: Sequence[str], *, gains: bool = True) -> Contract:
    dof = len(joints)
    return Contract(
        revision=1,
        joint_order=list(joints),
        angle_unit=AngleUnit.RAD,
        state_rate_hz=500.0,
        command_rate_hz=50.0,
        observation=[ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0)],
        action=ActionSpec(
            type=ActionType.JOINT_POSITION,
            kp=[40.0] * dof if gains else None,
            kd=[1.0] * dof if gains else None,
        ),
    )


def instance(profile: RobotProfile | None = None) -> Instance:
    return Instance(
        id="g1",
        profile=profile or gazebo_profile(),
        description=Path("g1_23dof.urdf"),
        role=Role.SCRIPTED,
    )


def description(
    joints: Sequence[str], *, mounted: bool = False, imu: bool = True
) -> UrdfModel:
    """A chain of revolute joints with these names: all the bridge's block reads.

    ``mounted`` bolts the pelvis to the world, as the mounted G1's description does (ADR-0063).
    ``imu`` hangs the vendor's IMU link on the pelvis, as both vendor descriptions do.
    """
    parts = ['<robot name="g1">', '<link name="pelvis"/>']
    if imu:
        parts.append(
            f'<link name="{IMU_LINK}"/><joint name="{IMU_LINK}_joint" type="fixed">'
            f'<parent link="pelvis"/><child link="{IMU_LINK}"/></joint>'
        )
    if mounted:
        parts.append(
            '<link name="world"/><joint name="world_joint" type="fixed">'
            '<parent link="world"/><child link="pelvis"/></joint>'
        )
    parent = "pelvis"
    for name in joints:
        child = name.removesuffix("_joint") + "_link"
        parts.append(f'<link name="{child}"/>')
        parts.append(
            f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="0 0 1"/>'
            '<limit lower="-3" upper="3" effort="50" velocity="20"/></joint>'
        )
        parent = child
    parts.append("</robot>")
    return parse_urdf_text("".join(parts), Path("g1.urdf"))


def manager_document(twin: G1GazeboTwin, item: Instance) -> dict[str, Any]:
    (generated,) = twin.files(item, ROOT)
    assert str(generated.path) == "g1/controllers.yaml"
    document: dict[str, Any] = yaml.safe_load(generated.content)
    return document


# -- registration ---------------------------------------------------------------


def test_the_package_registers_its_gazebo_twin_for_the_same_adapter() -> None:
    twins = {item.name: item.value for item in entry_points(group="heron.twins")}
    assert twins["unitree_g1_gazebo"] == "unitree_g1.bridge:G1GazeboTwin"

    twin = G1GazeboTwin()
    assert isinstance(twin, GazeboTwin)
    assert (twin.adapter, twin.backend, twin.kind) == (PLUGIN, GAZEBO, TwinKind.BRIDGE)


# -- what the controller is told -----------------------------------------------


def test_the_bridge_is_told_the_joints_their_slots_and_the_body() -> None:
    document = manager_document(G1GazeboTwin(), instance())

    manager = document["/g1/controller_manager"]["ros__parameters"]
    assert manager["update_rate"] == 500
    assert manager[CONTROLLER] == {"type": CONTROLLER_TYPE}
    assert manager["joint_state_broadcaster"] == {
        "type": "joint_state_broadcaster/JointStateBroadcaster"
    }
    assert document["/g1/joint_state_broadcaster"]["ros__parameters"] == {
        "use_local_topics": False
    }

    bridge = document[f"/g1/{CONTROLLER}"]["ros__parameters"]
    assert list(bridge) == list(PARAMETERS)
    assert bridge["joints"] == list(JOINTS_23DOF)
    assert bridge["slots"] == list(slots_of(JOINTS_23DOF))
    assert 13 not in bridge["slots"] and 14 not in bridge["slots"]
    assert bridge["mode_machine"] == 4
    assert (bridge["command_topic"], bridge["state_topic"]) == ("/lowcmd", "/lowstate")
    assert bridge["imu_sensor"] == "g1_imu"


# -- the body, measured (ADR-0076) -------------------------------------------------


def test_the_pelvis_imu_is_a_sensor_of_the_world_loaned_to_the_bridge() -> None:
    """The same name in three places: the world's sensor, the loan, the bridge's parameter."""
    block, _, sensor = G1GazeboTwin().description_elements(
        instance(), description(JOINTS_23DOF), ROOT
    )
    (loaned,) = block.findall("sensor")
    assert loaned.get("name") == "g1_imu"
    assert [s.get("name") for s in loaned.findall("state_interface")] == list(IMU_INTERFACES)
    assert loaned.find("command_interface") is None

    assert sensor.tag == "gazebo" and sensor.get("reference") == IMU_LINK
    world = sensor.find("sensor")
    assert world is not None
    assert (world.get("name"), world.get("type")) == ("g1_imu", "imu")
    assert world.findtext("update_rate") == "500", "на частоте состояния контракта"
    assert world.findtext("imu/orientation_reference_frame/localization") == "ENU"


def test_a_description_without_the_imu_link_gets_no_bridge() -> None:
    with pytest.raises(GeneratorError, match=IMU_LINK):
        G1GazeboTwin().description_elements(
            instance(), description(JOINTS_23DOF, imu=False), ROOT
        )


def test_slots_follow_joint_names_not_their_order() -> None:
    """A contract listing joints in another order sends each to its own motor (ADR-0027)."""
    shuffled = list(reversed(JOINTS_23DOF))
    profile = gazebo_profile(contract=contract(shuffled))
    bridge = manager_document(G1GazeboTwin(), instance(profile))[f"/g1/{CONTROLLER}"][
        "ros__parameters"
    ]
    assert bridge["joints"] == shuffled
    assert bridge["slots"] == [LAYOUT.index(name) for name in shuffled]


def test_every_joint_is_commanded_with_effort_and_read_back_in_position_and_velocity() -> None:
    elements = G1GazeboTwin().description_elements(
        instance(), description(JOINTS_23DOF), ROOT
    )
    block, plugin, _ = elements
    assert block.tag == "ros2_control"
    assert block.findtext("hardware/plugin") == ros2_control.HARDWARE_PLUGIN

    joints = block.findall("joint")
    assert [item.get("name") for item in joints] == list(JOINTS_23DOF)
    for item in joints:
        assert [c.get("name") for c in item.findall("command_interface")] == ["effort"]
        assert [s.get("name") for s in item.findall("state_interface")] == ["position", "velocity"]

    parameters = plugin.findtext("plugin/parameters") or ""
    assert parameters.endswith("g1/controllers.yaml")
    assert plugin.findtext("plugin/ros/namespace") == "g1"


def test_the_launch_starts_the_broadcaster_then_the_bridge() -> None:
    nodes = G1GazeboTwin().launch_nodes(instance())
    assert [node.name for node in nodes] == ["spawn_joint_state_broadcaster", f"spawn_{CONTROLLER}"]
    assert [node.arguments[0] for node in nodes] == ["joint_state_broadcaster", CONTROLLER]


def test_the_link_is_at_the_root_as_on_the_robot() -> None:
    """ADR-0034, p. 5: whatever the robot's id, its topics carry no namespace."""
    plane = G1GazeboTwin().control_plane("g1_left", gazebo_profile())
    assert plane["command_topic"] == "/lowcmd"
    assert plane["state_topic"] == "/lowstate"
    assert plane["command_message"] == "unitree_hg/msg/LowCmd"
    assert plane["state_message"] == "unitree_hg/msg/LowState"
    assert plane["controller"] == CONTROLLER


# -- the environment of the world ------------------------------------------------


def test_the_world_runs_on_the_link_the_adapter_uses() -> None:
    """One configuration for both sides of a local link (ADR-0034, p. 3)."""
    environment = G1GazeboTwin().environment(instance())
    assert environment == {
        "CYCLONEDDS_URI": LinkConfig.from_params(LINK_PARAMS).cyclonedds_xml(),
        "RMW_IMPLEMENTATION": RMW,
        "ROS_DOMAIN_ID": "1",
    }
    assert "MaxAutoParticipantIndex" in environment["CYCLONEDDS_URI"]


def test_a_link_for_the_robot_reaches_the_world_unchanged() -> None:
    robot = {"domain": 0, "interface": "enp3s0"}
    profile = gazebo_profile(adapters={"sim.gazebo": AdapterSpec(plugin=PLUGIN, params=robot)})
    environment = G1GazeboTwin().environment(instance(profile))
    assert environment["ROS_DOMAIN_ID"] == "0"
    assert "enp3s0" in environment["CYCLONEDDS_URI"]
    assert "Peers" not in environment["CYCLONEDDS_URI"]


# -- refusals ----------------------------------------------------------------------


def test_a_contract_without_gains_gets_no_bridge() -> None:
    profile = gazebo_profile(contract=contract(JOINTS_23DOF, gains=False))
    with pytest.raises(GeneratorError, match="kp и kd"):
        G1GazeboTwin().files(instance(profile), ROOT)


def test_a_contract_of_no_known_body_gets_no_bridge() -> None:
    legs = list(LAYOUT[:12])
    profile = gazebo_profile(contract=contract(legs), limits={})
    with pytest.raises(GeneratorError, match="ни одно тело G1"):
        G1GazeboTwin().files(instance(profile), ROOT)


def test_a_joint_outside_the_layout_gets_no_bridge() -> None:
    joints = [*JOINTS_23DOF[:-1], "tail_joint"]
    profile = gazebo_profile(contract=contract(joints))
    with pytest.raises(GeneratorError, match="tail_joint"):
        G1GazeboTwin().description_elements(instance(profile), description(joints), ROOT)


def test_a_world_without_the_adapters_link_is_refused_by_name() -> None:
    with pytest.raises(GeneratorError, match=r"sim\.gazebo"):
        G1GazeboTwin().environment(instance(profile_23dof()))
    broken = gazebo_profile(
        adapters={"sim.gazebo": AdapterSpec(plugin=PLUGIN, params={"interface": "lo"})}
    )
    with pytest.raises(GeneratorError, match=r"params\.domain"):
        G1GazeboTwin().environment(instance(broken))


# -- requirements, derived from what is written ------------------------------------


def test_the_twin_declares_exactly_the_packages_it_names() -> None:
    """ADR-0025: every ROS 2 package in the manager's file, the description, the launch,
    the environment and the messages of the link -- and nothing else."""
    twin, item = G1GazeboTwin(), instance()
    named = {node.package for node in twin.launch_nodes(item)}

    manager = manager_document(twin, item)["/g1/controller_manager"]["ros__parameters"]
    named |= {entry["type"].split("/")[0] for entry in manager.values() if isinstance(entry, dict)}

    written = "".join(
        ET.tostring(element, encoding="unicode")
        for element in twin.description_elements(item, description(JOINTS_23DOF), ROOT)
    )
    for reference in re.findall(r'(?:filename="|<plugin>)([a-z_0-9]+)[/-]', written):
        named.add(reference)

    named.add(twin.environment(item)["RMW_IMPLEMENTATION"])
    plane = twin.control_plane("g1", item.profile)
    named |= {str(plane[key]).split("/")[0] for key in ("command_message", "state_message")}

    assert named == needed(twin.requirements, RequirementKind.ROS2)


# -- the C++ side speaks the same names ------------------------------------------


def test_the_controller_declares_exactly_the_parameters_the_twin_writes() -> None:
    source = (ROS2_DIR / "src" / "low_level_bridge.cpp").read_text(encoding="utf-8")
    declared = re.findall(r'constexpr char k\w+Param\[\] = "(\w+)";', source)
    assert declared == list(PARAMETERS)
    assert source.count("auto_declare<") == len(PARAMETERS)


def test_the_controller_reads_the_imu_interfaces_the_twin_loans_in_that_order() -> None:
    source = (ROS2_DIR / "src" / "low_level_bridge.cpp").read_text(encoding="utf-8")
    table = re.search(r"kImuInterfaces = \{(.*?)\};", source, re.DOTALL)
    assert table is not None
    assert re.findall(r'"([\w.]+)"', table.group(1)) == list(IMU_INTERFACES)


def test_the_controller_class_is_the_type_the_manager_is_told() -> None:
    source = (ROS2_DIR / "src" / "low_level_bridge.cpp").read_text(encoding="utf-8")
    package, name = CONTROLLER_TYPE.split("/")
    assert package == BRIDGE_PACKAGE
    assert f"namespace {package}" in source and f"class {name} " in source
    assert f"PLUGINLIB_EXPORT_CLASS({package}::{name}, " in source

    library = ET.parse(ROS2_DIR / f"{BRIDGE_PACKAGE}.xml").getroot()
    (declared,) = library.findall("class")
    assert library.get("path") == BRIDGE_PACKAGE
    assert declared.get("name") == CONTROLLER_TYPE
    assert declared.get("type") == f"{package}::{name}"
    assert declared.get("base_class_type") == "controller_interface::ControllerInterface"


def test_the_ros2_package_is_the_one_the_twin_requires() -> None:
    manifest = ET.parse(ROS2_DIR / "package.xml").getroot()
    assert manifest.findtext("name") == BRIDGE_PACKAGE
    assert MESSAGES_PACKAGE in {item.text for item in manifest.findall("depend")}

    cmake = (ROS2_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f"project({BRIDGE_PACKAGE} " in cmake
    assert f"{BRIDGE_PACKAGE}.xml" in cmake
    assert MESSAGES_PACKAGE in cmake


def test_importing_the_bridge_twin_imports_no_dds_and_no_ros() -> None:
    """Every twin of the group is loaded on every lookup (ADR-0024): it must stay light."""
    code = (
        "import sys, unitree_g1.bridge; "
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'cyclonedds', 'unitree_sdk2py', 'rclpy'}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# -- the upper body of a mounted robot (ADR-0064) ---------------------------------


def test_an_upper_body_on_a_mounted_description_is_bridged_as_its_body() -> None:
    item = instance(profile_upper_body())
    whole = [*LAYOUT[:12], *JOINTS_UPPER_BODY_6]
    G1GazeboTwin().description_elements(item, description(whole, mounted=True), ROOT)

    bridge = manager_document(G1GazeboTwin(), item)[f"/g1/{CONTROLLER}"]["ros__parameters"]
    assert bridge["joints"] == list(JOINTS_UPPER_BODY_6)
    assert bridge["slots"] == [12, *range(15, 29)]
    assert bridge["mode_machine"] == 6


def test_the_legs_of_an_upper_body_are_read_and_not_commanded() -> None:
    """ADR-0012 as it stands: a leg of the description outside the contract is observed only."""
    whole = [*LAYOUT[:12], *JOINTS_UPPER_BODY_6]
    block, _, _ = G1GazeboTwin().description_elements(
        instance(profile_upper_body()), description(whole, mounted=True), ROOT
    )
    for joint in block.iter("joint"):
        commands = [item.get("name") for item in joint.iter("command_interface")]
        assert commands == ([] if joint.get("name") in LAYOUT[:12] else ["effort"]), joint.get(
            "name"
        )


def test_an_upper_body_on_a_free_base_gets_no_bridge() -> None:
    whole = [*LAYOUT[:12], *JOINTS_UPPER_BODY_6]
    with pytest.raises(GeneratorError, match="не крепит"):
        G1GazeboTwin().description_elements(
            instance(profile_upper_body()), description(whole), ROOT
        )


def test_an_upper_body_in_a_profile_that_may_meet_the_robot_gets_no_bridge() -> None:
    base = profile_upper_body()
    adapters = {**base.adapters, "real": AdapterSpec(plugin=PLUGIN, params={"domain": 0})}
    with pytest.raises(GeneratorError, match="'real'"):
        G1GazeboTwin().files(instance(base.model_copy(update={"adapters": adapters})), ROOT)
