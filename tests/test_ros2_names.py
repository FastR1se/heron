"""The names of the control plane, and the two sides that must agree on them.

Four of phase 2's failures had one shape: an artifact correct on its own, read
differently by whoever consumed it. Three were closed by a test that knows about
the outside consumer, and these are the same kind of test -- they hold the
generator, the adapter, the manifest and the phase 0 script against one another
so that none of them can be changed alone.
"""

from __future__ import annotations

import re

import pytest

from conftest import REPO_ROOT, make_contract, make_profile
from heron.core.contract import ActionSpec, ObservationTerm
from heron.core.enums import ActionType, ControlMode, ObservationType
from heron.generators import ros2_control as generator
from heron.ros2 import names
from heron.ros2.transport import MESSAGE_FIELDS, message_class_path

# -- what the names are ----------------------------------------------------


def test_streaming_position_robot_is_driven_by_a_group_controller() -> None:
    profile = make_profile()
    assert names.controller_name(profile) == "joint_position_controller"
    assert names.controller_type(profile) == (
        "position_controllers/JointGroupPositionController"
    )


def test_command_mode_robot_is_driven_by_the_trajectory_controller() -> None:
    profile = make_profile(control_mode=ControlMode.COMMAND)
    assert names.controller_name(profile) == names.TRAJECTORY_CONTROLLER_NAME
    assert names.controller_type(profile) == names.TRAJECTORY_CONTROLLER


def test_topics_sit_under_the_scenario_identifier() -> None:
    """Never under the profile name: two robots of one kind share a profile."""
    profile = make_profile(name="arm6")
    assert names.state_topic("arm1") == "/arm1/joint_states"
    assert names.command_topic("arm1", profile) == (
        "/arm1/joint_position_controller/commands"
    )
    assert names.command_topic("arm2", profile) == (
        "/arm2/joint_position_controller/commands"
    )


def test_trajectory_controller_listens_on_its_own_topic() -> None:
    profile = make_profile(control_mode=ControlMode.COMMAND)
    assert names.command_topic("arm1", profile) == (
        "/arm1/joint_trajectory_controller/joint_trajectory"
    )
    assert names.command_message(profile) == names.TRAJECTORY_MESSAGE


def test_no_namespace_means_the_root_and_not_a_guess() -> None:
    """A robot brought up without a namespace publishes at the root."""
    assert names.state_topic("") == "/joint_states"
    assert names.qualified("", "a", "b") == "/a/b"
    assert names.qualified("/arm1/", "/c/") == "/arm1/c"


def test_action_without_a_joint_interface_has_no_topic() -> None:
    profile = make_profile(
        contract=make_contract(3, action=ActionSpec(type=ActionType.CARTESIAN_POSE))
    )
    assert names.command_interface(profile) is None
    assert names.command_topic("arm1", profile) is None
    assert names.command_message(profile) is None


def test_state_interfaces_always_include_position() -> None:
    """A contract that observes only velocity still describes a robot whose
    joints have positions, and nothing draws a robot without them."""
    profile = make_profile(
        contract=make_contract(
            3, observation=[ObservationTerm(type=ObservationType.JOINT_VEL, scale=1.0)]
        )
    )
    assert names.state_interfaces(profile) == ["position", "velocity"]


# -- the two sides -------------------------------------------------------


def test_generator_reads_its_names_from_the_shared_module() -> None:
    """One definition, not two that happen to match today (ADR-0014)."""
    for profile in (make_profile(), make_profile(control_mode=ControlMode.COMMAND)):
        assert generator.controller_name(profile) == names.controller_name(profile)
        assert generator.controller_type(profile) == names.controller_type(profile)
        assert generator.state_interfaces(profile) == names.state_interfaces(profile)


def test_every_message_the_plane_can_ask_for_can_be_built() -> None:
    """A controller the generator can configure must be drivable."""
    assert set(MESSAGE_FIELDS) >= names.MESSAGES


def test_every_buildable_message_belongs_to_the_plane() -> None:
    """The other side: no conversion code for a message nobody sends."""
    assert set(MESSAGE_FIELDS) <= names.MESSAGES


@pytest.mark.parametrize("message_type", sorted(names.MESSAGES))
def test_message_types_name_an_importable_class(message_type: str) -> None:
    module, class_name = message_class_path(message_type)
    assert module.endswith(".msg")
    assert class_name[:1].isupper()


def test_every_package_the_transport_imports_is_installed_by_phase_zero() -> None:
    """The rule of ADR-0013, applied to the adapter instead of the launch."""
    script = REPO_ROOT / "scripts" / "phase0.sh"
    if not script.is_file():  # pragma: no cover - the script travels with the repo
        pytest.skip("scripts/phase0.sh отсутствует")
    text = script.read_text(encoding="utf-8")
    missing = sorted({apt for apt in names.ADAPTER_PACKAGES.values() if apt not in text})
    assert not missing, f"фаза 0 не ставит: {missing}"


def test_transport_imports_nothing_it_did_not_declare() -> None:
    """The other side: a ROS 2 import that was never declared is an undeclared
    environment, which is exactly how phase 2 lost a scene."""
    source = (REPO_ROOT / "src" / "heron" / "ros2" / "transport.py").read_text(
        encoding="utf-8"
    )
    quoted = set(re.findall(r'_import_ros2\("([a-z_0-9.]+)"\)', source))
    quoted |= {
        item.split("/")[0] for item in re.findall(r'_message_class\("([^"]+)"\)', source)
    }
    roots = {item.split(".")[0] for item in quoted}
    undeclared = sorted(roots - set(names.ADAPTER_PACKAGES))
    assert not undeclared, f"импортируется, но не объявлено: {undeclared}"


def test_declared_packages_are_actually_used() -> None:
    """A declaration nobody uses rots: it is the second list again."""
    source = (REPO_ROOT / "src" / "heron" / "ros2").glob("*.py")
    text = "".join(path.read_text(encoding="utf-8") for path in source)
    unused = sorted(
        package
        for package in names.ADAPTER_PACKAGES
        if package not in text.replace("ADAPTER_PACKAGES", "")
    )
    assert not unused, f"объявлено, но нигде не используется: {unused}"


def test_phase_zero_checks_exactly_the_imports_the_transport_declares() -> None:
    """The rule of ADR-0013 applied to itself.

    The phase 0 script verifies that the execution environment can import what
    the adapter needs -- which means it carries a second copy of the list, and a
    second copy is precisely what let a launch file name a package nobody
    installed. Held together here, from both sides.
    """
    script = REPO_ROOT / "scripts" / "phase0.sh"
    if not script.is_file():  # pragma: no cover - the script travels with the repo
        pytest.skip("scripts/phase0.sh отсутствует")

    text = script.read_text(encoding="utf-8")
    declared = re.search(r'ADAPTER_IMPORTS="([^"]*)"', text)
    assert declared, "scripts/phase0.sh не объявляет ADAPTER_IMPORTS"

    assert set(declared.group(1).split()) == set(names.ADAPTER_PACKAGES), (
        "список импортов в скрипте фазы 0 разошёлся с ADAPTER_PACKAGES"
    )
