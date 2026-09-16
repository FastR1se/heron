"""Generation: what is derived, what is refused, and what stays put.

The tests are grouped the way ADR-0009 and ADR-0010 split the subject. First:
every derivation is a sentence of the contract turned into a sentence of the
control stack, so each one is checked against the contract it came from. Second:
the input is never written, and the output is the same twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import make_contract, make_profile, urdf_text, write_profile, write_urdf
from heron.core.contract import ActionSpec, ObservationTerm
from heron.core.enums import ActionType, ControlMode, ObservationType, Role
from heron.core.models import AdapterSpec, JointRange, Limits, Scenario
from heron.generators import GeneratorError, build, scenario_for_profile
from heron.generators.common import Instance, resolve_description
from heron.generators.ros2_control import (
    command_interface,
    controller_type,
    controllers_config,
    merged_description,
    state_interfaces,
)
from heron.generators.world import physics_step
from heron.registry.registry import Registry
from heron.ros2 import names

GAZEBO = "gazebo"


def _profile(tmp_path: Path, **overrides: Any) -> Any:
    overrides.setdefault("description", {"urdf": "urdf/arm.urdf"})
    overrides.setdefault(
        "adapters",
        {
            "sim.gazebo": AdapterSpec(plugin="ros2_control"),
            "sim.dummy": AdapterSpec(plugin="dummy"),
        },
    )
    profile = make_profile(name="arm", **overrides)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    return profile


def _instance(tmp_path: Path, profile: Any, role: Role = Role.SCRIPTED) -> Instance:
    return Instance(
        id="arm1",
        profile=profile,
        description=resolve_description(profile, tmp_path / "arm.yaml"),
        role=role,
    )


# -- derivations -----------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "interface"),
    [
        (ActionType.JOINT_POSITION, "position"),
        (ActionType.JOINT_VELOCITY, "velocity"),
        (ActionType.JOINT_TORQUE, "effort"),
    ],
)
def test_the_command_interface_comes_from_the_action(
    action: ActionType, interface: str
) -> None:
    profile = make_profile(contract=make_contract(action=ActionSpec(type=action)))
    assert command_interface(profile) == interface


@pytest.mark.parametrize(
    "action", [ActionType.CARTESIAN_POSE, ActionType.GRIPPER_WIDTH, ActionType.BASE_VELOCITY]
)
def test_an_action_without_a_joint_interface_is_refused(action: ActionType) -> None:
    """Writing a Cartesian pose as joint positions would produce a robot that
    moves and is wrong -- the failure the platform exists to prevent."""
    profile = make_profile(contract=make_contract(action=ActionSpec(type=action)))
    with pytest.raises(GeneratorError) as caught:
        command_interface(profile)
    assert str(action) in str(caught.value)


def test_state_interfaces_come_from_the_observation() -> None:
    profile = make_profile(
        contract=make_contract(
            observation=[
                ObservationTerm(type=ObservationType.JOINT_POS),
                ObservationTerm(type=ObservationType.JOINT_TORQUE),
                ObservationTerm(type=ObservationType.PREVIOUS_ACTION),
            ]
        )
    )
    assert state_interfaces(profile) == ["position", "effort"]


def test_position_is_always_a_state_interface() -> None:
    """A contract observing only velocity still describes joints that have
    positions, and nothing could be shown without them."""
    profile = make_profile(
        contract=make_contract(observation=[ObservationTerm(type=ObservationType.JOINT_VEL)])
    )
    assert state_interfaces(profile) == ["position", "velocity"]


def test_the_control_mode_chooses_the_class_of_controller() -> None:
    streaming = make_profile(control_mode=ControlMode.STREAMING)
    command = make_profile(control_mode=ControlMode.COMMAND)

    assert controller_type(streaming) == "position_controllers/JointGroupPositionController"
    assert controller_type(command) == "joint_trajectory_controller/JointTrajectoryController"


def test_the_manager_rate_comes_from_the_contract(tmp_path: Path) -> None:
    profile = _profile(tmp_path, contract=make_contract(dof=3, state_rate_hz=250.0))
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    assert document["/arm1/controller_manager"]["ros__parameters"]["update_rate"] == 250


def test_joints_keep_the_order_of_the_contract(tmp_path: Path) -> None:
    order = ["shoulder", "elbow", "wrist"]
    profile = _profile(tmp_path, contract=make_contract(dof=3, joint_order=order))
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    assert document["/arm1/joint_position_controller"]["ros__parameters"]["joints"] == order


def test_the_physics_step_follows_the_fastest_state_rate(tmp_path: Path) -> None:
    slow = _profile(tmp_path, contract=make_contract(dof=3, state_rate_hz=100.0))
    fast = make_profile(name="fast", contract=make_contract(dof=3, state_rate_hz=4000.0))

    assert physics_step([_instance(tmp_path, slow)]) == pytest.approx(0.001)
    assert physics_step(
        [_instance(tmp_path, slow), _instance(tmp_path, fast)]
    ) == pytest.approx(0.00025)


# -- the description is an input -------------------------------------------


def test_the_input_description_is_never_written(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    original = (tmp_path / "urdf" / "arm.urdf").read_text(encoding="utf-8")

    merged_description(_instance(tmp_path, profile), tmp_path / "out")

    assert (tmp_path / "urdf" / "arm.urdf").read_text(encoding="utf-8") == original


def test_the_merged_description_adds_interfaces_and_no_kinematics(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    for name in profile.contract.joint_order:
        assert f'<joint name="{name}">' in generated
    assert "<ros2_control" in generated
    assert "gz_ros2_control" in generated
    # Every link of the input survives, and no link appears that was not there.
    assert generated.count("<link ") == urdf_text("arm", list(profile.contract.joint_order)).count(
        "<link "
    )


def test_command_bounds_come_from_the_profile_limits(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        contract=make_contract(dof=3),
        limits=Limits(joint_position=JointRange(min=[-1.0] * 3, max=[1.5] * 3)),
    )
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    assert '<param name="min">-1</param>' in generated
    assert '<param name="max">1.5</param>' in generated


def test_an_obstacle_gets_no_controllers(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    instance = _instance(tmp_path, profile, role=Role.STATIC)
    generated = merged_description(instance, tmp_path / "out").content

    assert "<static>true</static>" in generated
    assert "<ros2_control" not in generated


# -- the bundle ------------------------------------------------------------


def _scenario(name: str = "scene", robots: tuple[dict[str, Any], ...] = ()) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": name,
            "world": {"backend": GAZEBO},
            "robots": list(robots) or [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        }
    )


def test_a_bundle_holds_everything_the_scenario_needs(tmp_path: Path) -> None:
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")

    assert [str(p) for p in bundle.paths()] == [
        "arm1/controllers.yaml",
        "arm1/description.urdf",
        "bringup.launch.xml",
        "view.rviz",
        "world.sdf",
        "manifest.json",
    ]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """The whole of ``--check`` rests on this (ADR-0010)."""
    _profile(tmp_path)
    first = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    second = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")

    assert [f.content for f in first.files] == [f.content for f in second.files]


def test_differences_are_found_when_the_profile_moves_on(tmp_path: Path) -> None:
    _profile(tmp_path)
    registry = Registry([tmp_path])
    build(_scenario(), registry, into=tmp_path / "out").write()

    assert build(_scenario(), registry, into=tmp_path / "out").differences() == []

    moved = _profile(tmp_path, contract=make_contract(dof=3, state_rate_hz=200.0))
    assert moved.contract.state_rate_hz == 200.0

    # The rate is a normative field, so the contract identity moves with it, and
    # every artifact that recorded that identity goes stale -- which is the point
    # of recording it (ADR-0010).
    stale = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out").differences()
    assert [str(item.path) for item in stale] == [
        "arm1/controllers.yaml",
        "arm1/description.urdf",
        "world.sdf",
        "manifest.json",
    ]


def test_twins_of_one_profile_get_separate_namespaces(tmp_path: Path) -> None:
    _profile(tmp_path)
    scenario = _scenario(
        robots=(
            {"id": "left", "profile": "arm", "role": "scripted"},
            {"id": "right", "profile": "arm", "pose": [1.0, 0.0, 0.0], "role": "scripted"},
        )
    )
    bundle = build(scenario, Registry([tmp_path]), into=tmp_path / "out")
    launch = next(f for f in bundle.files if str(f.path).endswith(".launch.xml")).content

    assert '<push-ros-namespace namespace="left" />' in launch
    assert '<push-ros-namespace namespace="right" />' in launch
    assert 'value="left/"' in launch and 'value="right/"' in launch


def test_a_stated_world_is_an_input(tmp_path: Path) -> None:
    """ADR-0009 treats a named world exactly like a named description."""
    _profile(tmp_path)
    (tmp_path / "worlds").mkdir()
    (tmp_path / "worlds" / "shop.sdf").write_text("<sdf/>", encoding="utf-8")
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": GAZEBO, "file": "worlds/shop.sdf"},
            "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        }
    )
    bundle = build(
        scenario, Registry([tmp_path]), into=tmp_path / "out", scenario_path=tmp_path / "s.yaml"
    )

    assert "world.sdf" not in [str(p) for p in bundle.paths()]
    launch = next(f for f in bundle.files if str(f.path).endswith(".launch.xml")).content
    assert "worlds/shop.sdf" in launch
    assert (tmp_path / "worlds" / "shop.sdf").read_text(encoding="utf-8") == "<sdf/>"


def test_the_manifest_records_the_contract_it_was_generated_from(tmp_path: Path) -> None:
    import json

    profile = _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    manifest = json.loads(next(f for f in bundle.files if str(f.path) == "manifest.json").content)

    assert manifest["robots"][0]["contract_id"] == profile.contract.contract_id
    assert manifest["robots"][0]["description"] == "urdf/arm.urdf"


def test_writing_then_reading_back_finds_no_difference(tmp_path: Path) -> None:
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    written = bundle.write()

    assert all(path.is_file() for path in written)
    assert bundle.differences() == []


# -- a lone profile --------------------------------------------------------


def test_a_lone_profile_generates_under_its_only_backend(tmp_path: Path) -> None:
    profile = make_profile(adapters={"sim.gazebo": AdapterSpec(plugin="ros2_control")})
    scenario = scenario_for_profile(profile)

    assert scenario.world.backend == GAZEBO
    assert scenario.robots[0].id == profile.name


def test_a_profile_with_two_simulators_has_to_say_which(tmp_path: Path) -> None:
    """One scenario, one owner of the physics (ADR-0002)."""
    profile = make_profile(
        adapters={
            "sim.gazebo": AdapterSpec(plugin="ros2_control"),
            "sim.dummy": AdapterSpec(plugin="dummy"),
        }
    )
    with pytest.raises(GeneratorError):
        scenario_for_profile(profile)

    assert scenario_for_profile(profile, GAZEBO).world.backend == GAZEBO


def test_a_backend_the_profile_has_no_adapter_for_is_refused() -> None:
    profile = make_profile(adapters={"sim.dummy": AdapterSpec(plugin="dummy")})
    with pytest.raises(GeneratorError):
        scenario_for_profile(profile, GAZEBO)


def test_a_backend_without_generators_is_refused(tmp_path: Path) -> None:
    """Everything generated here belongs to one simulator. Handing out the same
    bundle under another backend's name would describe a world nobody asked
    for."""
    _profile(tmp_path)
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": "dummy"},
            "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        }
    )
    with pytest.raises(GeneratorError) as caught:
        build(scenario, Registry([tmp_path]), into=tmp_path / "out")
    assert "dummy" in str(caught.value)


def test_a_fresh_stub_generates_its_environments(tmp_path: Path) -> None:
    """The golden path of section 9: new robot -> validate -> environments,
    without editing the profile first."""
    from heron.cli.new_robot import write_stub
    from heron.core.loading import load_model
    from heron.core.models import RobotProfile
    from heron.generators import build_profile

    profile_path, _, _ = write_stub("some_robot", tmp_path, dof=4)
    profile = load_model(profile_path, RobotProfile)

    bundle = build_profile(profile, profile_path, into=tmp_path / "out", backend=GAZEBO)
    bundle.write()

    assert (tmp_path / "out" / "some_robot" / "controllers.yaml").is_file()
    assert (tmp_path / "out" / "bringup.launch.xml").is_file()


def test_free_text_parameters_declare_their_type(tmp_path: Path) -> None:
    """Regression, found by a real run: the launch frontend guesses parameter
    types with a YAML loader, and a robot description is not YAML. The generated
    header is enough to break it -- "Источник: файл.urdf" reads as a mapping --
    and the launch then fails to parse before starting anything."""
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    launch = next(f for f in bundle.files if str(f.path).endswith(".launch.xml")).content

    assert 'name="robot_description" value="$(command' in launch
    text_params = ("robot_description", "frame_prefix")
    for line in launch.splitlines():
        stripped = line.strip()
        if stripped.startswith("<param") and any(name in stripped for name in text_params):
            assert 'type="str"' in stripped, stripped


def test_the_generated_description_is_not_valid_yaml(tmp_path: Path) -> None:
    """The other half of the regression: proof that the type declaration is
    load-bearing rather than decorative."""
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    description = next(
        f for f in bundle.files if str(f.path).endswith("description.urdf")
    ).content

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(description)


def test_parameter_keys_are_fully_qualified(tmp_path: Path) -> None:
    """Regression, found by a real run: a parameter file is matched against a
    node's fully qualified name. Keyed on a bare ``controller_manager`` it never
    reaches ``/arm1/controller_manager``, nothing warns, and the first symptom is
    a controller reported as having no type."""
    profile = _profile(tmp_path)
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    assert set(document) == {
        "/arm1/controller_manager",
        "/arm1/joint_state_broadcaster",
        "/arm1/joint_position_controller",
    }
    manager = document["/arm1/controller_manager"]["ros__parameters"]
    assert manager["joint_state_broadcaster"]["type"].startswith("joint_state_broadcaster/")


def test_the_broadcaster_is_told_where_to_publish(tmp_path: Path) -> None:
    """The adapter subscribes to the result, so it is stated and not inherited.

    With ``use_local_topics`` true the states move to the broadcaster's own node
    and the adapter finds an empty topic -- and nothing reports that, because
    nothing is wrong on either side taken alone (ADR-0014)."""
    profile = _profile(tmp_path)
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    broadcaster = document["/arm1/joint_state_broadcaster"]["ros__parameters"]
    assert broadcaster[names.LOCAL_TOPICS_PARAM] is False
    assert names.state_topic("arm1") == "/arm1/joint_states"


def test_the_manifest_names_the_topics_the_adapter_will_use(tmp_path: Path) -> None:
    """What a run will look for, on disk before the run (ADR-0014)."""
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    manifest = json.loads(
        next(f for f in bundle.files if str(f.path) == "manifest.json").content
    )

    plane = {item["id"]: item.get("control") for item in manifest["robots"]}
    assert plane["arm1"]["state_topic"] == "/arm1/joint_states"
    assert plane["arm1"]["command_topic"] == "/arm1/joint_position_controller/commands"
    assert plane["arm1"]["controller"] == "joint_position_controller"


def test_an_obstacle_has_no_control_plane_in_the_manifest(tmp_path: Path) -> None:
    """Nothing drives it, so there is no topic to name (ADR-0012)."""
    _profile(tmp_path)
    scenario = _scenario(
        robots=(
            {"id": "arm1", "profile": "arm", "role": "scripted"},
            {"id": "box", "profile": "arm", "role": "static"},
        )
    )
    bundle = build(scenario, Registry([tmp_path]), into=tmp_path / "out")
    manifest = json.loads(
        next(f for f in bundle.files if str(f.path) == "manifest.json").content
    )
    plane = {item["id"]: item for item in manifest["robots"]}
    assert "control" in plane["arm1"]
    assert "control" not in plane["box"]


def test_the_namespace_is_the_same_everywhere(tmp_path: Path) -> None:
    """Four generated files have to agree on it: the plugin that starts the
    controller manager, the launch file that pushes the namespace, the parameter
    file that must reach that node, and the frames the visualization looks for."""
    _profile(tmp_path)
    bundle = build(_scenario(), Registry([tmp_path]), into=tmp_path / "out")
    files = {str(f.path): f.content for f in bundle.files}

    assert "<namespace>arm1</namespace>" in files["arm1/description.urdf"]
    assert "/arm1/controller_manager:" in files["arm1/controllers.yaml"]
    assert '<push-ros-namespace namespace="arm1" />' in files["bringup.launch.xml"]
    assert "arm1/base_link" in files["view.rviz"]


def test_movable_joints_outside_the_contract_are_observed(tmp_path: Path) -> None:
    """ADR-0012, found by a real run: the gripper rendered as missing links.
    The contract says what is commanded, the description says what moves."""
    profile = _profile(tmp_path, contract=make_contract(dof=2, joint_order=["a", "b"]))
    write_urdf(tmp_path, "arm", ["a", "b", "finger"])
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    assert '<joint name="finger">' in generated
    block = generated.split('<joint name="finger">')[1]
    assert "<state_interface" in block.split("</joint>")[0]
    assert "<command_interface" not in block.split("</joint>")[0], (
        "наблюдаемый сустав не должен получать командный интерфейс"
    )


def test_an_obstacle_gets_a_joint_state_publisher(tmp_path: Path) -> None:
    """It has no controller manager, so nothing else would report its joints."""
    _profile(tmp_path)
    scenario = _scenario(robots=({"id": "rock", "profile": "arm", "role": "static"},))
    bundle = build(scenario, Registry([tmp_path]), into=tmp_path / "out")
    launch = next(f for f in bundle.files if str(f.path).endswith(".launch.xml")).content

    assert 'pkg="joint_state_publisher"' in launch
    assert "spawner" not in launch, "у препятствия нет контроллеров"


# -- the environment the launch depends on ---------------------------------


def _launch_of(tmp_path: Path) -> str:
    _profile(tmp_path)
    scenario = _scenario(
        robots=(
            {"id": "arm1", "profile": "arm", "role": "scripted"},
            {"id": "rock", "profile": "arm", "role": "static"},
        )
    )
    bundle = build(scenario, Registry([tmp_path]), into=tmp_path / "out")
    return next(f for f in bundle.files if str(f.path).endswith(".launch.xml")).content


def test_the_launch_names_only_declared_packages(tmp_path: Path) -> None:
    """A launch that names an absent package does not degrade -- it fails to
    parse, and the whole scene dies with it (ADR-0013)."""
    import re

    from heron.core.environment import RequirementKind, needed
    from heron.generators.launch import LAUNCH_PACKAGES
    from heron.generators.ros2_control import Ros2ControlTwin

    launch = _launch_of(tmp_path)
    used = set(re.findall(r'pkg="([^"]+)"', launch))
    used |= set(re.findall(r"find-pkg-share ([a-z_0-9]+)", launch))
    declared = set(LAUNCH_PACKAGES) | needed(Ros2ControlTwin.requirements, RequirementKind.ROS2)

    assert used, "в запуске не нашлось ни одного узла"
    assert used <= declared, f"необъявленные пакеты: {used - declared}"


def test_the_builtin_twin_declares_exactly_the_packages_it_names(tmp_path: Path) -> None:
    """Derived from what the twin writes, not from what it says (ADR-0025).

    The spawner in the launch, the hardware component in the description, every
    controller class a configuration can name: a package added to any of them and
    not to the declaration fails here, and a declared one nobody uses fails too.
    """
    import re

    from heron.core.environment import RequirementKind, needed
    from heron.generators import ros2_control as twin
    from heron.generators.launch import LAUNCH_PACKAGES
    from heron.ros2 import names

    spawned = set(re.findall(r'pkg="([^"]+)"', _launch_of(tmp_path))) - set(LAUNCH_PACKAGES)
    classes = {
        *names.STREAMING_CONTROLLERS.values(),
        names.TRAJECTORY_CONTROLLER,
        names.BROADCASTER_TYPE,
        twin.HARDWARE_PLUGIN,
    }
    named = spawned | {item.split("/")[0] for item in classes}

    assert named == set(twin.TWIN_PACKAGES)
    assert needed(twin.Ros2ControlTwin.requirements, RequirementKind.ROS2) == named


@pytest.mark.parametrize("module", ["launch", "ros2_control"])
def test_every_declared_package_is_installed_by_phase_zero(module: str) -> None:
    """The other side of the same rule: the environment must provide them."""
    import importlib

    from conftest import REPO_ROOT

    declared = {
        "launch": "LAUNCH_PACKAGES",
        "ros2_control": "TWIN_PACKAGES",
    }[module]
    packages = getattr(importlib.import_module(f"heron.generators.{module}"), declared)

    script = REPO_ROOT / "scripts" / "phase0.sh"
    if not script.is_file():  # pragma: no cover - the script travels with the repo
        pytest.skip("scripts/phase0.sh отсутствует")
    text = script.read_text(encoding="utf-8")

    missing = sorted({apt for apt in packages.values() if apt not in text})
    assert not missing, f"фаза 0 не ставит: {missing}"


# -- meshes in the copy ----------------------------------------------------

MESH_LINKS = """  <link name="shell">
    <visual><geometry><mesh filename="meshes/shell.stl"/></geometry></visual>
  </link>
  <joint name="shell_joint" type="fixed">
    <parent link="base_link"/>
    <child link="shell"/>
  </joint>
  <link name="badge">
    <visual><geometry><mesh filename="package://vendor/badge.dae"/></geometry></visual>
  </link>
  <joint name="badge_joint" type="fixed">
    <parent link="base_link"/>
    <child link="badge"/>
  </joint>
"""


def _mesh_references(content: str) -> list[str]:
    from xml.etree import ElementTree as ET

    body = content.split("?>", 1)[1]
    return [mesh.get("filename", "") for mesh in ET.fromstring(body).iter("mesh")]


def test_relative_meshes_are_anchored_to_the_input(tmp_path: Path) -> None:
    """Regression, found by the first vendor description (ADR-0017): the copy is
    read as a string with no directory, so ``meshes/shell.stl`` resolves against
    nothing and the robot arrives without geometry."""
    profile = _profile(tmp_path)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order), extra=MESH_LINKS)
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    references = _mesh_references(generated)
    expected = (tmp_path / "urdf" / "meshes" / "shell.stl").resolve().as_uri()
    assert expected in references
    assert "meshes/shell.stl" not in references


def test_resource_path_meshes_are_left_alone(tmp_path: Path) -> None:
    """Only a resource path can resolve them; guessing one would be inventing."""
    profile = _profile(tmp_path)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order), extra=MESH_LINKS)
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    assert "package://vendor/badge.dae" in _mesh_references(generated)


def _compliant_profile(tmp_path: Path) -> Any:
    return _profile(
        tmp_path,
        contract=make_contract(
            dof=3,
            action=ActionSpec(
                type=ActionType.JOINT_POSITION, kp=[60.0, 100.0, 40.0], kd=[1.0, 2.0, 1.0]
            ),
        ),
        limits=Limits(joint_position=JointRange(min=[-1.0] * 3, max=[1.5] * 3)),
    )


def test_gains_become_the_trajectory_controllers_law(tmp_path: Path) -> None:
    """ADR-0019: effort command, position and velocity read back, and the
    contract's coefficients written per joint in the contract's order."""
    profile = _compliant_profile(tmp_path)
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    manager = document["/arm1/controller_manager"]["ros__parameters"]
    assert manager["joint_trajectory_controller"]["type"] == names.TRAJECTORY_CONTROLLER
    controller = document["/arm1/joint_trajectory_controller"]["ros__parameters"]
    assert controller["command_interfaces"] == ["effort"]
    assert controller["state_interfaces"] == ["position", "velocity"]
    assert controller["gains"] == {
        "joint0": {"p": 60.0, "d": 1.0},
        "joint1": {"p": 100.0, "d": 2.0},
        "joint2": {"p": 40.0, "d": 1.0},
    }
    # Without this the controller interpolates from the state at receipt and damps the
    # slope of its own ramp, (q* - q) / T -- not the law the contract states (ADR-0045).
    assert controller["interpolation_method"] == "none"


def test_an_effort_interface_is_not_bounded_by_angles(tmp_path: Path) -> None:
    """Position limits hung on a torque would bound the wrong quantity."""
    profile = _compliant_profile(tmp_path)
    generated = merged_description(_instance(tmp_path, profile), tmp_path / "out").content

    assert '<command_interface name="effort"' in generated
    assert '<param name="min">' not in generated


def test_anchoring_meshes_leaves_the_input_alone(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order), extra=MESH_LINKS)
    original = (tmp_path / "urdf" / "arm.urdf").read_text(encoding="utf-8")

    merged_description(_instance(tmp_path, profile), tmp_path / "out")

    assert (tmp_path / "urdf" / "arm.urdf").read_text(encoding="utf-8") == original
