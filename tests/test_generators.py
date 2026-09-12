"""Generation: what is derived, what is refused, and what stays put.

The tests are grouped the way ADR-0009 and ADR-0010 split the subject. First:
every derivation is a sentence of the contract turned into a sentence of the
control stack, so each one is checked against the contract it came from. Second:
the input is never written, and the output is the same twice.
"""

from __future__ import annotations

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

    assert document["controller_manager"]["ros__parameters"]["update_rate"] == 250


def test_joints_keep_the_order_of_the_contract(tmp_path: Path) -> None:
    order = ["shoulder", "elbow", "wrist"]
    profile = _profile(tmp_path, contract=make_contract(dof=3, joint_order=order))
    document = yaml.safe_load(controllers_config(_instance(tmp_path, profile)).content)

    assert document["joint_position_controller"]["ros__parameters"]["joints"] == order


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
