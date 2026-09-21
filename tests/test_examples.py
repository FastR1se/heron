"""The shipped examples must validate, and must keep demonstrating what they
were written to demonstrate.

These tests skip themselves when ``examples/`` is absent: invariant 1 says the
core survives deleting the robots, and a test suite that fails without them
would contradict it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from heron.core.enums import ControlMode
from heron.core.loading import load_model
from heron.core.models import RobotProfile
from heron.registry.registry import Registry
from heron.scenario.loader import load_scenario
from heron.skills.library import SkillLibrary
from heron.validator.rules import check_profile, check_scenario, check_skill_against_profile


@pytest.fixture
def robots(examples_dir: Path) -> Path:
    return examples_dir / "robots"


def test_every_example_profile_validates(robots: Path) -> None:
    files = sorted(robots.glob("*.yaml"))
    assert files, "примеров профилей нет"
    for path in files:
        report = check_profile(load_model(path, RobotProfile))
        assert report.ok, f"{path.name}: {[f.code for f in report.errors]}"


def test_example_scenario_validates(examples_dir: Path, robots: Path) -> None:
    scenario = load_scenario(examples_dir / "scenarios" / "bench.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]


def test_registry_finds_examples_by_name(robots: Path) -> None:
    registry = Registry([robots])
    assert "arm6" in registry.names()
    assert registry.load("arm6").contract.dof == 6


def test_the_two_examples_differ_in_interface_class(robots: Path) -> None:
    """Section 9 wants reference implementations of different interface classes."""
    registry = Registry([robots])
    assert registry.load("arm6").control_mode is ControlMode.STREAMING
    assert registry.load("rest_arm").control_mode is ControlMode.COMMAND


def test_streaming_skill_is_rejected_on_the_command_robot(robots: Path) -> None:
    """The example that earns its place: same skill, two robots, one refusal."""
    registry = Registry([robots])
    spec = SkillLibrary().spec("hold")

    assert check_skill_against_profile(spec, registry.load("arm6")).ok

    report = check_skill_against_profile(spec, registry.load("rest_arm"))
    assert not report.ok
    assert "skill.control_mode_incompatible" in report.codes()


def test_locked_contracts_match_the_profiles(robots: Path) -> None:
    """The lock file in the repository is current (ADR-0004)."""
    from heron.registry.lock import ContractLock

    lock = ContractLock.load(robots)
    assert lock.entries, "contracts.lock.json пуст — выполните contract lock"
    registry = Registry([robots])
    for name, entry in lock.entries.items():
        assert registry.load(name).contract.contract_id == entry.contract_id


# -- phase 2: the examples carry descriptions and generate worlds ----------


def test_every_example_profile_agrees_with_its_description(robots: Path) -> None:
    """The examples exercise the rules of ADR-0009 on real files, not fixtures."""
    for path in sorted(robots.glob("*.yaml")):
        report = check_profile(load_model(path, RobotProfile), path=path)
        assert report.ok, f"{path.name}: {[f.code for f in report.errors]}"


def test_the_only_description_warning_is_the_gripper(robots: Path) -> None:
    """A warning the example is written to produce: the fingers are driven by
    the adapter, not by the contract's action vector."""
    path = robots / "arm6.yaml"
    report = check_profile(load_model(path, RobotProfile), path=path)
    codes = [code for code in report.codes() if code.startswith("urdf.")]
    assert codes == ["urdf.joint_not_in_contract"]


def test_the_reference_arm_is_bolted_to_the_world(robots: Path) -> None:
    """ADR-0048: the base is fixed in the description, and both worlds read it from there.

    Until phase 6 there was no ``world`` link, and the arm stood in the ``table``
    scene on a free base -- unnoticed, because nothing asked.
    """
    from heron.core.urdf import parse_urdf

    model = parse_urdf(robots / "urdf" / "arm6.urdf")
    assert model.fixed_to_world
    assert model.root_link == "world"


def test_the_reference_arm_states_what_a_policy_needs(robots: Path) -> None:
    """ADR-0041, ADR-0042, ADR-0045 on the example: a reference pose, a mapping, a law."""
    from heron.core.enums import ActionOffset, ObservationType

    contract = Registry([robots]).load("arm6").contract
    assert contract.revision == 2
    assert contract.default_pose is not None
    assert contract.action.has_gains
    assert contract.action.offset is ActionOffset.DEFAULT_POSE
    assert [term.type for term in contract.observation] == [
        ObservationType.JOINT_POS_REL,
        ObservationType.JOINT_VEL,
        ObservationType.TARGET_JOINT_POS,
        ObservationType.PREVIOUS_ACTION,
    ]


def test_every_goal_the_arm_can_be_commanded_is_one_it_can_start_from(
    examples_dir: Path, robots: Path
) -> None:
    """The set of targets covers the scene's goal and the pose the world starts in.

    The first is what the trained skill is asked for (ADR-0043); the second is where
    the arm stands when a run begins, and a policy that cannot command it cannot
    correct it.
    """
    contract = Registry([robots]).load("arm6").contract
    clip = contract.action.clip
    assert clip is not None
    reach = [
        (origin + clip[0] * scale, origin + clip[1] * scale)
        for origin, scale in zip(contract.origin(), contract.action_scale(), strict=True)
    ]
    task = load_scenario(examples_dir / "scenarios" / "table.yaml").task
    assert task is not None
    for (low, high), value in zip(reach, task.goal["target"], strict=True):
        assert low <= value <= high
    assert all(low <= 0.0 <= high for low, high in reach)


def test_the_command_robot_is_described_in_degrees(robots: Path) -> None:
    """Its contract says degrees, its description says radians, and the two
    agree only because the comparison converts."""
    from heron.core.enums import AngleUnit

    path = robots / "rest_arm.yaml"
    profile = load_model(path, RobotProfile)
    assert profile.contract.angle_unit is AngleUnit.DEG
    assert check_profile(profile, path=path).ok


def test_the_gazebo_scenario_validates(examples_dir: Path, robots: Path) -> None:
    scenario = load_scenario(examples_dir / "scenarios" / "table.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]


def test_the_gazebo_scenario_generates_a_world(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """Phase 2 end to end, minus the simulator: profile -> validator ->
    generators -> artifacts a world can be started from."""
    from heron.generators import build

    path = examples_dir / "scenarios" / "table.yaml"
    bundle = build(
        load_scenario(path), Registry([robots]), into=tmp_path / "out", scenario_path=path
    )
    bundle.write()

    assert (tmp_path / "out" / "bringup.launch.xml").is_file()
    assert (tmp_path / "out" / "arm1" / "controllers.yaml").is_file()
    # The obstacle gets a description and no controllers at all.
    assert (tmp_path / "out" / "arm2" / "description.urdf").is_file()
    assert not (tmp_path / "out" / "arm2" / "controllers.yaml").exists()
    assert bundle.differences() == []


# -- phase 4: a vendor description, fetched rather than copied -------------

G1 = "g1_23dof"
FETCH_HINT = "python examples/robots/g1_23dof/fetch_description.py"


def _vendor_profile(robots: Path) -> Path:
    """The G1 profile, skipping when its description was never fetched.

    The description is not in the repository (it is the vendor's, and 110 MB),
    so a checkout without the download is a normal state rather than a broken
    one. The tests that do not need the file run regardless.
    """
    from heron.generators.common import resolve_description

    path = robots / G1 / "profile.yaml"
    if not resolve_description(load_model(path, RobotProfile), path).is_file():
        pytest.skip(f"описание {G1} не загружено: {FETCH_HINT}")
    return path


def test_the_vendor_robot_is_found_by_its_directory(robots: Path) -> None:
    profile = Registry([robots]).load(G1)
    assert profile.contract.dof == 23


def test_the_vendor_robot_has_compliant_joints(robots: Path) -> None:
    """Regression, found by pushing the robot over in Gazebo: it fell as a
    statue. Its action states the drive's gains, so its joints are commanded
    with effort through the drive's law rather than held by a stiff servo
    (ADR-0019)."""
    from heron.ros2 import names

    profile = Registry([robots]).load(G1)
    assert profile.contract.action.has_gains
    assert names.command_interface(profile) == "effort"
    assert names.controller_type(profile) == names.TRAJECTORY_CONTROLLER


def test_the_vendor_robot_contract_is_locked(robots: Path) -> None:
    """A profile in its own directory keeps its lock next to it, where
    ``contract lock`` and ``validate`` both look."""
    from heron.registry.lock import ContractLock

    lock = ContractLock.load(robots / G1)
    entry = lock.get(G1)
    assert entry is not None, f"контракт {G1} не зафиксирован — выполните contract lock"
    assert Registry([robots]).load(G1).contract.contract_id == entry.contract_id


def test_the_vendor_robot_agrees_with_its_description(robots: Path) -> None:
    """23 joints, their limits and every mesh, against the vendor's own file.

    Two warnings, both conditions of a run rather than defects. The base is free
    and the built-in controller is off (ADR-0018). And the vendor's own
    ``<collision>`` elements are its visual surfaces: 21 meshes, 323 924
    triangles, the lightest of them 1422 (ADR-0073). That number is recorded here
    rather than quietly excused by a higher threshold -- it is the same file that
    every G1 scene is built from, and what a step of that world costs is a fact
    about it."""
    path = _vendor_profile(robots)
    report = check_profile(load_model(path, RobotProfile), path=path)

    assert report.ok, [f.code for f in report.errors]
    assert sorted(c for c in report.codes() if c.startswith("urdf.")) == [
        "urdf.collision_mesh_heavy",
        "urdf.free_base_uncontrolled",
    ]
    heavy = next(f for f in report.findings if f.code == "urdf.collision_mesh_heavy")
    assert sum(item["triangles"] for item in heavy.detail["collision_meshes"]) == 323_924


def test_the_vendor_robot_agrees_with_its_mjcf_and_names_where_the_two_disagree(
    robots: Path,
) -> None:
    """ADR-0031 on the vendor's two files: the profile fits both, and the one thing
    the files disagree on is the ankle torque found in stage 0 -- 35 N·m and 50."""
    path = _vendor_profile(robots)
    report = check_profile(load_model(path, RobotProfile), path=path)

    assert report.ok, [f.code for f in report.errors]
    findings = [item for item in report if item.code.startswith("mjcf.")]
    assert [item.code for item in findings] == ["mjcf.differs_from_urdf"]
    assert findings[0].detail["quantity"] == "effort"
    assert findings[0].detail["joints"] == {
        f"{side}_ankle_{axis}_joint": {"urdf": 35.0, "mjcf": 50.0}
        for side in ("left", "right")
        for axis in ("pitch", "roll")
    }


def test_the_vendor_twin_scenario_validates_and_starts_the_vendors_program(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """Stage 4 minus the simulator: the scene, the twin from the robot's package, the
    link from the profile. The package is installed wherever examples are (CI)."""
    from heron.sim.mujoco import MujocoBackend
    from heron.sim.process import PlanningLauncher

    _vendor_profile(robots)
    path = examples_dir / "scenarios" / "humanoid_mujoco.yaml"
    scenario = load_scenario(path)
    assert check_scenario(scenario, Registry([robots]), path=path).ok

    planner = PlanningLauncher()
    backend = MujocoBackend(
        Registry([robots]), into=tmp_path, launcher=planner, scenario_path=path
    )
    world = backend.build_world(scenario)
    backend.start()

    assert world.scene.world.name == "scene_23dof.xml"
    assert planner.commands == [
        ("unitree_mujoco", "-r", "g1", "-s", str(world.scene.world), "-i", "1", "-n", "lo")
    ]
    gazebo = load_scenario(examples_dir / "scenarios" / "humanoid.yaml")
    assert scenario.task == gazebo.task, "сверка двойников (этап 6) — одна цель в обоих мирах"


def test_the_vendor_robot_joins_gazebo_through_its_bridge(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """Stage 5 minus the simulator (ADR-0033, ADR-0034): the same adapter as in MuJoCo, its
    twin in Gazebo found by the pair, the scene in the environment of the same link."""
    import json
    from xml.etree import ElementTree as ET

    import yaml

    from heron.sim.gazebo import GazeboBackend
    from heron.sim.process import PlanningLauncher

    _vendor_profile(robots)
    path = examples_dir / "scenarios" / "humanoid.yaml"
    scenario = load_scenario(path)
    profile = Registry([robots]).load(G1)
    assert profile.adapters["sim.gazebo"] == profile.adapters["sim.mujoco"]
    assert profile.adapters["sim.gazebo"].plugin == "unitree_g1_dds"

    planner = PlanningLauncher()
    backend = GazeboBackend(Registry([robots]), into=tmp_path, launcher=planner, scenario_path=path)
    world = backend.build_world(scenario)
    backend.start()
    assert planner.commands and planner.commands[0][:2] == ("ros2", "launch")

    files = {str(item.path): item.content for item in world.bundle.files}
    manifest = json.loads(files["manifest.json"])
    assert manifest["robots"][0]["control"]["command_topic"] == "/lowcmd"
    assert manifest["robots"][0]["control"]["state_message"] == "unitree_hg/msg/LowState"

    controllers = yaml.safe_load(files["g1/controllers.yaml"])
    bridge = controllers["/g1/low_level_bridge"]["ros__parameters"]
    assert bridge["joints"] == list(profile.contract.joint_order)
    assert bridge["mode_machine"] == 4
    assert bridge["slots"][13:] == [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]

    launch = ET.fromstring(files["bringup.launch.xml"].split("?>", 1)[1])
    environment = {item.get("name"): item.get("value") for item in launch.iter("set_env")}
    assert environment["ROS_DOMAIN_ID"] == "1"
    assert environment["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert '<Peer address="127.0.0.1"' in (environment["CYCLONEDDS_URI"] or "")


ARM_GOAL = {"target": [0.0, -0.6, 1.2, 0.0, 0.9, 0.0], "duration": 4.0}


def test_the_complex_scene_holds_two_links_in_one_world(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """Stage 7 minus the simulator: a robot through ros2_control and a robot through its
    vendor's DDS, one owner of the physics (ADR-0002), one environment for the world
    (ADR-0034), and the run of either robot joins it (ADR-0039)."""
    import json
    from xml.etree import ElementTree as ET

    import yaml

    from heron.cli.run import execute
    from heron.generators.bundle import world_environment
    from heron.sim.gazebo import GazeboBackend
    from heron.sim.process import PlanningLauncher

    _vendor_profile(robots)
    path = examples_dir / "scenarios" / "complex.yaml"
    scenario = load_scenario(path)
    registry = Registry([robots])
    assert check_scenario(scenario, registry, path=path).ok
    assert scenario.task is None, "задачи раздаёт координатор (фаза 8), а не сценарий"

    planner = PlanningLauncher()
    backend = GazeboBackend(registry, into=tmp_path, launcher=planner, scenario_path=path)
    world = backend.build_world(scenario)
    backend.start()
    assert len(planner.commands) == 1, "мир один — один запуск"

    files = {str(item.path): item.content for item in world.bundle.files}
    control = {
        item["id"]: item["control"] for item in json.loads(files["manifest.json"])["robots"]
    }
    assert control["arm1"]["state_topic"] == "/arm1/joint_states"
    assert control["g1"]["state_topic"] == "/lowstate"

    assert "/arm1/controller_manager" in yaml.safe_load(files["arm1/controllers.yaml"])
    assert "/g1/low_level_bridge" in yaml.safe_load(files["g1/controllers.yaml"])
    for robot in ("arm1", "g1"):
        description = ET.fromstring(files[f"{robot}/description.urdf"].split("?>", 1)[1])
        namespaces = [item.text for item in description.iter("namespace")]
        assert namespaces == [robot], "у каждого робота свой менеджер контроллеров"

    launch = ET.fromstring(files["bringup.launch.xml"].split("?>", 1)[1])
    variables = [(item.get("name"), item.get("value")) for item in launch.iter("set_env")]
    environment = world_environment(scenario, registry)
    assert dict(variables) == environment and len(variables) == len(environment)
    assert environment["ROS_DOMAIN_ID"] == "1"

    humanoid = load_scenario(examples_dir / "scenarios" / "humanoid.yaml").task
    assert humanoid is not None
    for robot, goal in (("arm1", ARM_GOAL), ("g1", humanoid.goal)):
        lines, _ = execute(
            scenario, registry, robot_id=robot, skill="move_joints", goal=goal,
            dry_run=True, environ={}, scenario_path=path,
        )
        assert any(line.startswith("окружение") and "ROS_DOMAIN_ID=1" in line for line in lines)


def test_the_arm_goal_of_the_complex_scene_is_the_table_scenes(examples_dir: Path) -> None:
    """Stage 7 drives the arm to the goal phase 3 proved it reaches."""
    table = load_scenario(examples_dir / "scenarios" / "table.yaml").task
    assert table is not None and table.goal == ARM_GOAL


def test_the_vendor_robot_generates_with_every_mesh_anchored(
    robots: Path, tmp_path: Path
) -> None:
    """ADR-0017 on a real description: nothing in the copy resolves against a
    directory the simulator will never see."""
    from xml.etree import ElementTree as ET

    from heron.generators import build_profile

    path = _vendor_profile(robots)
    profile = load_model(path, RobotProfile)
    bundle = build_profile(profile, path, into=tmp_path / "out", backend="gazebo")
    description = next(f for f in bundle.files if str(f.path) == f"{G1}/description.urdf")

    tree = ET.fromstring(description.content.split("?>", 1)[1])
    references = [mesh.get("filename", "") for mesh in tree.iter("mesh")]
    assert references
    assert all(ref.startswith("file://") for ref in references), references[:3]


def test_the_vendor_robot_scenario_validates_and_its_goal_fits(
    examples_dir: Path, robots: Path
) -> None:
    """The scene the Linux run starts from: valid, and a goal the skill accepts."""
    from heron.core.enums import Role

    _vendor_profile(robots)
    scenario = load_scenario(examples_dir / "scenarios" / "humanoid.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]

    profile = Registry([robots]).load(G1)
    driven = [robot for robot in scenario.robots if robot.role is not Role.STATIC]
    assert [robot.profile for robot in driven] == [G1]

    assert scenario.task is not None
    target = scenario.task.goal["target"]
    bounds = profile.limits.joint_position
    assert bounds is not None
    assert len(target) == profile.contract.dof
    assert all(lo <= v <= hi for v, lo, hi in zip(target, bounds.min, bounds.max, strict=True))


def test_the_command_robot_moves_in_a_world_of_numbers(examples_dir: Path, robots: Path) -> None:
    """The command class end to end on its example, before any such robot exists (ADR-0049)."""
    from heron.cli.run import execute

    scenario = load_scenario(examples_dir / "scenarios" / "rest_bench.yaml")
    _, result = execute(scenario, Registry([robots]))

    assert result is not None and result.ok, result
    assert len(result.commands) == 1
    assert result.states[-1].joint_position == pytest.approx((30.0, -20.0, 15.0, 0.0))


def test_the_same_mission_goes_to_hardware_by_target(examples_dir: Path, robots: Path) -> None:
    """One scenario, two owners of physics (ADR-0051): the plan says which."""
    from heron.cli.run import execute

    scenario = load_scenario(examples_dir / "scenarios" / "rest_bench.yaml")
    lines, result = execute(
        scenario, Registry([robots]), target="real", speed_scale=0.1, dry_run=True
    )

    assert result is None
    assert "мир       действительность, окружение мира не применяется" in lines
    assert "шкала     скорости 0.1" in lines


# -- phase 6, stage 7: the G1 bolted to a stand (ADR-0063, ADR-0064) ---------------------

MOUNTED = "g1_29dof_lock_waist_rev_1_0_mounted"
MOUNTED_FETCH = f"python examples/robots/{MOUNTED}/fetch_description.py"


def _mounted_profile(robots: Path) -> Path:
    """The mounted G1's profile, skipping when its two inputs were never written."""
    from heron.generators.common import resolve_description

    path = robots / MOUNTED / "profile.yaml"
    if not resolve_description(load_model(path, RobotProfile), path).is_file():
        pytest.skip(f"описание {MOUNTED} не загружено: {MOUNTED_FETCH}")
    return path


def test_the_mounted_robot_is_the_vendors_description_plus_a_stand(robots: Path) -> None:
    """ADR-0063, item 1: exactly three changes to the vendor's file -- a link, a joint, a name."""
    from xml.etree import ElementTree as ET

    from heron.generators.common import resolve_description

    path = _mounted_profile(robots)
    mounted = ET.parse(resolve_description(load_model(path, RobotProfile), path)).getroot()
    vendor_file = (
        robots / MOUNTED / "vendor" / "unitree_ros" / "robots" / "g1_description"
        / "g1_29dof_lock_waist_rev_1_0.urdf"
    )
    vendor = ET.parse(vendor_file).getroot()

    assert (vendor.get("name"), mounted.get("name")) == ("g1_29dof_lock_waist_rev_1_0", MOUNTED)
    added = [child for child in mounted if child.get("name") in ("world", "world_joint")]
    assert [child.tag for child in added] == ["link", "joint"]
    joint = added[1]
    assert joint.get("type") == "fixed"
    assert (joint.find("parent").get("link"), joint.find("child").get("link")) == (  # type: ignore[union-attr]
        "world",
        "pelvis",
    )
    assert joint.find("origin").get("xyz") == "0 0 1"  # type: ignore[union-attr]

    rest = [ET.tostring(child) for child in mounted if child not in added]
    assert rest == [ET.tostring(child) for child in vendor]


def test_the_mounted_robot_validates_with_only_its_passive_legs_named(robots: Path) -> None:
    """The base is held, so no free-base condition; the legs are passive links (ADR-0012)."""
    path = _mounted_profile(robots)
    report = check_profile(load_model(path, RobotProfile), path=path)
    assert report.ok, [f.code for f in report.errors]
    assert report.codes() == ["urdf.joint_not_in_contract"]
    (finding,) = report.findings
    assert len(finding.detail["joints"]) == 12
    assert all(("hip" in name or "knee" in name or "ankle" in name)
               for name in finding.detail["joints"])


def test_the_mounted_robot_trains_on_a_fixed_base_at_the_rates_of_its_link(
    robots: Path,
) -> None:
    """The derivation of ADR-0044 on the second robot of the phase, without its task."""
    from heron.core.capabilities import Capability
    from heron.core.models import Completion, SkillSpec
    from heron.generators.common import resolve_description
    from heron.train.spec import derive

    path = _mounted_profile(robots)
    profile = load_model(path, RobotProfile)
    spec = derive(
        scenario="g1_mounted_train",
        robot="g1",
        backend="isaaclab",
        task="t",
        profile=profile,
        description=resolve_description(profile, path),
        pose=(0.0, 0.0, 0.0),
        skill=SkillSpec(
            name="t", required_capabilities=frozenset({Capability("arm.joint_position")})
        ),
        completion=Completion(tolerance=0.08, hold_s=0.3, horizon_s=5.0),
    )
    assert spec.fix_base
    assert (spec.dt, spec.decimation) == (pytest.approx(0.002), 10)
    assert (spec.action_size, spec.observation_size) == (15, 60)
    assert spec.action.offset == list(profile.contract.default_pose or [])
    assert spec.drive.kd[5] == pytest.approx(0.18), "крен запястья — kd ниже I_min/dt (ADR-0063)"


def test_the_mounted_robot_joins_gazebo_as_an_upper_body_on_the_bridge(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """ADR-0064 through the platform's own world: the bridge drives fifteen joints of body 6,
    the legs are read and never commanded, the description carries the stand."""
    from xml.etree import ElementTree as ET

    import yaml

    from heron.sim.gazebo import GazeboBackend
    from heron.sim.process import PlanningLauncher

    _mounted_profile(robots)
    path = examples_dir / "scenarios" / "g1_mounted.yaml"
    scenario = load_scenario(path)
    report = check_scenario(scenario, Registry([robots]), path=path)
    assert report.ok, [f.code for f in report.errors]

    backend = GazeboBackend(
        Registry([robots]), into=tmp_path, launcher=PlanningLauncher(), scenario_path=path
    )
    files = {str(item.path): item.content for item in backend.build_world(scenario).bundle.files}

    bridge = yaml.safe_load(files["g1/controllers.yaml"])["/g1/low_level_bridge"][
        "ros__parameters"
    ]
    assert bridge["mode_machine"] == 6
    assert bridge["slots"] == [12, *range(15, 29)]

    description = ET.fromstring(files["g1/description.urdf"].split("?>", 1)[1])
    assert any(
        joint.get("name") == "world_joint" and joint.get("type") == "fixed"
        for joint in description.iter("joint")
    )
    block = description.find("ros2_control")
    assert block is not None
    joints = list(block.iter("joint"))
    commanded = {j.get("name") for j in joints if j.find("command_interface") is not None}
    read_only = {str(j.get("name")) for j in joints if j.find("command_interface") is None}
    assert commanded == set(bridge["joints"]) and len(commanded) == 15
    assert len(read_only) == 12
    assert all("hip" in n or "knee" in n or "ankle" in n for n in read_only)


def test_the_goal_of_the_mounted_scene_is_the_humanoids_for_the_arms(examples_dir: Path) -> None:
    """One arm pose across the G1 scenes: the stage 7 comparison starts from a known goal."""
    mounted = load_scenario(examples_dir / "scenarios" / "g1_mounted.yaml").task
    humanoid = load_scenario(examples_dir / "scenarios" / "humanoid.yaml").task
    assert mounted is not None and humanoid is not None
    arms_23 = humanoid.goal["target"][13:]
    target = mounted.goal["target"]
    left, right = target[1:8], target[8:15]
    assert left[:5] == arms_23[:5] and right[:5] == arms_23[5:]
    assert left[5:] == right[5:] == [0.0, 0.0]
