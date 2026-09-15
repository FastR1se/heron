"""An MJCF description, read and held against the profile and the URDF (ADR-0031).

The parser is tested where a raw read of the attributes would be wrong: includes,
default classes, degrees, force bounds spread over a joint and its actuator. The
rules are tested on documents written by hand, like every description in these
tests: a description is an input (ADR-0009), and a test that needs a broken one
breaks it explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from conftest import make_contract, make_profile, urdf_text, write_profile, write_urdf
from heron.core.enums import AngleUnit
from heron.core.loading import LoadError, load_model
from heron.core.mjcf import DEFAULT_TIMESTEP, parse_mjcf
from heron.core.models import AdapterSpec, JointRange, Limits, RobotProfile, Scenario
from heron.registry.registry import Registry
from heron.validator.report import Report
from heron.validator.rules import check_profile, check_scenario

JOINTS = ["joint0", "joint1", "joint2"]


def mjcf_text(
    joints: Sequence[str] = JOINTS,
    *,
    angle: str = "radian",
    low: float = -3.0,
    high: float = 3.0,
    effort: float = 10.0,
    actuated: Sequence[str] | None = None,
    kind: str = "hinge",
    option: str = "",
    meshes: Sequence[str] = (),
) -> str:
    """A chain of joints under one free body, each with a motor unless told otherwise."""
    bodies, closing = [], []
    for index, name in enumerate(joints):
        bodies.append(
            f'<body name="link{index}"><joint name="{name}" type="{kind}" '
            f'range="{low} {high}" actuatorfrcrange="{-effort} {effort}"/>'
        )
        closing.append("</body>")
    motors = "".join(
        f'<motor name="{name}_motor" joint="{name}"/>'
        for name in (joints if actuated is None else actuated)
    )
    assets = "".join(f'<mesh name="m{i}" file="{file}"/>' for i, file in enumerate(meshes))
    return (
        f'<mujoco model="arm"><compiler angle="{angle}" meshdir="meshes"/>{option}'
        f"<asset>{assets}</asset>"
        f'<worldbody><body name="base_link"><freejoint name="root"/>'
        f"{''.join(bodies)}{''.join(closing)}</body></worldbody>"
        f"<actuator>{motors}</actuator></mujoco>"
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# -- reading -------------------------------------------------------------------


def test_an_include_is_read_from_the_directory_of_the_main_file(tmp_path: Path) -> None:
    model = write(tmp_path / "robot" / "arm.xml", mjcf_text())
    scene = write(
        tmp_path / "robot" / "scene.xml",
        '<mujoco model="scene"><include file="arm.xml"/>'
        '<worldbody><geom name="floor" type="plane" size="0 0 1"/></worldbody></mujoco>',
    )

    parsed = parse_mjcf(scene)

    assert parsed.name == "scene"
    assert parsed.joint_names == ("root", *JOINTS)
    assert parsed.includes == (model.resolve(),)
    assert parsed.loads(model)
    assert [item.joint for item in parsed.actuators] == JOINTS


def test_an_include_that_includes_itself_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "a.xml", '<mujoco><include file="b.xml"/></mujoco>')
    write(tmp_path / "b.xml", '<mujoco><include file="a.xml"/></mujoco>')
    with pytest.raises(LoadError, match="по кругу"):
        parse_mjcf(tmp_path / "a.xml")


def test_a_document_that_is_not_mjcf_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="<robot>"):
        parse_mjcf(write(tmp_path / "arm.xml", "<robot/>"))


def test_defaults_fill_what_an_element_leaves_out(tmp_path: Path) -> None:
    """Own attribute, then the class, then its parent class; childclass of a body counts."""
    parsed = parse_mjcf(
        write(
            tmp_path / "arm.xml",
            '<mujoco><compiler angle="radian"/>'
            '<default><joint range="-1 1"/>'
            '  <default class="wide"><joint range="-2 2" actuatorfrcrange="-5 5"/>'
            '    <default class="slide"><joint type="slide"/></default>'
            "  </default>"
            "</default>"
            '<worldbody><body name="base" childclass="wide">'
            '  <joint name="from_childclass"/>'
            '  <joint name="own_range" range="-0.5 0.5"/>'
            '  <joint name="nested" class="slide"/>'
            '  <body name="plain" childclass="main"><joint name="from_main"/></body>'
            "</body></worldbody></mujoco>",
        )
    )

    childclass = parsed.joint("from_childclass")
    assert childclass is not None
    assert (childclass.range, childclass.effort) == ((-2.0, 2.0), 5.0)
    own = parsed.joint("own_range")
    assert own is not None and own.range == (-0.5, 0.5)
    nested = parsed.joint("nested")
    assert nested is not None and (nested.type, nested.range) == ("slide", (-2.0, 2.0))
    main = parsed.joint("from_main")
    assert main is not None and main.range == (-1.0, 1.0)


def test_angles_are_degrees_unless_the_compiler_says_radian(tmp_path: Path) -> None:
    text = mjcf_text(["hinge"], low=-90, high=45).replace(' angle="radian"', "")
    joint = parse_mjcf(write(tmp_path / "arm.xml", text)).joint("hinge")
    assert joint is not None and joint.range is not None
    assert joint.range == pytest.approx((-math.pi / 2, math.pi / 4))

    text = mjcf_text(["rail"], kind="slide", angle="degree", low=-0.2, high=0.3)
    rail = parse_mjcf(write(tmp_path / "slide.xml", text)).joint("rail")
    assert rail is not None and rail.range == (-0.2, 0.3)


def test_a_range_stated_as_not_limited_is_no_limit(tmp_path: Path) -> None:
    text = mjcf_text(["free_spin"]).replace('type="hinge"', 'type="hinge" limited="false"')
    joint = parse_mjcf(write(tmp_path / "arm.xml", text)).joint("free_spin")
    assert joint is not None and joint.range is None


def test_the_force_on_a_joint_is_its_tightest_bound(tmp_path: Path) -> None:
    """actuatorfrcrange on the joint, ctrlrange times gear on a motor: the smaller wins."""
    parsed = parse_mjcf(
        write(
            tmp_path / "arm.xml",
            '<mujoco><worldbody><body name="b">'
            '<joint name="geared" actuatorfrcrange="-50 50"/>'
            '<joint name="bare"/>'
            "</body></worldbody><actuator>"
            '<motor joint="geared" ctrlrange="-1 1" gear="20"/>'
            '<motor joint="bare"/>'
            "</actuator></mujoco>",
        )
    )
    assert parsed.effort("geared") == 20.0
    assert parsed.effort("bare") is None


def test_the_step_is_the_option_or_mujocos_default(tmp_path: Path) -> None:
    assert parse_mjcf(write(tmp_path / "a.xml", mjcf_text())).timestep == DEFAULT_TIMESTEP
    stepped = mjcf_text(option='<option timestep="0.001"/>')
    assert parse_mjcf(write(tmp_path / "b.xml", stepped)).timestep == 0.001


def test_meshes_are_found_under_the_meshdir_of_the_main_file(tmp_path: Path) -> None:
    parsed = parse_mjcf(write(tmp_path / "arm.xml", mjcf_text(meshes=["base.stl"])))
    assert parsed.meshes == (("base.stl", tmp_path.resolve() / "meshes" / "base.stl"),)


# -- a profile against its MJCF ------------------------------------------------


def _profile_path(tmp_path: Path, text: str | None = None, **overrides: Any) -> Path:
    """A profile with a URDF and, unless ``text`` is None, an MJCF next to it."""
    description = {"urdf": "urdf/arm.urdf"}
    if text is not None:
        description["mjcf"] = "mjcf/arm.xml"
        write(tmp_path / "mjcf" / "arm.xml", text)
    profile = make_profile(name="arm", description=description, **overrides)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    return write_profile(tmp_path, profile)


def _report(path: Path) -> Report:
    return check_profile(load_model(path, RobotProfile), path=path)


def _mjcf_codes(path: Path) -> list[str]:
    return sorted(code for code in _report(path).codes() if code.startswith("mjcf."))


def test_a_profile_without_an_mjcf_is_not_asked_about_one(tmp_path: Path) -> None:
    assert _mjcf_codes(_profile_path(tmp_path)) == []


def test_descriptions_that_agree_give_no_finding(tmp_path: Path) -> None:
    """conftest's URDF: limits ±3, effort 10 -- the same numbers in MJCF."""
    assert _mjcf_codes(_profile_path(tmp_path, mjcf_text())) == []


def test_a_contract_joint_the_mjcf_lacks_is_an_error(tmp_path: Path) -> None:
    assert _mjcf_codes(_profile_path(tmp_path, mjcf_text(["joint0", "joint1"]))) == [
        "mjcf.joint_missing"
    ]


def test_a_joint_without_an_actuator_is_an_error(tmp_path: Path) -> None:
    report = _report(_profile_path(tmp_path, mjcf_text(actuated=["joint0", "joint1"])))
    finding = next(item for item in report if item.code == "mjcf.joint_not_actuated")
    assert finding.detail == {"joint": "joint2"}


def test_a_joint_mjcf_describes_as_a_ball_is_not_commandable(tmp_path: Path) -> None:
    text = mjcf_text(kind="ball", low=0, high=1)
    assert "mjcf.joint_not_commandable" in _mjcf_codes(_profile_path(tmp_path, text))


def test_a_profile_wider_than_the_mjcf_is_an_error(tmp_path: Path) -> None:
    """Narrower is the safety envelope doing its job; wider is a promise the model cannot keep."""
    narrow = _profile_path(tmp_path / "narrow", mjcf_text(low=-2.5, high=2.5))
    assert "mjcf.position_limit_wider" not in _mjcf_codes(narrow)

    wide = _profile_path(tmp_path / "wide", mjcf_text(low=-1.0, high=1.0))
    assert "mjcf.position_limit_wider" in _mjcf_codes(wide)


def test_limits_in_degrees_are_compared_in_radians(tmp_path: Path) -> None:
    path = _profile_path(
        tmp_path,
        mjcf_text(low=-1.6, high=1.6),
        contract=make_contract(angle_unit=AngleUnit.DEG),
        limits=Limits(joint_position=JointRange(min=[-90.0] * 3, max=[90.0] * 3)),
    )
    assert "mjcf.position_limit_wider" not in _mjcf_codes(path)


def test_a_missing_mesh_is_an_error(tmp_path: Path) -> None:
    text = mjcf_text(meshes=["base.stl"])
    assert "mjcf.mesh_not_found" in _mjcf_codes(_profile_path(tmp_path, text))


def test_a_missing_and_an_unreadable_mjcf_are_named(tmp_path: Path) -> None:
    description = {"urdf": "urdf/arm.urdf", "mjcf": "nowhere.xml"}
    profile = make_profile(name="arm", description=description)
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    assert _mjcf_codes(write_profile(tmp_path, profile)) == ["mjcf.file_not_found"]

    assert _mjcf_codes(_profile_path(tmp_path / "broken", "<mujoco>")) == ["mjcf.unreadable"]


def test_two_descriptions_disagreeing_is_a_warning_with_both_numbers(tmp_path: Path) -> None:
    """The first difference phase 4 found was of this kind: 35 N·m in one file, 50 in the other."""
    report = _report(_profile_path(tmp_path, mjcf_text(effort=50.0, low=-2.9, high=3.0)))

    findings = [item for item in report if item.code == "mjcf.differs_from_urdf"]
    assert {item.detail["quantity"] for item in findings} == {"effort", "range"}
    assert report.ok
    effort = next(item for item in findings if item.detail["quantity"] == "effort")
    assert effort.detail["joints"]["joint0"] == {"urdf": 10.0, "mjcf": 50.0}
    assert "URDF 10, MJCF 50" in effort.message


def test_a_difference_below_the_printed_precision_is_none(tmp_path: Path) -> None:
    """1.97222 against 1.972222054: five decimals in one file, nine in the other."""
    wide = 1.972222054
    path = _profile_path(
        tmp_path,
        mjcf_text(low=-1.97222, high=1.97222),
        limits=Limits(joint_position=JointRange(min=[-wide] * 3, max=[wide] * 3)),
    )
    (tmp_path / "urdf" / "arm.urdf").write_text(
        urdf_text("arm", JOINTS, lower=-wide, upper=wide), encoding="utf-8"
    )
    assert _mjcf_codes(path) == []


# -- a MuJoCo world ------------------------------------------------------------


def _world(
    tmp_path: Path,
    *,
    scene: str | None = "scene.xml",
    scene_text: str = '<mujoco><include file="mjcf/arm.xml"/></mujoco>',
    mjcf: bool = True,
    state_rate_hz: float = 100.0,
) -> tuple[Scenario, Registry, Path]:
    """A MuJoCo scenario on disk. The adapter is not installed: twin rules stay out of it."""
    description = {"urdf": "urdf/arm.urdf"}
    if mjcf:
        description["mjcf"] = "mjcf/arm.xml"
        write(tmp_path / "mjcf" / "arm.xml", mjcf_text())
    profile = make_profile(
        name="arm",
        description=description,
        contract=make_contract(state_rate_hz=state_rate_hz),
        adapters={"sim.mujoco": AdapterSpec(plugin="vendor_absent")},
    )
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    write(tmp_path / "scene.xml", scene_text)
    world = {"backend": "mujoco", **({"file": scene} if scene is not None else {})}
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": world,
            "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        }
    )
    return scenario, Registry([tmp_path]), tmp_path / "scenario.yaml"


def _scene_codes(scenario: Scenario, registry: Registry, path: Path | None) -> list[str]:
    report = check_scenario(scenario, registry, path=path)
    return sorted(code for code in report.codes() if code.startswith("mjcf."))


def test_a_scene_loading_the_checked_description_gives_no_finding(tmp_path: Path) -> None:
    assert _scene_codes(*_world(tmp_path)) == []


def test_a_mujoco_world_without_a_scene_is_an_error(tmp_path: Path) -> None:
    assert _scene_codes(*_world(tmp_path, scene=None)) == ["mjcf.scene_missing"]


def test_a_scene_loading_another_description_is_an_error(tmp_path: Path) -> None:
    write(tmp_path / "other.xml", mjcf_text())
    scenario, registry, path = _world(
        tmp_path, scene_text='<mujoco><include file="other.xml"/></mujoco>'
    )
    report = check_scenario(scenario, registry, path=path)
    finding = next(item for item in report if item.code == "mjcf.scene_without_description")
    assert "other.xml" in finding.message


def test_a_scene_for_a_profile_without_an_mjcf_is_an_error(tmp_path: Path) -> None:
    write(tmp_path / "other.xml", mjcf_text())
    scenario, registry, path = _world(
        tmp_path, mjcf=False, scene_text='<mujoco><include file="other.xml"/></mujoco>'
    )
    assert _scene_codes(scenario, registry, path) == ["mjcf.scene_without_description"]


def test_a_scene_stepping_slower_than_the_contract_reports_is_an_error(tmp_path: Path) -> None:
    text = '<mujoco><option timestep="0.02"/><include file="mjcf/arm.xml"/></mujoco>'
    assert _scene_codes(*_world(tmp_path, scene_text=text, state_rate_hz=100.0)) == [
        "mjcf.timestep_slower_than_state"
    ]
    assert _scene_codes(*_world(tmp_path / "slow", scene_text=text, state_rate_hz=50.0)) == []


def test_a_missing_scene_file_is_named(tmp_path: Path) -> None:
    scenario, registry, path = _world(tmp_path, scene="absent.xml")
    assert _scene_codes(scenario, registry, path) == ["mjcf.file_not_found"]


def test_a_scenario_without_its_path_skips_the_files_it_names(tmp_path: Path) -> None:
    scenario, registry, _ = _world(tmp_path, scene="absent.xml")
    assert _scene_codes(scenario, registry, None) == []
