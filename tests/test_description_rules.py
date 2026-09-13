"""What the platform gains by not authoring kinematics.

Every test here is a disagreement between two descriptions of one robot that
would otherwise be discovered by watching the robot. Codes are asserted, not
wording (ADR-0007).
"""

from __future__ import annotations

from pathlib import Path

from conftest import make_contract, make_profile, urdf_text, write_profile, write_urdf
from heron.core.enums import AccessLevel, AngleUnit
from heron.core.models import JointMax, JointRange, Limits
from heron.validator.description import check_description


def _profile_with_description(
    tmp_path: Path,
    joints: list[str],
    *,
    urdf_joints: list[str] | None = None,
    urdf_kwargs: dict[str, object] | None = None,
    **overrides: object,
) -> tuple[Path, object]:
    overrides.setdefault("description", {"urdf": "urdf/checked.urdf"})
    overrides.setdefault("contract", make_contract(dof=len(joints), joint_order=joints))
    profile = make_profile(name="checked", dof=len(joints), **overrides)
    write_urdf(tmp_path, "checked", urdf_joints if urdf_joints is not None else joints,
               **(urdf_kwargs or {}))
    path = write_profile(tmp_path, profile)
    return path, profile


def test_a_joint_renamed_in_the_model_is_caught(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path, ["a", "b"], urdf_joints=["a", "renamed"]
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.joint_missing" in report.codes()
    assert not report.ok


def test_a_joint_the_model_welds_shut_is_an_error(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path, ["a"], urdf_kwargs={"kind": "fixed"}
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.joint_not_commandable" in report.codes()
    assert not report.ok


def test_a_movable_joint_outside_the_contract_is_only_a_warning(tmp_path: Path) -> None:
    """A gripper driven by its own channel is the normal case, not a defect."""
    path, profile = _profile_with_description(
        tmp_path, ["a", "b"], urdf_joints=["a", "b", "finger"]
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.joint_not_in_contract" in report.codes()
    assert report.ok


def test_a_profile_wider_than_the_model_is_an_error(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"lower": -1.0, "upper": 1.0},
        limits=Limits(joint_position=JointRange(min=[-2.0], max=[2.0])),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.position_limit_wider" in report.codes()


def test_a_profile_narrower_than_the_model_is_fine(tmp_path: Path) -> None:
    """``limits`` is a safety envelope: tightening it is how a robot is run
    carefully (section 12)."""
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"lower": -3.0, "upper": 3.0},
        limits=Limits(joint_position=JointRange(min=[-1.0], max=[1.0])),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert report.ok
    assert "urdf.position_limit_wider" not in report.codes()


def test_a_velocity_above_the_model_is_an_error(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"velocity": 1.0},
        limits=Limits(joint_velocity=JointMax(max=[5.0])),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.velocity_limit_wider" in report.codes()


# -- units -----------------------------------------------------------------


def test_degrees_are_converted_before_comparison(tmp_path: Path) -> None:
    """URDF is radians by specification. Comparing 180 with 3.14 as numbers
    would report a mismatch on a robot that is perfectly fine."""
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"lower": -3.15, "upper": 3.15},
        contract=make_contract(dof=1, joint_order=["a"], angle_unit=AngleUnit.DEG),
        limits=Limits(joint_position=JointRange(min=[-180.0], max=[180.0])),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert report.ok, report.codes()


def test_degrees_beyond_the_model_are_still_caught(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"lower": -3.0, "upper": 3.0},
        contract=make_contract(dof=1, joint_order=["a"], angle_unit=AngleUnit.DEG),
        limits=Limits(joint_position=JointRange(min=[-180.0], max=[180.0])),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.position_limit_wider" in report.codes()
    message = next(f for f in report if f.code == "urdf.position_limit_wider").message
    assert "deg" in message and "rad" in message, "сообщение обязано показать обе единицы"


# -- reachability ----------------------------------------------------------


def test_a_missing_description_is_an_error(tmp_path: Path) -> None:
    profile = make_profile(
        name="checked",
        dof=1,
        contract=make_contract(dof=1),
        description={"urdf": "urdf/checked.urdf"},
    )
    path = write_profile(tmp_path, profile)

    report = check_description(profile, path)
    assert report.codes() == ["urdf.file_not_found"]


def test_a_template_stops_the_comparison_with_a_warning(tmp_path: Path) -> None:
    """Not an error: the profile may be fine. But the checks did not run, and
    saying so is the difference between "checked" and "skipped"."""
    profile = make_profile(
        name="checked", dof=1, description={"urdf": "urdf/checked.urdf.xacro"}
    )
    (tmp_path / "urdf").mkdir()
    (tmp_path / "urdf" / "checked.urdf.xacro").write_text(
        urdf_text("checked", ["joint0"]), encoding="utf-8"
    )
    path = write_profile(tmp_path, profile)

    report = check_description(profile, path)
    assert report.codes() == ["urdf.not_rendered"]
    assert report.ok


def test_an_unreadable_description_is_reported_once(tmp_path: Path) -> None:
    profile = make_profile(
        name="checked", dof=1, description={"urdf": "urdf/checked.urdf"}
    )
    (tmp_path / "urdf").mkdir()
    (tmp_path / "urdf" / "checked.urdf").write_text("<robot><link>", encoding="utf-8")
    path = write_profile(tmp_path, profile)

    report = check_description(profile, path)
    assert report.codes() == ["urdf.unreadable"]


def test_a_description_naming_another_robot_is_a_warning(tmp_path: Path) -> None:
    profile = make_profile(
        name="checked",
        dof=1,
        contract=make_contract(dof=1),
        description={"urdf": "urdf/checked.urdf"},
    )
    (tmp_path / "urdf").mkdir()
    (tmp_path / "urdf" / "checked.urdf").write_text(
        urdf_text("someone_else", ["joint0"]), encoding="utf-8"
    )
    path = write_profile(tmp_path, profile)

    report = check_description(profile, path)
    assert "urdf.name_differs" in report.codes()
    assert report.ok


# -- meshes ----------------------------------------------------------------

MESH_LINK = """  <link name="decor">
    <visual><geometry><mesh filename="{reference}"/></geometry></visual>
  </link>
  <joint name="decor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="decor"/>
  </joint>
"""
"""A link carrying a mesh, attached -- an unattached link would make the tree
rootless and the test would be measuring the wrong rule."""


def test_a_missing_mesh_is_an_error(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={
            "extra": '  <link name="m"><visual><geometry>'
                     '<mesh filename="meshes/absent.stl"/>'
                     "</geometry></visual></link>\n"
        },
    )
    report = check_description(profile, path)  # type: ignore[arg-type]
    assert "urdf.mesh_not_found" in report.codes()


def test_a_resource_path_mesh_is_left_to_the_run(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={
            "extra": MESH_LINK.format(reference="package://vendor/part.dae")
        },
    )
    report = check_description(profile, path)  # type: ignore[arg-type]
    assert report.codes() == ["urdf.mesh_unresolvable"]
    assert report.ok


def test_a_relative_mesh_that_exists_is_found(tmp_path: Path) -> None:
    """The resolution the generator reuses (ADR-0017) must first be right here."""
    path, profile = _profile_with_description(
        tmp_path, ["a"], urdf_kwargs={"extra": MESH_LINK.format(reference="meshes/part.stl")}
    )
    (tmp_path / "urdf" / "meshes").mkdir()
    (tmp_path / "urdf" / "meshes" / "part.stl").write_bytes(b"solid part\nendsolid part\n")

    report = check_description(profile, path)  # type: ignore[arg-type]
    assert report.codes() == []


# -- the base --------------------------------------------------------------

WORLD_MOUNT = """  <link name="world"/>
  <joint name="mount" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>
"""


def test_a_free_base_with_the_built_in_controller_off_is_a_warning(tmp_path: Path) -> None:
    """Two facts, harmless apart: nothing holds the robot up (ADR-0018)."""
    path, profile = _profile_with_description(
        tmp_path, ["a"], access_levels=(AccessLevel.LOW,)
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert report.codes() == ["urdf.free_base_uncontrolled"]
    assert report.ok


def test_a_free_base_says_nothing_without_access_levels(tmp_path: Path) -> None:
    """A robot with no built-in controller to switch off: nothing was taken away."""
    path, profile = _profile_with_description(tmp_path, ["a"])
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.free_base_uncontrolled" not in report.codes()


def test_a_base_bolted_to_the_world_holds_itself(tmp_path: Path) -> None:
    path, profile = _profile_with_description(
        tmp_path,
        ["a"],
        urdf_kwargs={"extra": WORLD_MOUNT},
        access_levels=(AccessLevel.LOW,),
    )
    report = check_description(profile, path)  # type: ignore[arg-type]

    assert "urdf.free_base_uncontrolled" not in report.codes()
    assert "urdf.no_root_link" not in report.codes()
