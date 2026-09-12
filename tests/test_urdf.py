"""Reading a description.

The parser is deliberately narrow: it extracts what the platform compares with a
profile and leaves everything else alone. These tests hold that line from both
sides -- what must be extracted, and what must be refused rather than guessed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import urdf_text, write_urdf
from heron.core.loading import LoadError
from heron.core.urdf import is_template, parse_urdf, parse_urdf_text


def test_joints_links_and_limits_are_read(tmp_path: Path) -> None:
    path = write_urdf(tmp_path, "arm", ["a", "b"], lower=-1.5, upper=1.5, velocity=2.0)
    model = parse_urdf(path)

    assert model.name == "arm"
    assert model.joint_names == ("a", "b")
    assert model.links == ("base_link", "link0", "link1")
    joint = model.joint("a")
    assert joint is not None
    assert (joint.limit.lower, joint.limit.upper, joint.limit.velocity) == (-1.5, 1.5, 2.0)
    assert joint.commandable and joint.angular


def test_root_link_is_the_one_no_joint_owns(tmp_path: Path) -> None:
    model = parse_urdf(write_urdf(tmp_path, "arm", ["a", "b", "c"]))
    assert model.root_link == "base_link"


def test_a_description_without_a_single_root_has_none() -> None:
    text = (
        '<?xml version="1.0"?><robot name="two">'
        '<link name="a"/><link name="b"/><link name="c"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="c"/>'
        '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
        "</robot>"
    )
    model = parse_urdf_text(text, Path("<память>"))
    assert model.root_link is None


def test_fixed_joints_are_not_commandable(tmp_path: Path) -> None:
    model = parse_urdf(write_urdf(tmp_path, "arm", ["a"], kind="fixed"))
    joint = model.joint("a")
    assert joint is not None
    assert not joint.commandable


def test_prismatic_joints_are_commandable_but_not_angular(tmp_path: Path) -> None:
    model = parse_urdf(write_urdf(tmp_path, "arm", ["a"], kind="prismatic"))
    joint = model.joint("a")
    assert joint is not None
    assert joint.commandable and not joint.angular


def test_meshes_are_collected_but_not_read() -> None:
    text = urdf_text(
        "arm",
        ["a"],
        extra=(
            '  <link name="extra"><visual><geometry>'
            '<mesh filename="package://somewhere/part.dae"/>'
            "</geometry></visual></link>\n"
        ),
    )
    model = parse_urdf_text(text, Path("<память>"))
    assert model.meshes == ("package://somewhere/part.dae",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<?xml version="1.0"?><notrobot/>', "ожидался <robot>"),
        ('<?xml version="1.0"?><robot><joint type="fixed"/></robot>', "нет атрибута name"),
        ('<?xml version="1.0"?><robot><joint name="j"/></robot>', "нет атрибута type"),
        (
            '<?xml version="1.0"?><robot><joint name="j" type="wiggly"/></robot>',
            "неизвестный тип",
        ),
        ('<?xml version="1.0"?><robot><link name="a"/><link name="a"/></robot>', "повторы имён"),
    ],
)
def test_malformed_descriptions_are_refused(text: str, expected: str) -> None:
    with pytest.raises(LoadError) as caught:
        parse_urdf_text(text, Path("<память>"))
    assert expected in str(caught.value)


def test_broken_xml_is_reported_as_a_problem_not_a_crash() -> None:
    with pytest.raises(LoadError):
        parse_urdf_text("<robot><link>", Path("<память>"))


def test_a_template_is_refused_rather_than_guessed_at(tmp_path: Path) -> None:
    """ADR-0009: expanding needs ROS 2, and pretending it parsed would make
    every later comparison a check that silently did nothing."""
    path = tmp_path / "arm.urdf.xacro"
    path.write_text(urdf_text("arm", ["a"]), encoding="utf-8")

    assert is_template(path)
    with pytest.raises(LoadError) as caught:
        parse_urdf(path)
    assert "шаблон" in str(caught.value)


def test_a_missing_file_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError):
        parse_urdf(tmp_path / "absent.urdf")
