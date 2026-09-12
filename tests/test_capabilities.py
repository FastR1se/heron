"""Capability keys behave as ADR-0001 says, including where it says "error"."""

from __future__ import annotations

import pytest

from heron.core.capabilities import (
    CORE_CAPABILITIES,
    Capability,
    CapabilityError,
    missing,
    parse_set,
)


def test_core_keys_parse() -> None:
    for key in CORE_CAPABILITIES:
        assert Capability(key) == key


@pytest.mark.parametrize(
    "bad",
    [
        "arm",  # one segment
        "arm.joint.position",  # three
        "Arm.joint_position",  # upper case
        "arm.",  # empty aspect
        "1arm.position",  # digit first
        "arm joint.position",  # space
    ],
)
def test_malformed_keys_rejected(bad: str) -> None:
    with pytest.raises(CapabilityError):
        Capability(bad)


def test_typo_is_an_error_not_an_empty_intersection() -> None:
    """ADR-0001 rejects free-form keys precisely to make this loud."""
    with pytest.raises(CapabilityError, match="реестр"):
        Capability("arm.joint_positions")


def test_extension_domain_accepted_without_registry_entry() -> None:
    cap = Capability("x_unitree.dance_mode")
    assert cap.is_extension
    assert cap.domain == "x_unitree"
    assert cap.aspect == "dance_mode"
    assert cap not in CORE_CAPABILITIES


def test_core_key_is_not_an_extension() -> None:
    assert not Capability("arm.joint_position").is_extension


def test_matching_is_set_equality_without_inference() -> None:
    """Torque control does not imply position control (ADR-0001)."""
    required = parse_set(["arm.joint_position"])
    available = parse_set(["arm.joint_torque"])
    assert missing(required, available) == {Capability("arm.joint_position")}


def test_adapter_may_declare_both_itself() -> None:
    available = parse_set(["arm.joint_torque", "arm.joint_position"])
    assert missing(parse_set(["arm.joint_position"]), available) == frozenset()


def test_capability_is_a_string() -> None:
    """Profiles, YAML and sets keep working with plain strings."""
    assert Capability("arm.cartesian") == "arm.cartesian"
    assert sorted(parse_set(["gripper.binary", "arm.cartesian"])) == [
        "arm.cartesian",
        "gripper.binary",
    ]
