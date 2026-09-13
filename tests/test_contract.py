"""Contract identity: ADR-0004.

The golden test is the important one. ``contract_id`` is stored inside every
policy and trajectory the platform ever exports, so a change in the canonical
serialization -- key order, number formatting, escaping -- silently invalidates
all of them. Pinning the value here means such a change fails in CI instead of
in an experiment six months later.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import make_contract
from heron.core.contract import NORMATIVE_FIELDS, Contract, compatible, diff
from heron.core.enums import ActionType, AngleUnit, ObservationType

GOLDEN_ID = "236ab854cadef20e"
"""Identifier of the reference contract below. Changing it is a breaking change."""


def reference() -> Contract:
    return Contract(
        revision=1,
        joint_order=["joint0", "joint1", "joint2"],
        angle_unit=AngleUnit.RAD,
        state_rate_hz=100.0,
        command_rate_hz=50.0,
        observation=[
            {"type": ObservationType.JOINT_POS, "scale": 1.0},
            {"type": ObservationType.JOINT_VEL, "scale": 0.05},
        ],
        action={"type": ActionType.JOINT_POSITION, "clip": (-1.0, 1.0)},
    )


def variant(**overrides: Any) -> Contract:
    """The reference contract with fields replaced, revalidated.

    ``model_copy`` would skip validation and let a raw dict reach the serializer
    -- the kind of shortcut that lets a golden test pass while the real path is
    broken.
    """
    data = reference().model_dump(mode="json")
    data.update(overrides)
    return Contract.model_validate(data)


def test_golden_identifier() -> None:
    assert reference().contract_id == GOLDEN_ID


def test_canonical_form_is_pinned() -> None:
    """Keys sorted, no spaces, ASCII. Every stored identifier depends on it."""
    text = reference().canonical_json()
    assert text == (
        '{"action":{"clip":[-1.0,1.0],"type":"joint_position"},'
        '"angle_unit":"rad","command_rate_hz":50.0,'
        '"joint_order":["joint0","joint1","joint2"],'
        '"observation":[{"scale":1.0,"type":"joint_pos"},'
        '{"scale":0.05,"type":"joint_vel"}],"state_rate_hz":100.0}'
    )
    assert json.loads(text)  # still valid JSON


def test_revision_is_outside_the_hash() -> None:
    """Declaring a change must not be what defines the change (ADR-0004)."""
    assert variant(revision=7).contract_id == reference().contract_id


def test_limits_are_outside_the_contract() -> None:
    """Limits bound the data, they do not change its meaning."""
    assert "limits" not in NORMATIVE_FIELDS


def test_joint_reordering_changes_identity() -> None:
    """The classic silent failure: same joints, different order."""
    a = reference()
    b = variant(joint_order=["joint1", "joint0", "joint2"])
    assert not compatible(a, b)
    assert [d.field for d in diff(a, b)] == ["joint_order"]


def test_scale_change_changes_identity() -> None:
    a = reference()
    b = variant(
        observation=[
            {"type": "joint_pos", "scale": 1.0},
            {"type": "joint_vel", "scale": 0.5},
        ]
    )
    assert not compatible(a, b)
    assert [d.field for d in diff(a, b)] == ["observation"]


def test_state_rate_change_invalidates_policies() -> None:
    """Deliberately conservative: rate changes the meaning of observations."""
    a = reference()
    b = variant(state_rate_hz=200.0)
    assert not compatible(a, b)
    assert [d.field for d in diff(a, b)] == ["state_rate_hz"]


def test_integer_and_float_rates_agree() -> None:
    """``100`` and ``100.0`` in YAML must not produce different identifiers."""
    assert make_contract(state_rate_hz=100).contract_id == (
        make_contract(state_rate_hz=100.0).contract_id
    )


def test_diff_reports_fields_not_hashes() -> None:
    a = reference()
    b = variant(angle_unit=AngleUnit.DEG, command_rate_hz=10.0)
    assert {d.field for d in diff(a, b)} == {"angle_unit", "command_rate_hz"}
    assert "rad" in diff(a, b)[0].describe()


def test_duplicate_joints_rejected() -> None:
    with pytest.raises(ValueError, match="повтор"):
        make_contract(joint_order=["j0", "j1", "j0"])


def test_zero_scale_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        make_contract(observation=[{"type": ObservationType.JOINT_POS, "scale": 0.0}])


def test_reversed_clip_rejected() -> None:
    with pytest.raises(ValueError, match="clip"):
        make_contract(action={"type": ActionType.JOINT_POSITION, "clip": (1.0, -1.0)})


# -- gains (ADR-0019) --------------------------------------------------------


def _gains(dof: int = 3, kp: float = 60.0, kd: float = 1.0) -> dict[str, Any]:
    return {
        "type": ActionType.JOINT_POSITION,
        "clip": (-1.0, 1.0),
        "kp": [kp] * dof,
        "kd": [kd] * dof,
    }


def test_unstated_gains_leave_every_identifier_as_it_was() -> None:
    """Gains arrived after identifiers were stored. Absent, they are absent
    from the canonical form too -- the golden identifier above proves it."""
    assert "kp" not in reference().canonical_json()
    assert reference().contract_id == GOLDEN_ID


def test_gains_change_identity() -> None:
    """A policy trained against one stiffness is not the same policy against
    another: the difference shows up in the action, by field."""
    a = reference()
    b = variant(action=_gains())
    c = variant(action=_gains(kp=100.0))
    assert not compatible(a, b)
    assert not compatible(b, c)
    assert [d.field for d in diff(b, c)] == ["action"]


def test_gains_are_ordered_like_the_joints() -> None:
    with pytest.raises(ValueError, match="kp"):
        make_contract(3, action=_gains(dof=2))


def test_gains_come_in_pairs() -> None:
    action = _gains()
    del action["kd"]
    with pytest.raises(ValueError, match="вместе"):
        make_contract(3, action=action)


def test_gains_need_a_position_target() -> None:
    action = {**_gains(), "type": ActionType.JOINT_VELOCITY}
    with pytest.raises(ValueError, match="положению"):
        make_contract(3, action=action)


def test_gains_must_hold_and_damp() -> None:
    with pytest.raises(ValueError, match="kp"):
        make_contract(3, action=_gains(kp=0.0))
    with pytest.raises(ValueError, match="kd"):
        make_contract(3, action=_gains(kd=-1.0))


def test_gains_refuse_a_contract_in_degrees() -> None:
    """A drive takes N*m/rad; the same number per degree would mean another law."""
    with pytest.raises(ValueError, match="rad"):
        make_contract(3, angle_unit=AngleUnit.DEG, action=_gains())


def test_contract_is_frozen() -> None:
    """An identifier computed from a mutable object is a broken promise."""
    with pytest.raises(ValueError):
        reference().revision = 2  # type: ignore[misc]
