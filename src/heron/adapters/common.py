"""What the built-in adapters share about refusing a command."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import RobotProfile


def outside_bounds(
    profile: RobotProfile,
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> str:
    """Joints of a command outside the envelope, named with value and bounds.

    Named, not indexed: the first humanoid refusal said "joint [20]", and which
    joint that was, and how far outside, had to be reconstructed from the
    contract by hand. Empty when every value is inside.
    """
    return "; ".join(
        f"{profile.contract.joint_order[i]} = {value:.4g} при [{lower[i]:.4g}, {upper[i]:.4g}]"
        for i, value in enumerate(values)
        if not (lower[i] <= value <= upper[i])
    )
