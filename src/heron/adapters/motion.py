"""The motion a link of the command class plans for a goal (ADR-0049).

A controller of that class takes a pose and reaches it within its own bounds. The
bounds are the profile's -- ``limits.joint_velocity`` and ``limits.joint_acceleration``
-- times the run's speed scale ``s`` and its square, so that the motion takes ``1/s``
as long and keeps its shape (ADR-0052, ADR-0057). Each joint travels from rest to rest
along a trapezoid of its own; the motion is over when the slowest joint is.

Two links plan with it and differ only in who executes the plan: ``dummy`` moves its
numbers along it, ``ros2_control`` hands it to the trajectory controller as samples
(ADR-0060). One planner, so that the numbers a test predicts are the motion a
simulator is told to make.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Trapezoid:
    """One joint from rest to rest over ``distance``, within a speed and an acceleration.

    A triangle when the distance is too short to reach ``speed``.
    """

    distance: float
    speed: float
    acceleration: float

    @property
    def peak(self) -> float:
        return min(self.speed, math.sqrt(abs(self.distance) * self.acceleration))

    @property
    def ramp(self) -> float:
        return self.peak / self.acceleration

    @property
    def cruise(self) -> float:
        peak = self.peak
        return (abs(self.distance) - peak * self.ramp) / peak if peak > 0 else 0.0

    @property
    def duration(self) -> float:
        return 2 * self.ramp + self.cruise

    def at(self, t: float) -> tuple[float, float]:
        """Travelled distance and velocity, signed, ``t`` seconds after the start."""
        sign = math.copysign(1.0, self.distance)
        a, peak, ramp, total = self.acceleration, self.peak, self.ramp, self.duration
        if t <= 0 or total == 0:
            return 0.0, 0.0
        if t >= total:
            return self.distance, 0.0
        if t < ramp:
            return sign * a * t * t / 2, sign * a * t
        if t < ramp + self.cruise:
            return sign * (a * ramp * ramp / 2 + peak * (t - ramp)), sign * peak
        left = total - t
        return sign * (abs(self.distance) - a * left * left / 2), sign * a * left


@dataclass(frozen=True, slots=True)
class Sample:
    """The planned state of every joint at ``time`` seconds after the start."""

    time: float
    position: tuple[float, ...]
    velocity: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Motion:
    """The accepted motion of a command link."""

    started: float
    start: tuple[float, ...]
    joints: tuple[Trapezoid, ...]

    @property
    def duration(self) -> float:
        return max(joint.duration for joint in self.joints)

    @property
    def target(self) -> tuple[float, ...]:
        return tuple(
            begin + joint.distance for begin, joint in zip(self.start, self.joints, strict=True)
        )

    def over(self, time: float) -> bool:
        """Reported over by the reading after its end, never by the one it was accepted at."""
        return time - self.started > self.duration

    def at(self, time: float) -> tuple[list[float], list[float]]:
        position, velocity = [], []
        for begin, joint in zip(self.start, self.joints, strict=True):
            travelled, speed = joint.at(time - self.started)
            position.append(begin + travelled)
            velocity.append(speed)
        return position, velocity

    def samples(self, period: float) -> list[Sample]:
        """The plan every ``period`` seconds after the start, ending on the target at rest.

        Never empty: a motion to where the robot already is still ends one period
        later, so that a controller told to execute it has a point to execute.
        """
        if not period > 0:
            raise ValueError(f"период выборки {period:g} с: нужен > 0")
        end = max(self.duration, period)
        count = max(1, math.ceil(end / period - 1e-9))
        times = [min(step * period, end) for step in range(1, count)] + [end]
        planned = []
        for time in times:
            position, velocity = self.at(self.started + time)
            planned.append(Sample(time=time, position=tuple(position), velocity=tuple(velocity)))
        return planned


def plan(
    started: float,
    start: Sequence[float],
    target: Sequence[float],
    speeds: Sequence[float],
    accelerations: Sequence[float],
    scale: float,
) -> Motion:
    """Each joint along its own trapezoid, bounded by ``s·v`` and ``s²·a``.

    Velocity times s and acceleration times s squared: the same motion in 1/s the
    time. Acceleration times s would stretch a ramp by 1/sqrt(s) (ADR-0057).
    """
    return Motion(
        started=started,
        start=tuple(float(value) for value in start),
        joints=tuple(
            Trapezoid(float(goal) - float(begin), speed * scale, acceleration * scale**2)
            for goal, begin, speed, acceleration in zip(
                target, start, speeds, accelerations, strict=True
            )
        ),
    )
