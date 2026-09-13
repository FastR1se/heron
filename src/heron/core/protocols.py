"""The five extension points of the platform (ARCHITECTURE.md, section 5).

Everything vendor-specific implements one of these from outside the core. A new
extension point is a decision, recorded in ``docs/decisions/``, not an edit to
this file. The fifth -- the twin -- is ADR-0021.

They are ``Protocol`` classes rather than base classes on purpose: a vendor
package must not have to import the platform to satisfy them, and invariant 1
("no robot names in the core") is easier to hold when the core owns no
inheritance chain reaching into robot code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .capabilities import Capability
from .enums import ControlMode, Status, TwinKind
from .models import Limits, RobotProfile, Scenario
from .types import Command, Pose, RobotState


@runtime_checkable
class ControlAdapter(Protocol):
    """Link to a concrete robot or to its twin in a simulator.

    The same container image talks to simulation and to hardware; only the
    adapter configuration differs between them (invariant 3).
    """

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def send_command(self, cmd: Command) -> None: ...

    def capabilities(self) -> frozenset[Capability]:
        """What this link can actually do.

        Declared by the adapter, not inferred by the core: whether a
        torque-driven controller also accepts positions is knowledge that lives
        with the controller (ADR-0001).
        """

    def control_mode(self) -> ControlMode: ...

    def safety_limits(self) -> Limits: ...

    def emergency_stop(self) -> None:
        """Mandatory (section 12). An adapter without it is not deployable."""


@runtime_checkable
class SkillContext(Protocol):
    """What a running skill is allowed to touch."""

    @property
    def profile(self) -> RobotProfile: ...

    def send(self, cmd: Command) -> None: ...

    def now(self) -> float: ...

    def log(self, message: str, **fields: Any) -> None: ...


@runtime_checkable
class Skill(Protocol):
    """A unit of behaviour. Code and policy are indistinguishable from outside.

    Invariant 4: a mission never learns how a skill is made, so swapping a coded
    skill for a trained one is one line in the profile.
    """

    required_capabilities: frozenset[Capability]
    required_control_mode: ControlMode

    def start(self, goal: dict[str, Any], ctx: SkillContext) -> None: ...

    def tick(self, state: RobotState) -> Status: ...

    def cancel(self) -> None: ...


@runtime_checkable
class WorldHandle(Protocol):
    """Opaque handle to a built world."""

    def close(self) -> None: ...


@runtime_checkable
class RobotHandle(Protocol):
    """Opaque handle to a robot spawned in a world."""

    @property
    def id(self) -> str: ...


@runtime_checkable
class SimBackend(Protocol):
    """A simulation environment. One scenario has exactly one (ADR-0002)."""

    def build_world(self, scenario: Scenario) -> WorldHandle: ...

    def spawn(self, profile: RobotProfile, pose: Pose) -> RobotHandle: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class Twin(Protocol):
    """The other side of a link in simulation (ADR-0021, ADR-0024).

    An adapter is one side of a link. On hardware the robot is the other; in a
    simulator something has to speak the robot's protocol in its place, and only
    whoever wrote the adapter knows that protocol. So a robot package ships the
    pair, and a twin names the adapter and the backend it answers for.

    This is the part every twin shares. What a twin contributes to a scene is
    phrased in the vocabulary of its backend -- a Gazebo twin adds interfaces to
    a description and nodes to a launch, a simulator of its own is a process --
    and each backend declares that protocol next to itself.
    """

    adapter: str
    """Plugin name of the adapter, exactly as a profile writes it."""

    backend: str
    """The ``world.backend`` this twin lives in."""

    kind: TwinKind


@runtime_checkable
class TrainBackend(Protocol):
    """A training environment.

    Its only output crossing the boundary is an artifact -- ONNX plus contract
    (invariant 2). Nothing here returns a live object to the execution side.
    """

    def build_env(self, scenario: Scenario, learner: str) -> Any: ...

    def train(self, cfg: dict[str, Any]) -> Any: ...

    def export(self, ckpt: Any) -> Any: ...
