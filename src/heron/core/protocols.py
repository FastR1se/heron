"""The six extension points of the platform (ARCHITECTURE.md, section 5).

Everything vendor-specific implements one of these from outside the core. A new
extension point is a decision, recorded in ``docs/decisions/``, not an edit to
this file. The fifth -- the twin -- is ADR-0021, the sixth -- the training task --
is ADR-0044.

They are ``Protocol`` classes rather than base classes on purpose: a vendor
package must not have to import the platform to satisfy them, and invariant 1
("no robot names in the core") is easier to hold when the core owns no
inheritance chain reaching into robot code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .capabilities import Capability
from .enums import ControlMode, Status, TwinKind
from .environment import Requirement
from .models import Completion, Limits, RobotProfile, Scenario, SkillSpec
from .types import Command, Pose, RobotState


class CommandRefusedError(RuntimeError):
    """An adapter declined a command, and nothing of it reached the robot (ADR-0026).

    The one failure of ``send_command`` the protocol names, because the executor
    must tell it from a defect: a refusal is the adapter doing its job -- a
    position outside the envelope, a command after an emergency stop -- and ends
    the skill with a reason. Anything else raised there is a bug, and stays one.
    """


@runtime_checkable
class ControlAdapter(Protocol):
    """Link to a concrete robot or to its twin in a simulator.

    The same container image talks to simulation and to hardware; only the
    adapter configuration differs between them (invariant 3).
    """

    requirements: Sequence[Requirement]
    """What the process driving the robot needs from its environment (ADR-0025).

    Read off the class before anything is constructed or connected, and declared
    even when empty: a side that says nothing has not been asked."""

    supports_speed_scale: bool
    """Whether this link executes the run's speed scale (ADR-0052). Read off the class,
    with no default: an adapter that never heard of the scale must not be handed one
    and ignore it on the first run on hardware."""

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def send_command(self, cmd: Command) -> None:
        """Send one command, or raise ``CommandRefusedError`` and send nothing.

        A ``command`` link also refuses a command while a motion is ``moving`` or
        ``halted``, and reports the accepted one as not ``idle`` until the robot
        says it is over (ADR-0049)."""

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

    requirements: Sequence[Requirement]
    """What the process starting the world needs for this twin (ADR-0025). The
    adapter declares its own: the two sides run in different processes."""


@runtime_checkable
class TrainBackend(Protocol):
    """A training environment.

    Its only output crossing the boundary is an artifact -- ONNX plus contract
    (invariant 2). Nothing here returns a live object to the execution side, and
    nothing here is a live object either: a plan is data, a run is a directory,
    and an artifact is three files (ADR-0044).
    """

    requirements: Sequence[Requirement]
    """What the process that trains needs from its environment (ADR-0025)."""

    def build_env(self, scenario: Scenario, learner: str) -> Any:
        """Derive the environment from the profile and the scenario. No simulator.

        Everything a training environment is -- rates, action, observation, drive
        -- is already stated by the contract, so this is a derivation and not a
        configuration: it produces data, and refuses a scenario it cannot derive
        from before anything is started.
        """

    def train(self, plan: Any, conditions: Any) -> Any:
        """Run the training the plan describes. Returns the run's directory."""

    def export(self, run: Any) -> Any:
        """Turn a finished run into an artifact of the store (ADR-0046)."""


@runtime_checkable
class TrainTask(Protocol):
    """What is being learned: the sixth extension point (ADR-0044).

    The part every task shares, whatever backend trains it. What a task
    contributes to a concrete training environment -- rewards, terminations,
    events, the range a goal is drawn from -- is phrased in the vocabulary of its
    backend and declared next to that backend, as a twin's contribution is
    (ADR-0024).

    A task says what is learned and when it is done. It does not say how the data
    is represented: the action, the observation, the drive and the rates come
    from the contract, and this protocol has no member through which they could
    be changed.
    """

    backend: str
    """The ``world.backend`` this task is trained in."""

    spec: SkillSpec
    """What the future skill demands of a robot -- capabilities, control mode,
    quantities. Travels into the manifest of the artifact (ADR-0046) and is what
    the validator compares with a profile."""

    completion: Completion
    """When the trained skill has reached its goal, and how long it is given
    (ADR-0043). The same tolerance the task scores success by in training, so
    that it is one quantity in both worlds; ``horizon_s`` is the length of an
    episode -- past it the policy was never trained."""
