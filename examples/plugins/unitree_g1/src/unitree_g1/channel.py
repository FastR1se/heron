"""The link to a G1, as values (ADR-0029).

The adapter decides everything that goes on the wire -- slot, gains, value --
and hands a channel a ``LowCommand``; the channel reads ``LowState`` and hands
back a ``LowStateReading``. Neither is an object of the vendor's SDK, so every
decision of the adapter is checkable on a machine that has neither CycloneDDS
nor the SDK. What cannot be checked there is conversion, and it lives alone in
``cyclone``.

Two channels ship: the real one, and one that writes commands down and answers
with readings it was given.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from .slots import SLOT_COUNT

COMMAND_TOPIC: Final = "rt/lowcmd"
STATE_TOPIC: Final = "rt/lowstate"
"""DDS topic names the robot and the vendor's twin use. ROS 2 sees them as
``/lowcmd`` and ``/lowstate`` (journal 2026-09-13, session 5, fact 0.7)."""

ROS2_TOPIC_PREFIX: Final = "rt/"
"""What ``rmw_cyclonedds_cpp`` puts in front of a ROS 2 topic on the wire."""

QUEUE_DEPTH: Final = 64
"""Most samples one read takes to reach the newest."""

MAX_PARTICIPANT_INDEX: Final = 32
"""Highest participant index CycloneDDS picks, and probes a peer for, under unicast discovery.

The default of 0.10 is 9. A Gazebo scene with the bridge is about ten participants on
one machine -- launch, simulator, clock bridge, description publisher, spawners,
visualization, the adapter, a witness -- and the tenth would find no index, while a
peer is only probed at the ports of indices up to the maximum (ADR-0034, p. 4)."""


def ros2_topic(dds_topic: str) -> str:
    """The ROS 2 name of a DDS topic of the robot: ``rt/lowcmd`` is ``/lowcmd``.

    A bridge speaking ROS 2 subscribes under this name and meets the adapter, which
    speaks DDS directly, on the same topic (fact 0.7).
    """
    if not dds_topic.startswith(ROS2_TOPIC_PREFIX):
        raise ValueError(
            f"{dds_topic!r}: не топик ROS 2 на проводе — нет префикса {ROS2_TOPIC_PREFIX!r}"
        )
    return "/" + dds_topic.removeprefix(ROS2_TOPIC_PREFIX)


class ChannelError(RuntimeError):
    """The link cannot be opened or used."""


class LinkConfigError(ChannelError):
    """The adapter's ``params`` do not describe a link."""


@dataclass(frozen=True, slots=True)
class MotorCommand:
    """One slot of ``LowCmd``. The defaults are an empty slot: off, no gains."""

    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True, slots=True)
class LowCommand:
    """One ``LowCmd``, every slot stated. The CRC is the channel's to compute."""

    mode_pr: int
    mode_machine: int
    motors: tuple[MotorCommand, ...]

    def __post_init__(self) -> None:
        if len(self.motors) != SLOT_COUNT:
            raise ValueError(f"LowCmd несёт {SLOT_COUNT} слотов, передано {len(self.motors)}")


@dataclass(frozen=True, slots=True)
class LowStateReading:
    """The part of one ``LowState`` the adapter reads, per slot."""

    tick: int
    """Milliseconds of world time in the vendor's twin (ADR-0023)."""

    mode_machine: int
    mode_pr: int
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau_est: tuple[float, ...]
    imu_quaternion: tuple[float, float, float, float]
    """``imu_state.quaternion`` of the pelvis IMU, ``w, x, y, z`` (ADR-0076)."""

    imu_gyroscope: tuple[float, float, float]
    """``imu_state.gyroscope``, rad/s in the IMU's frame, whose axes are the pelvis's."""

    def __post_init__(self) -> None:
        for name in ("q", "dq", "tau_est"):
            width = len(getattr(self, name))
            if width != SLOT_COUNT:
                raise ValueError(f"LowState.{name}: {width} значений при {SLOT_COUNT} слотах")


@runtime_checkable
class Channel(Protocol):
    """What the adapter may do with a link."""

    def publish(self, command: LowCommand) -> None: ...

    def poll(self, timeout: float) -> LowStateReading | None:
        """The newest state that arrived, waiting **up to** ``timeout`` for one.

        Blocking is part of the contract: the adapter waits for a link to come up
        against a deadline, and a channel answering at once turns that into a
        busy loop (ADR-0016).
        """

    def publisher_count(self) -> int:
        """Publishers of ``rt/lowstate`` this link is matched with.

        What tells "nobody publishes" from "somebody publishes and nothing
        arrives" -- two failures with different fixes (ADR-0016).
        """

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LinkConfig:
    """Domain, interface and peers: all that tells a twin from the robot (ADR-0022, p. 5)."""

    domain: int
    interface: str
    peers: tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> LinkConfig:
        """Read the link from an adapter's ``params``, naming any key that is wrong.

        Domain and interface have no defaults. The vendor's twin lives in domain
        1 and the robot in 0, and a default would pick one of them silently: the
        adapter would then wait in an empty domain and report that nobody
        publishes, which is true and misleading.
        """
        example = "например {domain: 1, interface: lo, peers: [127.0.0.1]}"
        domain = params.get("domain")
        if isinstance(domain, bool) or not isinstance(domain, int) or domain < 0:
            raise LinkConfigError(
                f"params.domain: нужен неотрицательный целый номер домена DDS, получено "
                f"{domain!r}; {example}"
            )
        interface = params.get("interface")
        if not isinstance(interface, str) or not interface.strip():
            raise LinkConfigError(
                f"params.interface: нужно имя сетевого интерфейса, получено {interface!r}; "
                f"{example}"
            )
        peers = params.get("peers", [])
        if isinstance(peers, str) or not isinstance(peers, Sequence):
            raise LinkConfigError(f"params.peers: нужен список адресов, получено {peers!r}")
        wrong = [peer for peer in peers if not isinstance(peer, str) or not peer.strip()]
        if wrong:
            raise LinkConfigError(f"params.peers: не адреса: {wrong!r}")
        return cls(domain=domain, interface=interface, peers=tuple(peers))

    def cyclonedds_xml(self) -> str:
        """The CycloneDDS configuration of this link.

        Without peers: the interface and nothing else, discovery by multicast,
        as the vendor's SDK configures the robot. With peers: multicast off and
        unicast discovery -- the form proven against the vendor's twin on ``lo``
        (journal 2026-09-13, session 9) -- with room for the participants of a
        whole scene (``MAX_PARTICIPANT_INDEX``). Built as a tree, so an interface
        name cannot close a tag.

        The bridge's side of a Gazebo scene runs on this same text (ADR-0034), so
        the two sides of a local link cannot disagree about it.
        """
        root = ET.Element("CycloneDDS")
        domain = ET.SubElement(root, "Domain", Id="any")
        general = ET.SubElement(domain, "General")
        interfaces = ET.SubElement(general, "Interfaces")
        ET.SubElement(interfaces, "NetworkInterface", name=self.interface)
        if self.peers:
            ET.SubElement(general, "AllowMulticast").text = "false"
            discovery = ET.SubElement(domain, "Discovery")
            ET.SubElement(discovery, "ParticipantIndex").text = "auto"
            ET.SubElement(discovery, "MaxAutoParticipantIndex").text = str(MAX_PARTICIPANT_INDEX)
            peers = ET.SubElement(discovery, "Peers")
            for address in self.peers:
                ET.SubElement(peers, "Peer", address=address)
        return ET.tostring(root, encoding="unicode")


@dataclass(slots=True)
class RecordingChannel:
    """Writes commands down and answers from readings it was given.

    Not a simulator: it moves nothing. One reading per poll, in the order given
    -- the real channel returns the newest of those that arrived, and a test
    that wants that says so by giving one. It cannot block either: a caller
    waiting against a deadline shortens the deadline in a test.
    """

    published: list[LowCommand] = field(default_factory=list)
    readings: deque[LowStateReading] = field(default_factory=deque)
    publishers: int = 0
    """Publishers this link pretends to be matched with. Set by ``deliver``, and
    settable by hand for a publisher whose messages never arrive."""

    closed: bool = False

    def publish(self, command: LowCommand) -> None:
        if self.closed:
            raise ChannelError("канал закрыт")
        self.published.append(command)

    def poll(self, timeout: float) -> LowStateReading | None:
        return self.readings.popleft() if self.readings else None

    def publisher_count(self) -> int:
        return self.publishers

    def close(self) -> None:
        self.closed = True

    # -- feeding it -----------------------------------------------------------

    def deliver(self, reading: LowStateReading) -> None:
        self.readings.append(reading)
        self.publishers = max(self.publishers, 1)

    def deliver_state(
        self,
        tick: int,
        *,
        mode_machine: int,
        q: Sequence[float] = (),
        dq: Sequence[float] = (),
        tau_est: Sequence[float] = (),
        mode_pr: int = 0,
        imu_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        imu_gyroscope: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Queue one state; slots nothing is said about read zero, as empty slots do,
        and an IMU nothing is said about reads upright and still."""

        def padded(values: Sequence[float]) -> tuple[float, ...]:
            return (*map(float, values), *(0.0,) * (SLOT_COUNT - len(values)))

        self.deliver(
            LowStateReading(
                tick=tick,
                mode_machine=mode_machine,
                mode_pr=mode_pr,
                q=padded(q),
                dq=padded(dq),
                tau_est=padded(tau_est),
                imu_quaternion=imu_quaternion,
                imu_gyroscope=imu_gyroscope,
            )
        )
