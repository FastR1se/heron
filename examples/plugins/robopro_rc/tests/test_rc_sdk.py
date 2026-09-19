"""The real channel, against a stand-in for the vendor's ``API`` package.

The stand-in is written here, not copied: it has the surface of ``RobotApi`` 1.4.1 the
channel touches, and does to the process what the source says the real one does --
``__init__`` installs its own ``sys.excepthook``. What these tests prove is what the
channel does with that surface (ADR-0053, ADR-0054, ADR-0062); that the real SDK
behaves like the stand-in is the stand's to show.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
from robopro_rc import sdk
from robopro_rc.adapter import RoboproRcAdapter
from robopro_rc.channel import ChannelError, LinkConfig, Waypoint, WaypointRefusedError
from robopro_rc.sdk import SdkChannel, copy_rtd, refuse_confirmation
from robopro_rc.testing import LINK_PARAMS, MODEL, profile

from heron.core.enums import MotionPhase

LINK = LinkConfig(host="192.0.2.10", model=MODEL)


@dataclass
class FakeInfo:
    """``RobotInfo``: its defaults are what 1.4.1 returns when the read fails."""

    robot_model: str = "rc10"
    client_version: str = "1.4.0.33555712/4"
    dh_model: tuple[float, ...] = (0.0,) * 6


class FakeRtd:
    """The live ``RTD`` object: a controller in ``run``, ``normal``, holding still."""

    def __init__(self) -> None:
        self.packet_begin = 0x12
        self.packet_end = 0x21
        self.sec = 1_726_000_000
        self.nsec = 0
        self.act_q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.act_qd = (0.0,) * 6
        self.act_trq = (1.0,) * 6
        self.act_flange_x = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
        self.state = 4.0  # doubles on the wire
        self.safety = 2.0
        self.motion_mode = 0.0
        self.buff_fill = 0
        self.wp_cntr = 7
        self.act_vel_scale = 1.0
        self.act_vel_limit_reason = 0
        self.state_flags = 0
        self.joint_state = (1,) * 6
        self.cycle_time = 0.004


@dataclass
class Sdk:
    """What the stand-in does, and what was done to it."""

    connects: bool | Exception = True
    info: FakeInfo = field(default_factory=lambda: FakeInfo(robot_model=MODEL))
    with_receiver: bool = True
    constructor_error: Exception | None = None
    waypoint_error: Exception | None = None
    waypoint_taken_anyway: bool = False
    mode_answer: bool | Exception = True
    scale_answer: Any = (1.0, 1.0)

    created: list[dict[str, Any]] = field(default_factory=list)
    confirm: list[Any] = field(default_factory=list)
    disconnects: int = 0
    calls: list[tuple[str, Any]] = field(default_factory=list)
    api: Any = None


def install(monkeypatch: pytest.MonkeyPatch, behaviour: Sdk) -> Sdk:
    """Put a module ``API.rc_api`` with a stand-in ``RobotApi`` on the import path."""

    class RobotApi:
        def __init__(self, ip: str, **kwargs: Any) -> None:
            sys.excepthook = self._exit_program  # what 1.4.1 does to the process
            behaviour.created.append({"ip": ip, **kwargs})
            behaviour.api = self
            if behaviour.constructor_error is not None:
                raise behaviour.constructor_error
            self.rtd = FakeRtd()
            self._connected = False
            self._on_lost: Any = None
            if behaviour.with_receiver:
                self._rtd_receiver = types.SimpleNamespace(get_data=lambda: self.rtd)
            self.controller_state = types.SimpleNamespace(
                set_confirm_position_callback=behaviour.confirm.append
            )
            self.motion = types.SimpleNamespace(
                scale_setup=types.SimpleNamespace(set=self._set_scale, get=self._get_scale),
                joint=types.SimpleNamespace(add_new_waypoint=self._add_waypoint),
                mode=types.SimpleNamespace(set=self._set_mode),
            )

        def _exit_program(self, *args: Any) -> None:  # pragma: no cover - never called
            pass

        def set_disconnection_callback(self, callback: Any) -> None:
            self._on_lost = callback

        def connect(self, read_only: bool = False) -> bool:
            behaviour.calls.append(("connect", read_only))
            if isinstance(behaviour.connects, Exception):
                raise behaviour.connects
            self._connected = behaviour.connects
            return behaviour.connects

        def is_connected(self) -> bool:
            return self._connected

        def get_robot_info(self) -> FakeInfo:
            return behaviour.info

        def disconnect(self) -> bool:
            behaviour.disconnects += 1
            self._connected = False
            return True

        def drop(self) -> None:
            """What the SDK does on a broken link: sockets closed, callback called."""
            self._connected = False
            self._on_lost()

        def _set_scale(self, velocity: float, acceleration: float) -> bool:
            behaviour.calls.append(("scale", (velocity, acceleration)))
            return True

        def _get_scale(self) -> Any:
            return behaviour.scale_answer

        def _add_waypoint(self, **kwargs: Any) -> bool:
            behaviour.calls.append(("waypoint", kwargs))
            if behaviour.waypoint_taken_anyway:
                self.rtd.wp_cntr += 1
            if behaviour.waypoint_error is not None:
                raise behaviour.waypoint_error
            self.rtd.wp_cntr += 1
            return True

        def _set_mode(self, mode: str) -> bool:
            behaviour.calls.append(("mode", mode))
            if isinstance(behaviour.mode_answer, Exception):
                raise behaviour.mode_answer
            return behaviour.mode_answer

    package = types.ModuleType("API")
    module = types.ModuleType("API.rc_api")
    module.RobotApi = RobotApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "API", package)
    monkeypatch.setitem(sys.modules, "API.rc_api", module)
    return behaviour


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The process's hook before the SDK, restored after the test whatever happens."""

    def ours(*args: Any) -> None:  # pragma: no cover - never called
        pass

    monkeypatch.setattr(sys, "excepthook", ours)
    return ours


def opened(monkeypatch: pytest.MonkeyPatch, **behaviour: Any) -> tuple[SdkChannel, Sdk]:
    fake = install(monkeypatch, Sdk(**behaviour))
    return SdkChannel.open(LINK), fake


# -- opening: the process stays the platform's (ADR-0053) -----------------------


def test_the_sdk_is_created_ignoring_its_own_reactions_without_connecting(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    _, fake = opened(monkeypatch)
    [created] = fake.created
    assert created["ip"] == "192.0.2.10"
    assert created["ignore_controller_exceptions"] is True
    assert created["autoconnect"] is False and created["read_only"] is False
    assert isinstance(created["logger"], logging.Logger)
    assert fake.calls[0] == ("connect", False)


def test_the_process_gets_its_exception_hook_back(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    opened(monkeypatch)
    assert sys.excepthook is hook


def test_the_hook_comes_back_even_when_the_sdk_fails_to_construct(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    install(monkeypatch, Sdk(constructor_error=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        SdkChannel.open(LINK)
    assert sys.excepthook is hook


def test_a_confirmation_of_position_is_always_refused(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    _, fake = opened(monkeypatch)
    assert fake.confirm == [refuse_confirmation]
    discrepancy = types.SimpleNamespace(
        joint_number=2, allowed_discrepancy=5.0, actual_position=31.0, saved_position=20.0
    )
    assert refuse_confirmation((discrepancy,)) is False


def test_the_identity_is_read_in_full_mode(monkeypatch: pytest.MonkeyPatch, hook: Any) -> None:
    channel, _ = opened(monkeypatch)
    assert channel.identity().model == MODEL


@pytest.mark.parametrize(
    "behaviour",
    [
        {"connects": False},
        {"connects": ConnectionRefusedError("29000")},
        {"info": FakeInfo()},  # the default: a read that failed and says "rc10"
        {"with_receiver": False},
    ],
)
def test_every_failure_after_construction_disconnects_the_sdk(
    monkeypatch: pytest.MonkeyPatch, hook: Any, behaviour: dict[str, Any]
) -> None:
    fake = install(monkeypatch, Sdk(**behaviour))
    with pytest.raises(ChannelError):
        SdkChannel.open(LINK)
    assert fake.disconnects == 1


def test_a_missing_sdk_is_named_with_its_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "API.rc_api", None)
    with pytest.raises(ChannelError, match=r"1\.4\.1"):
        SdkChannel.open(LINK)


# -- reading ----------------------------------------------------------------------


def test_one_reading_is_one_copy_in_wire_units(monkeypatch: pytest.MonkeyPatch, hook: Any) -> None:
    channel, _ = opened(monkeypatch)
    reading = channel.poll(0.0)
    assert reading is not None
    assert reading.q == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert (reading.controller_state, reading.safety, reading.motion_mode) == (4, 2, 0)
    assert isinstance(reading.controller_state, int)
    assert (reading.waypoint_counter, reading.stamp) == (7, 1_726_000_000.0)
    assert reading.cycle_time == 0.004


def test_the_same_packet_is_not_a_new_reading(monkeypatch: pytest.MonkeyPatch, hook: Any) -> None:
    channel, fake = opened(monkeypatch)
    assert channel.poll(0.0) is not None
    assert channel.poll(0.01) is None
    fake.api.rtd.nsec = 2_000_000
    assert channel.poll(0.0) is not None


def test_a_packet_read_out_of_step_is_dropped() -> None:
    data = FakeRtd()
    data.packet_end = 0x00
    with pytest.raises(ChannelError, match="сдвигом"):
        copy_rtd(data)


def test_a_misframed_packet_reads_as_nothing(monkeypatch: pytest.MonkeyPatch, hook: Any) -> None:
    channel, fake = opened(monkeypatch)
    fake.api.rtd.packet_begin = 0x33
    assert channel.poll(0.0) is None


def test_a_link_the_sdk_closed_reads_as_nothing_and_says_why(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    channel, fake = opened(monkeypatch)
    fake.api.drop()
    assert channel.poll(0.0) is None
    assert channel.lost() is not None and "SDK закрыл" in (channel.lost() or "")
    with pytest.raises(ChannelError, match="связь потеряна"):
        channel.hold()


# -- commanding ---------------------------------------------------------------------


def test_a_waypoint_states_every_argument_in_radians(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    channel, fake = opened(monkeypatch)
    channel.add_waypoint(Waypoint(target=(0.1,) * 6, speed=0.5, acceleration=0.7))
    [(_, kwargs)] = [call for call in fake.calls if call[0] == "waypoint"]
    assert kwargs == {
        "angle_pose": [0.1] * 6, "speed": 0.5, "accel": 0.7, "blend": 0.0, "units": "rad",
    }


def test_a_point_the_counter_did_not_take_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    channel, _ = opened(
        monkeypatch, waypoint_error=RuntimeError("Failed to add the waypoint (singularity)")
    )
    with pytest.raises(WaypointRefusedError, match="singularity"):
        channel.add_waypoint(Waypoint(target=(0.0,) * 6, speed=0.5, acceleration=0.5))


def test_a_point_the_counter_took_despite_an_error_is_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    channel, _ = opened(
        monkeypatch, waypoint_error=RuntimeError("timeout"), waypoint_taken_anyway=True
    )
    with pytest.raises(ChannelError) as raised:
        channel.add_waypoint(Waypoint(target=(0.0,) * 6, speed=0.5, acceleration=0.5))
    assert not isinstance(raised.value, WaypointRefusedError)


@pytest.mark.parametrize("answer", [False, TimeoutError("Motion mode did not change in 5 sec")])
def test_a_mode_not_confirmed_is_an_error(
    monkeypatch: pytest.MonkeyPatch, hook: Any, answer: bool | Exception
) -> None:
    channel, _ = opened(monkeypatch, mode_answer=answer)
    with pytest.raises(ChannelError):
        channel.start()
    with pytest.raises(ChannelError):
        channel.hold()


def test_a_scale_the_controller_did_not_answer_is_an_error(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    channel, _ = opened(monkeypatch, scale_answer=())
    with pytest.raises(ChannelError, match="не прочитана"):
        channel.scale()


def test_close_disconnects_the_sdk_once(monkeypatch: pytest.MonkeyPatch, hook: Any) -> None:
    channel, fake = opened(monkeypatch)
    channel.close()
    channel.close()
    assert fake.disconnects == 1


# -- the adapter over the real channel ----------------------------------------------------


def test_the_adapter_opens_the_sdk_on_connect_and_closes_it_on_disconnect(
    monkeypatch: pytest.MonkeyPatch, hook: Any
) -> None:
    fake = install(monkeypatch, Sdk(scale_answer=(0.25, 0.0625)))
    adapter = RoboproRcAdapter()
    adapter.connect(profile(), {**LINK_PARAMS, "speed_scale": 0.25})
    assert ("scale", (0.25, 0.0625)) in fake.calls
    assert adapter.read_state().motion is MotionPhase.IDLE
    adapter.disconnect()
    assert fake.disconnects == 1 and sys.excepthook is hook


def test_importing_the_package_imports_neither_the_sdk_nor_numpy() -> None:
    code = (
        "import sys; import robopro_rc, robopro_rc.adapter, robopro_rc.channel, "
        "robopro_rc.testing, robopro_rc.kinematics; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'API', 'numpy', 'scipy'}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert sdk.SDK_MODULE == "API.rc_api"
