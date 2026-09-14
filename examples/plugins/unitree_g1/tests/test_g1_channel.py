"""The channel's values and configuration, without CycloneDDS (ADR-0029).

What these tests cannot say: that CycloneDDS accepts the configuration, that the
SDK's types take the values, or that a sample arrives. The first run of the real
channel is stage 4 of the phase-4 plan.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import pytest
from unitree_g1.channel import (
    ChannelError,
    LinkConfig,
    LinkConfigError,
    LowCommand,
    LowStateReading,
    MotorCommand,
    RecordingChannel,
)
from unitree_g1.cyclone import CycloneChannel
from unitree_g1.slots import SLOT_COUNT

PROVEN_PEERS_LO = (
    '<CycloneDDS><Domain Id="any"><General>'
    '<Interfaces><NetworkInterface name="lo"/></Interfaces>'
    "<AllowMulticast>false</AllowMulticast></General>"
    "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
    '<Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
    "</Domain></CycloneDDS>"
)
"""The configuration that exchanged LowCmd and LowState with the vendor's twin,
copied from scripts/phase4-stage1.sh (journal 2026-09-13, session 9)."""


def canonical(xml: str) -> str:
    return ET.canonicalize(xml_data=xml, strip_text=True)


# -- link configuration ------------------------------------------------------


def test_params_describe_a_link() -> None:
    link = LinkConfig.from_params({"domain": 1, "interface": "lo", "peers": ["127.0.0.1"]})
    assert link == LinkConfig(domain=1, interface="lo", peers=("127.0.0.1",))


def test_peers_are_optional() -> None:
    assert LinkConfig.from_params({"domain": 0, "interface": "eth0"}).peers == ()


@pytest.mark.parametrize(
    ("params", "key"),
    [
        ({"interface": "lo"}, "params.domain"),
        ({"domain": True, "interface": "lo"}, "params.domain"),
        ({"domain": -1, "interface": "lo"}, "params.domain"),
        ({"domain": "1", "interface": "lo"}, "params.domain"),
        ({"domain": 1}, "params.interface"),
        ({"domain": 1, "interface": " "}, "params.interface"),
        ({"domain": 1, "interface": "lo", "peers": "127.0.0.1"}, "params.peers"),
        ({"domain": 1, "interface": "lo", "peers": ["127.0.0.1", ""]}, "params.peers"),
    ],
)
def test_a_wrong_param_is_named(params: dict[str, object], key: str) -> None:
    with pytest.raises(LinkConfigError, match=key):
        LinkConfig.from_params(params)


def test_with_peers_the_configuration_is_the_one_proven_against_the_twin() -> None:
    link = LinkConfig(domain=1, interface="lo", peers=("127.0.0.1",))
    assert canonical(link.cyclonedds_xml()) == canonical(PROVEN_PEERS_LO)


def test_without_peers_discovery_is_left_to_multicast() -> None:
    root = ET.fromstring(LinkConfig(domain=0, interface="eth0").cyclonedds_xml())
    assert [item.attrib["name"] for item in root.iter("NetworkInterface")] == ["eth0"]
    assert root.find(".//Peers") is None
    assert root.find(".//AllowMulticast") is None


def test_an_interface_name_cannot_close_a_tag() -> None:
    hostile = 'lo"/></Interfaces><Tracing'
    root = ET.fromstring(LinkConfig(domain=1, interface=hostile).cyclonedds_xml())
    assert [item.attrib["name"] for item in root.iter("NetworkInterface")] == [hostile]
    assert root.find(".//Tracing") is None


# -- values -------------------------------------------------------------------


def test_a_command_states_every_slot() -> None:
    with pytest.raises(ValueError, match=str(SLOT_COUNT)):
        LowCommand(mode_pr=0, mode_machine=4, motors=(MotorCommand(),) * 29)


def test_a_reading_states_every_slot() -> None:
    with pytest.raises(ValueError, match="dq"):
        LowStateReading(
            tick=0, mode_machine=4, mode_pr=0,
            q=(0.0,) * SLOT_COUNT, dq=(0.0,) * 23, tau_est=(0.0,) * SLOT_COUNT,
        )


def test_an_empty_slot_is_off_and_pulls_nowhere() -> None:
    assert MotorCommand() == MotorCommand(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)


# -- the recording channel ------------------------------------------------------


def test_the_recording_channel_answers_in_order_and_then_nothing() -> None:
    channel = RecordingChannel()
    channel.deliver_state(10, mode_machine=4, q=[0.5])
    channel.deliver_state(12, mode_machine=4)
    first, second = channel.poll(0.1), channel.poll(0.1)
    assert first is not None and second is not None
    assert (first.tick, first.q[0], first.q[1]) == (10, 0.5, 0.0)
    assert second.tick == 12
    assert channel.poll(0.1) is None
    assert channel.publisher_count() == 1


def test_a_closed_channel_publishes_nothing() -> None:
    channel = RecordingChannel()
    channel.close()
    with pytest.raises(ChannelError):
        channel.publish(
            LowCommand(mode_pr=0, mode_machine=4, motors=(MotorCommand(),) * SLOT_COUNT)
        )


# -- the real channel, as far as it can be reached here -------------------------


def test_importing_the_adapter_imports_no_dds() -> None:
    """The platform lists and checks a robot on machines without its SDK."""
    code = (
        "import sys; import unitree_g1, unitree_g1.adapter, unitree_g1.testing; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'cyclonedds', 'unitree_sdk2py', 'rclpy'}); "
        "print('leaked:', leaked)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "leaked: []" in result.stdout, result.stdout


def test_the_real_channel_refuses_a_process_with_rclpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked before CycloneDDS is imported: the refusal does not depend on having it."""
    monkeypatch.setitem(sys.modules, "rclpy", types.ModuleType("rclpy"))
    with pytest.raises(ChannelError, match="rclpy"):
        CycloneChannel(LinkConfig(domain=1, interface="lo"))


@pytest.mark.skipif(
    importlib.util.find_spec("cyclonedds") is not None,
    reason="cyclonedds установлен: отсутствие не проверить",
)
def test_without_cyclonedds_the_real_channel_says_what_is_missing() -> None:
    with pytest.raises(ChannelError, match="cyclonedds"):
        CycloneChannel(LinkConfig(domain=1, interface="lo"))
