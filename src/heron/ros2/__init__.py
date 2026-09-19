"""What the ROS 2 control plane is called, and how to talk to it.

Two modules, and the split between them is the point: ``names`` is a vocabulary
of strings derived from a profile, and ``transport`` is the only place allowed
to import a ROS 2 distribution. Everything that decides *what* is published is
therefore checkable on a machine with no ROS 2 on it, which is where this
platform is written (ADR-0014).
"""

from .names import (
    ADAPTER_PACKAGES,
    BROADCASTER,
    command_action,
    command_message,
    command_topic,
    controller_name,
    state_topic,
)
from .transport import Publication, Reading, RecordingTransport, Transport

__all__ = [
    "ADAPTER_PACKAGES",
    "BROADCASTER",
    "Publication",
    "Reading",
    "RecordingTransport",
    "Transport",
    "command_action",
    "command_message",
    "command_topic",
    "controller_name",
    "state_topic",
]
