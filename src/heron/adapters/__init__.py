"""Adapters shipped with the core: nothing vendor-specific lives here (section 10).

``ros2_control`` is general in the same sense ``dummy`` is: it is the standard
path of the ROS 2 control stack and belongs to no vendor. It imports nothing
from a ROS 2 distribution until it is asked to connect (ADR-0014), so importing
this package on a machine without ROS 2 stays free.
"""

from .dummy import DummyAdapter, NotConnectedError, SafetyViolationError
from .ros2_control import Ros2ControlAdapter, StateMismatchError, UnsupportedActionError

__all__ = [
    "DummyAdapter",
    "NotConnectedError",
    "Ros2ControlAdapter",
    "SafetyViolationError",
    "StateMismatchError",
    "UnsupportedActionError",
]
