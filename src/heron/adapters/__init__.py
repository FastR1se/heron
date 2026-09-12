"""Adapters shipped with the core: nothing vendor-specific lives here (section 10)."""

from .dummy import DummyAdapter, NotConnectedError, SafetyViolationError

__all__ = ["DummyAdapter", "NotConnectedError", "SafetyViolationError"]
