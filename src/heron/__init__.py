"""Modelling platform for heterogeneous robotic systems.

The package name is the one thing that spells the platform name outside
``core.branding``, and section 17 accepts that: renaming is a directory rename
plus two files. Inside the package every import is relative for the same reason.
"""

from .core.branding import APP_TITLE

# Between releases main carries the next version as a development one: a package that needs
# what main added pins ">=<next>.dev0" and resolves against this tree (ADR-0071).
__version__ = "0.5.0.dev0"
__all__ = ["APP_TITLE", "__version__"]
