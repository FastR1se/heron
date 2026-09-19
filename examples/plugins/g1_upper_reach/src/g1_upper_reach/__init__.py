"""Training task for the mounted G1: its upper body to a joint pose (ADR-P6A, ADR-0043).

The second implementation of the pipeline of phase 6 (ADR-0040), built the way a user
builds one for their own robot: a package, an entry point, a scene. The platform finds it
through the entry point and imports nothing of it by path.
"""

from .task import G1UpperReachTask

__all__ = ["G1UpperReachTask"]
