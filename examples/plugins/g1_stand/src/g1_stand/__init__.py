"""Training task for the G1 on a free base: keep standing (ADR-0075).

Built the way a user builds one for their own robot: a package, an entry point, a scene.
The platform finds it through the entry point and imports nothing of it by path.
"""

from .task import G1StandTask

__all__ = ["G1StandTask"]
