"""The coordinator of a robotic system (section 4, phase 8).

A mission is data (ADR-P8A); each task goes to the one robot able to do it,
decided before the run (ADR-P8B), in an order written top-down (ADR-P8C). The
coordinator is a process of its own and every robot runs in its executor's own
process, joined by a line protocol (ADR-P8D). A failed task stops the system
(ADR-P8E), and a run is a directory (ADR-P8F).
"""

from .mission import Mission, MissionTask, load_mission

__all__ = ["Mission", "MissionTask", "load_mission"]
