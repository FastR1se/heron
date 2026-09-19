"""The coordinator of a robotic system (section 4, phase 8).

A mission is data (ADR-0065); each task goes to the one robot able to do it,
decided before the run (ADR-0066), in an order written top-down (ADR-0067). The
coordinator is a process of its own and every robot runs in its executor's own
process, joined by a line protocol (ADR-0068). A failed task stops the system
(ADR-0069), and a run is a directory (ADR-0070).
"""

from .mission import Mission, MissionTask, load_mission

__all__ = ["Mission", "MissionTask", "load_mission"]
