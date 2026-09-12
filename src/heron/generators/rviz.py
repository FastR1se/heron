"""What a human looks at while the world runs.

Phase 2 of the roadmap is proved by sight: the robot described by the profile
stands in the world and its model matches what the physics is moving. RViz shows
exactly that pair -- ``robot_description`` and the transform tree -- so the
configuration here is not decoration but the instrument of the phase.

Everything in it is derived. The fixed frame is the root link of the robot's own
description, prefixed by the scenario id, because each robot's transforms are
published under its own prefix; without that, two instances of one profile would
publish transforms with identical names into one tree and quietly overwrite each
other. Deriving the prefix removes that collision instead of detecting it.

Foxglove is not generated here, and not by oversight: its value is time series
and recorded runs, and until a skill moves the robot there is nothing to plot
(ADR-0010).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Final

import yaml

from .common import GeneratedFile, Instance, header

RVIZ_FILE: Final = "view.rviz"


def frame_prefix(instance: Instance) -> str:
    """Transform prefix of one robot. Ends with a slash, as ROS 2 expects."""
    return f"{instance.id}/"


def fixed_frame(instance: Instance, root_link: str) -> str:
    return f"{frame_prefix(instance)}{root_link}"


def _robot_display(instance: Instance) -> dict[str, Any]:
    return {
        "Class": "rviz_default_plugins/RobotModel",
        "Name": instance.id,
        "Enabled": True,
        "Visual Enabled": True,
        "Collision Enabled": False,
        "Alpha": 1.0,
        "Description Source": "Topic",
        "Description Topic": {
            "Value": f"/{instance.id}/robot_description",
            "Depth": 5,
            "Durability Policy": "Transient Local",
            "Reliability Policy": "Reliable",
        },
        "TF Prefix": instance.id,
    }


def view_tree(instances: Sequence[Instance], frames: dict[str, str]) -> dict[str, Any]:
    """The configuration document. ``frames`` maps instance id to its root link."""
    displays: list[dict[str, Any]] = [
        {"Class": "rviz_default_plugins/Grid", "Name": "Grid", "Enabled": True,
         "Cell Size": 0.5, "Plane Cell Count": 20},
    ]
    displays.extend(_robot_display(item) for item in instances)
    displays.append({"Class": "rviz_default_plugins/TF", "Name": "TF", "Enabled": False})

    first = instances[0]
    return {
        "Panels": [
            {"Class": "rviz_common/Displays", "Name": "Displays"},
            {"Class": "rviz_common/Views", "Name": "Views"},
        ],
        "Visualization Manager": {
            "Class": "",
            "Name": "root",
            "Displays": displays,
            "Global Options": {
                "Fixed Frame": fixed_frame(first, frames[first.id]),
                "Background Color": "48; 48; 48",
                "Frame Rate": 30,
            },
            "Tools": [
                {"Class": "rviz_default_plugins/MoveCamera"},
                {"Class": "rviz_default_plugins/FocusCamera"},
            ],
            "Views": {
                "Current": {
                    "Class": "rviz_default_plugins/Orbit",
                    "Name": "Orbit",
                    "Distance": 2.5,
                    "Pitch": 0.4,
                    "Yaw": 0.8,
                    "Focal Point": {"X": 0.0, "Y": 0.0, "Z": 0.4},
                },
            },
        },
        "Window Geometry": {"Height": 800, "Width": 1200, "Displays": {"collapsed": False}},
    }


def rviz_file(instances: Sequence[Instance], frames: dict[str, str]) -> GeneratedFile:
    head = header(
        "Конфигурация визуализации",
        "роботы сцены: " + ", ".join(item.id for item in instances),
    )
    body = yaml.safe_dump(
        view_tree(instances, frames), allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    return GeneratedFile(path=PurePosixPath(RVIZ_FILE), content=head + body)
