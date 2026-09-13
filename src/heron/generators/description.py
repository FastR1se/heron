"""The copy of the input description a world is started from.

ADR-0009: not a line of kinematics is written here. The input tree is copied
whole -- including everything the platform does not understand -- and what the
scene needs is appended: for a driven robot, whatever its twin says; for an
obstacle, that its physics is frozen. Mesh references are anchored to the input,
because the copy is read from somewhere else (ADR-0017).
"""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Final
from xml.etree import ElementTree as ET

from ..core.urdf import parse_urdf, resolve_mesh
from .common import XML_COMMENT, GeneratedFile, Instance, header, xml_text
from .twin import require_twin

DESCRIPTION_FILE: Final = "description.urdf"


def description_path(instance: Instance) -> PurePosixPath:
    return instance.directory / DESCRIPTION_FILE


def anchor_meshes(tree: ET.Element, instance: Instance) -> None:
    """Point every mesh of the copy at the file the input meant (ADR-0017).

    The input may say ``meshes/pelvis.STL`` and be right -- relative to the
    input. The copy lives in another directory, and the simulator and the
    visualization receive it as a string with no directory at all, so a relative
    reference resolves against nothing and the robot arrives without geometry.
    The reference is resolved exactly as the validator resolved it when it
    checked the file exists, and written as an absolute ``file://`` URI.
    References only a resource path can resolve are left as they are.
    """
    mesh_dir = instance.profile.description.mesh_dir
    for mesh in tree.iter("mesh"):
        reference = mesh.get("filename")
        if reference is None:
            continue
        target = resolve_mesh(reference, instance.description, mesh_dir)
        if target is not None:
            mesh.set("filename", target.resolve().as_uri())


def description_copy(instance: Instance, root: Path) -> GeneratedFile:
    """The input description with what the scene needs appended to a copy."""
    description = parse_urdf(instance.description)
    tree = copy.deepcopy(description.tree)
    anchor_meshes(tree, instance)

    # How a robot is driven in the world is the twin's to say, never the
    # input's: a control block left in a vendor's file would be a second,
    # unchecked statement of the same thing.
    for existing in tree.findall("ros2_control"):
        tree.remove(existing)

    if instance.controlled:
        tree.extend(require_twin(instance).description_elements(instance, description, root))
    else:
        frozen = ET.Element("gazebo")
        ET.SubElement(frozen, "static").text = "true"
        tree.append(frozen)

    head = header(
        "Итоговое описание робота для запуска",
        f"{instance.description.name} + профиль {instance.profile.name}",
        contract_id=instance.profile.contract.contract_id,
        comment=XML_COMMENT,
    )
    return GeneratedFile(path=description_path(instance), content=xml_text(tree, head=head))
