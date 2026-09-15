"""The vendor's MuJoCo twin of a G1: ``unitree_mujoco``, unmodified (ADR-0023).

A simulator of its own -- it loads the scene, steps the physics and speaks
``rt/lowcmd`` and ``rt/lowstate`` -- so it owns the physics of its scene
(ADR-0021) and contributes one process to it.

Everything it is started with comes from somewhere else and is only passed on:
the scene from the scenario, the domain and the interface from the adapter's
``params``. Peers are not passed at all: the vendor's simulator has no way to
take them, and the adapter's side declaring them is enough (ADR-0023, p. 3).

Two refusals are made here, before a window opens, because both would otherwise
be discovered by the adapter after it: the simulator has no headless mode, and
it names the body from the scene's file name, so a scene for another body makes
the adapter refuse the link at ``connect``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from heron.core.enums import TwinKind
from heron.core.environment import Requirement, RequirementKind, needs
from heron.core.mjcf import MUJOCO
from heron.sim.base import SimulationError
from heron.sim.mujoco_twin import MujocoScene, TwinProcess

from .channel import LinkConfig
from .slots import MACHINES, machine_for

ADAPTER: Final = "unitree_g1_dds"
"""The adapter this twin is the other side of, as its entry point is named."""

PROGRAM: Final = "unitree_mujoco"
ROBOT: Final = "g1"
"""``-r``: which of the vendor's robots the simulator plays."""

REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    needs(
        RequirementKind.EXECUTABLE,
        PROGRAM,
        because="двойник производителя — отдельная программа",
        hint="сборка unitree_mujoco/simulate на коммитах ADR-0023 (unitree_mujoco 1eb6642, "
        "unitree_sdk2 9754cd1, MuJoCo 3.3.6), каталог сборки — на PATH; порядок сборки — шаг twin "
        "scripts/phase4-stage1.sh",
    ),
    needs(
        RequirementKind.VARIABLE,
        "DISPLAY",
        because="окно GLFW у двойника обязательно, безголового режима нет",
        hint="запускайте sim run из графической сессии",
    ),
)
"""What the process starting the world needs for this twin (ADR-0025)."""


def reported_machine(scene: Path) -> int:
    """The ``mode_machine`` the vendor's simulator reports for a scene file.

    4 when the file name contains "23", 5 otherwise: read from its bridge
    (ADR-0023), and the reason a scene is passed under its own name.
    """
    return 4 if "23" in scene.name else 5


class G1MujocoTwin:
    """``unitree_mujoco`` as the MuJoCo world of one G1."""

    adapter = ADAPTER
    backend = MUJOCO
    kind = TwinKind.OWNS_PHYSICS
    requirements: Sequence[Requirement] = REQUIREMENTS

    def processes(self, scene: MujocoScene) -> list[TwinProcess]:
        if not scene.gui:
            raise SimulationError(
                f"{scene.robot}: у {PROGRAM} нет безголового режима — окно GLFW обязательно "
                f"(ADR-0023). Запустите мир без --headless из графической сессии"
            )
        link = LinkConfig.from_params(scene.params)

        joints = scene.profile.contract.joint_order
        wanted = machine_for(joints)
        reported = reported_machine(scene.world)
        if wanted is None:
            known = ", ".join(f"{len(body)} DoF" for body in MACHINES.values())
            raise SimulationError(
                f"{scene.robot}: контракт профиля {scene.profile.name!r} ({len(joints)} суставов) "
                f"не описывает ни одно тело G1, которое знает пакет ({known})"
            )
        if wanted != reported:
            raise SimulationError(
                f"{scene.robot}: {PROGRAM} назовёт тело по имени файла сцены {scene.world.name} — "
                f"mode_machine {reported} ({len(MACHINES[reported])} DoF), а контракт описывает "
                f"{len(joints)} суставов (mode_machine {wanted}). Адаптер отказал бы при "
                f"подключении; возьмите сцену производителя для этого тела (ADR-0023, п. 4)"
            )
        return [
            TwinProcess(
                name=PROGRAM,
                command=(
                    PROGRAM,
                    "-r", ROBOT,
                    "-s", str(scene.world),
                    "-i", str(link.domain),
                    "-n", link.interface,
                ),
            )
        ]
