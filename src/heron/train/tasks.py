"""Finding the task a learner is learning (ADR-0044).

A task is named by the scenario, in the field of the role that has one, and found
in its own entry-point group -- the sixth extension point. Unlike a twin, which
is found by the pair it answers for, a task is found by name: a scenario says
what is being learned, and two tasks under one name would be two answers to a
question that has one.

Loading a task must not need Isaac. The common layer of the protocol carries what
the platform reads before anything starts -- the backend, the declaration of the
future skill, when it is done -- and a task package therefore imports its
backend's machinery inside the methods that build a configuration, the way an
adapter imports its transport inside ``connect`` (ADR-0014).
"""

from __future__ import annotations

from ..core.plugins import PluginError, available, group, load
from ..core.protocols import TrainTask

KIND = "tasks"
"""Plugin group of training tasks, as ``branding`` names it."""


class TaskError(LookupError):
    """A task cannot be found, loaded, or does not answer for the world it is asked about.

    Carries the validator's code for what went wrong: the same three refusals
    reach a human through ``validate`` and through the command that trains, and
    one of them must not be reworded into the other (ADR-0007).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def installed_tasks() -> dict[str, str]:
    """Tasks installed in the group: name -> where it is loaded from."""
    return available(KIND)


def find_task(name: str, backend: str) -> TrainTask:
    """The task under this name, refused unless it answers for this backend.

    A task declares the backend it is trained in, as a twin declares the one it
    lives in: rewards and terminations are written in the vocabulary of one
    training environment, and a task asked for another one would be configured
    with members nobody would read.
    """
    try:
        factory = load(KIND, name)
    except PluginError as exc:
        installed = ", ".join(sorted(installed_tasks())) or "—"
        raise TaskError(
            "train.task_not_found",
            f"задача обучения {name!r} не найдена в группе {group(KIND)!r}; "
            f"установлены: {installed}. Задача — точка расширения: её ставят пакетом, "
            f"ядро о ней не знает (ADR-0044)"
        ) from exc

    task = factory() if isinstance(factory, type) else factory
    if not isinstance(task, TrainTask):
        raise TaskError(
            "train.task_unreadable",
            f"задача {name!r} ({installed_tasks().get(name, '?')}) не объявляет "
            f"backend, spec и completion",
        )
    if task.backend != backend:
        raise TaskError(
            "train.task_backend_mismatch",
            f"задача {name!r} обучается в бэкенде {task.backend!r}, а сцена объявляет "
            f"владельцем физики {backend!r}: награды и завершения задачи написаны "
            f"в словаре одной среды обучения"
        )
    return task
