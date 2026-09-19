"""``mission`` -- a system of robots doing one mission (phase 8).

Three commands, and the split is the shape of the deployment (ADR-0068):

* ``validate`` checks a mission against a scenario and prints who does what;
* ``run`` is the coordinator: it checks, writes the plan, waits for the robots,
  hands out tasks and stops the system on the first failure;
* ``robot`` is one robot's process -- the robot container of section 4 -- which
  reads its part of the plan, connects its link and runs what it is told.

``run`` starts the robot processes itself by default. With ``--no-spawn`` they are
started by hand, each in its own environment -- which a vendor's DDS robot next to
a robot on ``rclpy`` requires (ADR-0022): one process cannot hold both.

``run`` of a single robot is untouched and stays the way to drive one robot.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import typer

from ..coordinator import protocol as p
from ..coordinator.coordinator import Coordinator, MissionResult, TaskState
from ..coordinator.journal import Journal
from ..coordinator.mission import Mission, load_mission
from ..coordinator.plan import PLAN_FILE, MissionPlan, make_plan
from ..coordinator.robot import PreparedTask, RobotSide
from ..coordinator.server import RobotServer
from ..core.branding import APP_NAME, data_home
from ..core.loading import LoadError
from ..core.models import RobotProfile, Scenario, Task
from ..generators.bundle import world_environment
from ..generators.common import GeneratorError
from ..registry.registry import ProfileNotFoundError, Registry
from ..scenario.loader import load_scenario
from ..sdk.executor import Executor
from ..sdk.record import McapRecorder, RecordError, require_writer
from ..sim.process import Process, SubprocessLauncher, stop_all
from ..skills.learned import PolicyRejectedError, PolicyRuntimeError
from ..skills.library import SkillLibrary, SkillNotFoundError
from ..store.policies import ArtifactNotFoundError, PolicyStore
from ..validator.environment import Machine, check_adapter
from ..validator.mission import MissionCheck, check_mission, describe_reasons
from ..validator.report import Report
from ..validator.rules import check_scenario
from .run import (
    RunError,
    adapter_class,
    adapter_config,
    describe_environment,
    environment_conflicts,
    record_metadata,
)

EXIT_OK = 0
EXIT_FOUND_ERRORS = 1
EXIT_UNREADABLE = 2

CLI_MODULE: Final = str(__package__)
"""The command line as a module (``__main__``), so a robot process is this interpreter and
this package."""

STOP_TIMEOUT_S: Final = 10.0
"""How long a robot process is given to disconnect after ``end`` before it is stopped."""

STATE_WORDS: Final = {
    TaskState.SUCCEEDED: "выполнена",
    TaskState.FAILED: "отказ",
    TaskState.CANCELLED: "отменена",
    TaskState.NOT_STARTED: "не начата",
    TaskState.RUNNING: "идёт",
    TaskState.PENDING: "ждёт",
}

mission_app = typer.Typer(
    help="Миссии комплекса: задачи роботам по возможностям (фаза 8).", no_args_is_help=True
)

RootsOption = Annotated[
    list[Path] | None,
    typer.Option("--registry", "-R", help="Каталог с профилями; можно повторять."),
]
StoreOption = Annotated[
    Path | None,
    typer.Option("--store", help="Корень хранилища артефактов; по умолчанию — из окружения."),
]


def _main() -> Any:
    """The command line module, for what it shares; imported late, it imports this one."""
    from . import main

    return main


# -- loading and checking -----------------------------------------------------------


def _load(scenario_file: Path, mission_file: Path) -> tuple[Scenario, Mission]:
    try:
        return load_scenario(scenario_file), load_mission(mission_file)
    except LoadError as exc:
        typer.echo(f"не читается  {exc}", err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc


def check(
    scenario: Scenario,
    mission: Mission,
    registry: Registry,
    *,
    scenario_path: Path | None,
    store: PolicyStore,
    library: SkillLibrary | None = None,
) -> tuple[Report, MissionCheck]:
    """The scenario as ``validate`` checks it, and the mission against it (ADR-0066)."""
    scenario_report = check_scenario(scenario, registry, path=scenario_path, store=store)
    result = check_mission(mission, scenario, registry, library=library, store=store)
    return scenario_report.merged(result.report), result


def assignment_lines(mission: Mission, result: MissionCheck) -> list[str]:
    """Who does what, in the order written, with what each waits for."""
    width = max(len(task.id) for task in mission.tasks)
    lines = []
    for index, task in enumerate(mission.tasks):
        robot = result.assignment.get(task.id, "—")
        waits = ", ".join(mission.waits_for(index)) or "—"
        lines.append(
            f"  {task.id:<{width}}  -> {robot:<8} навык {task.skill:<12} после {waits}"
        )
        if task.id not in result.assignment:
            refusals = describe_reasons(result.reasons.get(task.id, {}))
            lines.extend(f"      {line}" for line in refusals)
    return lines


def robot_command(run_dir: Path, robot: str) -> tuple[str, ...]:
    """How the coordinator starts a robot's process: this interpreter, this package."""
    return (sys.executable, "-m", CLI_MODULE, "mission", "robot", str(run_dir), "--robot", robot)


def human_command(run_dir: Path, robot: str) -> str:
    return f"{APP_NAME} mission robot {run_dir} --robot {robot}"


def default_run_dir(mission: Mission) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return data_home() / "missions" / mission.name / stamp


def plan_lines(plan: MissionPlan, run_dir: Path) -> list[str]:
    lines = [
        f"мир       {plan.backend}  (владелец физики)",
        "роботы",
    ]
    for robot in plan.robots:
        tasks = ", ".join(task.id for task in plan.tasks_of(robot.id))
        lines.append(
            f"  {robot.id:<8} профиль {robot.profile}  связь {robot.target} -> {robot.plugin}  "
            f"контракт {robot.contract_id}  задачи {tasks}"
        )
    if plan.environment:
        lines.append(f"окружение {describe_environment(plan.environment)}")
    lines.append(f"прогон    {run_dir}")
    if plan.record:
        lines.append("запись    MCAP на каждую задачу в каталоге прогона (ADR-0070)")
    return lines


# -- validate -----------------------------------------------------------------------


@mission_app.command("validate")
def mission_validate(
    scenario_file: Annotated[Path, typer.Argument(help="Файл сценария: состав комплекса.")],
    mission_file: Annotated[Path, typer.Argument(help="Файл миссии: задачи.")],
    roots: RootsOption = None,
    store: StoreOption = None,
) -> None:
    """Проверить миссию против сцены: кто выполняет каждую задачу (ADR-0066)."""
    scenario, mission = _load(scenario_file, mission_file)
    report, result = check(
        scenario,
        mission,
        Registry(roots or []),
        scenario_path=scenario_file,
        store=PolicyStore(store),
    )
    typer.echo(f"миссия    {mission.name}: задач {len(mission.tasks)}, сцена {scenario.name}")
    for line in assignment_lines(mission, result):
        typer.echo(line)
    typer.echo("")
    _main()._finish(report)


# -- run: the coordinator -----------------------------------------------------------


@mission_app.command("run")
def mission_run(
    scenario_file: Annotated[Path, typer.Argument(help="Файл сценария: состав комплекса.")],
    mission_file: Annotated[Path, typer.Argument(help="Файл миссии: задачи.")],
    into: Annotated[
        Path | None,
        typer.Option("--into", help="Каталог прогона; по умолчанию — под данными платформы."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Показать план и команды, ничего не запуская.")
    ] = False,
    record: Annotated[
        bool, typer.Option("--record", help="Записать каждую задачу в MCAP (ADR-0070).")
    ] = False,
    spawn: Annotated[
        bool,
        typer.Option(
            "--spawn/--no-spawn",
            help="Запустить процессы роботов самому или ждать запущенных отдельно (ADR-0068).",
        ),
    ] = True,
    roots: RootsOption = None,
    store: StoreOption = None,
) -> None:
    """Выполнить миссию комплексом: координатор раздаёт задачи роботам (фаза 8).

    Мир поднимается отдельно — ``sim run``. Каждый робот работает в своём процессе;
    первый отказ останавливает комплекс (ADR-0069).
    """
    scenario, mission = _load(scenario_file, mission_file)
    registry = Registry(roots or [])
    policies = PolicyStore(store)
    report, result = check(
        scenario, mission, registry, scenario_path=scenario_file, store=policies
    )
    typer.echo(f"миссия    {mission.name}: задач {len(mission.tasks)}, сцена {scenario.name}")
    for line in assignment_lines(mission, result):
        typer.echo(line)
    if not report.ok:
        typer.echo("\nмиссия не запускается: проверка нашла ошибки (инвариант 6)\n", err=True)
        _main()._print(report, err=True)
        raise typer.Exit(EXIT_FOUND_ERRORS)

    used = set(result.assignment.values())
    try:
        profiles = {
            robot.id: registry.load(robot.profile) for robot in scenario.robots if robot.id in used
        }
    except (ProfileNotFoundError, LoadError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc

    notes: list[str] = []
    try:
        environment = world_environment(scenario, registry)
    except GeneratorError as exc:
        if not dry_run:
            typer.echo(
                f"окружение мира {scenario.name!r} на этой машине не выводится, а без него связи "
                f"ждали бы роботов в чужом домене (ADR-0039): {exc}",
                err=True,
            )
            raise typer.Exit(EXIT_FOUND_ERRORS) from exc
        environment = {}
        notes.append(f"окружение не выводится: {exc}")

    run_dir = (into if into is not None else default_run_dir(mission)).resolve()
    plan = make_plan(
        mission,
        scenario,
        result.assignment,
        profiles,
        scenario_path=scenario_file,
        registry_roots=tuple(registry.roots),
        store=policies.root,
        environment=environment,
        record=record,
    )
    for line in plan_lines(plan, run_dir) + notes:
        typer.echo(line)

    if dry_run:
        typer.echo(
            "\nпроцессы роботов (запустит координатор):"
            if spawn
            else "\nпроцессы роботов (запускаются отдельно, каждый в своём окружении):"
        )
        for robot in plan.robots:
            typer.echo(
                "  " + (" ".join(robot_command(run_dir, robot.id)) if spawn
                        else human_command(run_dir, robot.id))
            )
        typer.echo("\nсухой прогон: ничего не записано и не запущено")
        raise typer.Exit(EXIT_OK)

    if (run_dir / PLAN_FILE).exists():
        typer.echo(f"{run_dir}: в каталоге уже есть прогон ({PLAN_FILE}); укажите другой --into",
                   err=True)
        raise typer.Exit(EXIT_FOUND_ERRORS)
    if record:
        try:
            require_writer()
        except RecordError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(EXIT_FOUND_ERRORS) from exc

    try:
        outcome = coordinate(plan, run_dir, spawn=spawn)
    except KeyboardInterrupt:
        typer.echo(
            f"\nпрервано повторно; роботы остановятся сами, потеряв связь. Прогон: {run_dir}",
            err=True,
        )
        raise typer.Exit(EXIT_FOUND_ERRORS) from None
    typer.echo("")
    for line in describe_result(outcome):
        typer.echo(line)
    typer.echo(f"прогон    {run_dir}")
    raise typer.Exit(EXIT_OK if outcome.ok else EXIT_FOUND_ERRORS)


def coordinate(
    plan: MissionPlan,
    run_dir: Path,
    *,
    spawn: bool,
    env: Mapping[str, str] | None = None,
) -> MissionResult:
    """Listen, write the plan, get the robots going, run the mission, clean up."""
    server = RobotServer(robot.id for robot in plan.robots)
    plan = plan.model_copy(update={"address": server.address})
    plan.write(run_dir)
    server.open(plan.plan_id)
    journal = Journal(run_dir)
    launcher = SubprocessLauncher() if spawn else None
    started: dict[str, Process] = {}
    reported: set[str] = set()

    def watch() -> list[tuple[str, str]]:
        gone = []
        for robot, process in started.items():
            code = process.poll()
            if code is not None and robot not in reported:
                reported.add(robot)
                gone.append((robot, f"процесс робота вышел с кодом {code}"))
        return gone

    def on_event(record: dict[str, Any]) -> None:
        journal.event(record)
        line = event_line(record)
        if line is not None:
            typer.echo(line)

    result: MissionResult | None = None
    with _main()._stop_signals_as_interrupt():
        try:
            if launcher is not None:
                for robot in plan.robots:
                    started[robot.id] = launcher.start(robot_command(run_dir, robot.id), env=env)
                typer.echo(f"\nзапущено процессов роботов: {len(started)}")
            else:
                typer.echo("\nзапустите процессы роботов, каждый в своём окружении:")
                for robot in plan.robots:
                    typer.echo(f"  {human_command(run_dir, robot.id)}")
            typer.echo("")
            coordinator = Coordinator(plan, server.send, server, watch=watch, on_event=on_event)
            result = coordinator.run()
        finally:
            server.close()
            if result is not None:
                journal.finish(result)
            else:
                journal.close()
            if launcher is not None:
                _stop_robots(list(launcher.processes))
    return result


def _stop_robots(processes: list[Process]) -> None:
    """Give each robot process the time to disconnect, then stop what is left."""
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=STOP_TIMEOUT_S)
    stop_all(processes, STOP_TIMEOUT_S)


def event_line(record: Mapping[str, Any]) -> str | None:
    """An event of the journal, for a human; ``None`` for those the summary covers."""
    t = f"{record['t']:8.2f} с"
    kind = record["event"]
    if kind == "robot_connected":
        return f"{t}  {record['robot']}: подключён"
    if kind == "robot_ready":
        return f"{t}  {record['robot']}: связь установлена, готов"
    if kind == "robot_refused":
        return f"{t}  {record['robot']}: не готов — {record['reason']}"
    if kind == "robot_lost":
        return f"{t}  {record['robot']}: связь потеряна — {record['reason']}"
    if kind == "task_started":
        return f"{t}  {record['task']} -> {record['robot']}: начата, навык {record['skill']}"
    if kind == "task_finished":
        word = STATE_WORDS[TaskState(record["state"])]
        reason = f" — {record['reason']}" if record.get("reason") else ""
        return f"{t}  {record['task']} -> {record['robot']}: {word}{reason}"
    if kind == "mission_stopping":
        return f"{t}  останов комплекса: {record['reason']}"
    return None


def describe_result(result: MissionResult) -> list[str]:
    head = (
        "миссия выполнена"
        if result.ok
        else f"миссия не выполнена ({result.status}): {result.reason}"
    )
    lines = [head]
    for item in result.tasks:
        timing = (
            f"  {item.started:.2f}…{item.finished:.2f} с"
            if item.started is not None and item.finished is not None
            else ""
        )
        lines.append(
            f"  {item.task:<14} {item.robot:<8} {STATE_WORDS[item.state]:<10} тактов {item.ticks}, "
            f"команд {item.commands}{timing}"
        )
    return lines


# -- robot: one robot's process ---------------------------------------------------


@dataclass(slots=True)
class PreparedRobot:
    """A robot's process after everything that can be refused was refused."""

    executor: Executor
    tasks: dict[str, PreparedTask]
    recording: Any


def prepare_robot(
    plan: MissionPlan,
    robot_id: str,
    run_dir: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    machine: Machine | None = None,
    library: SkillLibrary | None = None,
) -> PreparedRobot:
    """What ``run`` does before it moves a robot, for this robot's part of the plan.

    In the order ``run`` has it: the world's environment (ADR-0039), the skills --
    policies included -- before the link (ADR-0047), the adapter's requirements
    (ADR-0025), then ``connect``. Raises ``RunError`` with the reason; nothing is
    connected when it does.
    """
    process = os.environ if environ is None else environ
    entry = plan.robot(robot_id)
    if entry is None:
        raise RunError(f"робота {robot_id!r} в плане нет; есть: "
                       f"{', '.join(item.id for item in plan.robots)}")
    conflicts = environment_conflicts(plan.environment, process)
    if conflicts:
        raise RunError(
            f"{robot_id}: окружение процесса расходится с окружением мира (ADR-0039):\n  "
            + "\n  ".join(conflicts)
        )
    process.update(plan.environment)

    try:
        scenario = load_scenario(Path(plan.scenario_path))
        registry = Registry([Path(root) for root in plan.registry])
        robot = next(item for item in scenario.robots if item.id == robot_id)
        profile: RobotProfile = registry.load(robot.profile)
    except (LoadError, ProfileNotFoundError, StopIteration) as exc:
        raise RunError(f"{robot_id}: сцена или профиль не читаются — {exc!r}") from exc
    if profile.contract.contract_id != entry.contract_id:
        raise RunError(
            f"{robot_id}: контракт профиля {profile.name!r} изменился после плана: было "
            f"{entry.contract_id}, стало {profile.contract.contract_id}; проверьте миссию заново"
        )

    skills = library if library is not None else SkillLibrary()
    policies = PolicyStore(Path(plan.store) if plan.store is not None else None)
    tasks: dict[str, PreparedTask] = {}
    for task in plan.tasks_of(robot_id):
        try:
            skill = skills.create_for(profile, task.skill, store=policies)
        except PolicyRejectedError as exc:
            raise RunError(
                f"{robot_id}: политика навыка {task.skill!r} не может вести этого робота:",
                exc.report,
            ) from exc
        except (PolicyRuntimeError, ArtifactNotFoundError, SkillNotFoundError, LoadError) as exc:
            raise RunError(f"{robot_id}: {exc}") from exc
        tasks[task.id] = PreparedTask(
            id=task.id, skill_name=task.skill, skill=skill, goal=dict(task.goal)
        )

    if plan.record:
        try:
            require_writer()
        except RecordError as exc:
            raise RunError(f"{robot_id}: {exc}") from exc

    environment = check_adapter(entry.plugin, machine=machine)
    if not environment.report.ok:
        raise RunError(
            f"{robot_id}: окружению этой машины не хватает того, что нужно адаптеру "
            f"{entry.plugin!r}:",
            environment.report,
        )
    adapter = adapter_class(profile, entry.target)()
    try:
        adapter.connect(profile, adapter_config(profile, robot, entry.target))
    except (RuntimeError, ImportError, ValueError) as exc:
        raise RunError(f"{robot_id}: связь {entry.target!r} не установлена — {exc}") from exc

    def recording(task: PreparedTask) -> McapRecorder | None:
        if not plan.record:
            return None
        metadata = record_metadata(
            scenario, robot, profile, entry.target, Task(skill=task.skill_name, goal=task.goal)
        )
        metadata.update({"mission": plan.mission, "task": task.id, "plan": plan.plan_id})
        return McapRecorder(run_dir / f"{task.id}.mcap", metadata)

    executor = Executor(adapter, profile, library=skills, store=policies)
    return PreparedRobot(executor=executor, tasks=tasks, recording=recording)


@mission_app.command("robot")
def mission_robot(
    run_dir: Annotated[Path, typer.Argument(help="Каталог прогона с планом.")],
    robot: Annotated[str, typer.Option("--robot", help="Идентификатор робота в сцене.")],
) -> None:
    """Процесс одного робота миссии: исполнитель, которым управляет координатор (ADR-0068)."""
    try:
        plan = MissionPlan.read(run_dir)
    except LoadError as exc:
        typer.echo(f"не читается  {exc}", err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc
    if plan.address is None:
        typer.echo(f"{run_dir}: в плане нет адреса координатора", err=True)
        raise typer.Exit(EXIT_UNREADABLE)
    try:
        channel = p.Channel.connect(plan.address.host, plan.address.port)
        channel.send(p.message(p.HELLO, robot=robot, plan=plan.plan_id))
        answer = channel.receive()
    except (OSError, p.ProtocolError) as exc:
        typer.echo(f"{robot}: координатор {plan.address.host}:{plan.address.port} не "
                   f"отвечает — {exc}", err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc
    if answer is None or answer["type"] != p.WELCOME:
        reason = (answer or {}).get("reason", "связь закрыта")
        typer.echo(f"{robot}: координатор не принял процесс — {reason}", err=True)
        channel.close()
        raise typer.Exit(EXIT_FOUND_ERRORS)

    def log(line: str) -> None:
        typer.echo(line, err=True)

    with _main()._stop_signals_as_interrupt():
        try:
            try:
                prepared = prepare_robot(plan, robot, run_dir)
            except RunError as exc:
                log(str(exc))
                _say(channel, p.message(p.REFUSED, reason=str(exc)))
                raise typer.Exit(EXIT_FOUND_ERRORS) from exc
            try:
                _say(channel, p.message(p.READY))
                log(f"{robot}: связь установлена, задач {len(prepared.tasks)}")
                ok = RobotSide(
                    robot,
                    channel,
                    prepared.executor,
                    prepared.tasks,
                    recording_for=prepared.recording,
                    log=log,
                ).serve()
            finally:
                prepared.executor.adapter.disconnect()
        finally:
            channel.close()
    raise typer.Exit(EXIT_OK if ok else EXIT_FOUND_ERRORS)


def _say(channel: p.Channel, message: Mapping[str, Any]) -> None:
    with contextlib.suppress(OSError):
        channel.send(message)
