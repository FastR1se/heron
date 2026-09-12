"""Command line interface.

Every user-facing string that would name the platform reads it from
``core.branding`` instead (section 17): the help text below says "платформа",
and the program name comes from the package metadata Typer already knows.

Exit codes are part of the interface, because CI depends on them:
0 -- nothing found, 1 -- errors found, 2 -- could not even read the input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..core.branding import APP_TITLE, data_home, env_var, image_name, registry_root
from ..core.capabilities import CORE_CAPABILITIES, EXTENSION_PREFIX
from ..core.enums import Severity
from ..core.loading import LoadError, load_model, read_yaml
from ..core.models import RobotProfile
from ..registry.lock import ContractLock
from ..registry.registry import ProfileNotFoundError, Registry
from ..scenario.loader import load_scenario
from ..skills.library import SkillLibrary, SkillNotFoundError
from ..validator.report import Report
from ..validator.rules import check_profile, check_scenario, check_skill_against_profile
from .new_robot import write_stub

EXIT_OK = 0
EXIT_FOUND_ERRORS = 1
EXIT_UNREADABLE = 2

app = typer.Typer(
    help=f"{APP_TITLE}: моделирование гетерогенных робототехнических комплексов.",
    no_args_is_help=True,
    add_completion=False,
)
profile_app = typer.Typer(help="Профили роботов.", no_args_is_help=True)
contract_app = typer.Typer(help="Контракты: идентичность и сверка.", no_args_is_help=True)
skill_app = typer.Typer(help="Библиотека навыков.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")
app.add_typer(contract_app, name="contract")
app.add_typer(skill_app, name="skill")

RootsOption = Annotated[
    list[Path] | None,
    typer.Option("--registry", "-R", help="Каталог с профилями; можно повторять."),
]


def _registry(roots: list[Path] | None) -> Registry:
    return Registry(roots or [])


def _print(report: Report) -> None:
    """Print findings, errors first: a warning after a stop is noise."""
    for finding in sorted(report, key=lambda f: (f.severity is not Severity.ERROR, f.code)):
        mark = "ошибка " if finding.severity is Severity.ERROR else "внимание"
        typer.echo(f"{mark}  {finding}")
    if not report.findings:
        typer.echo("замечаний нет")
        return
    typer.echo(f"\nитого: ошибок {len(report.errors)}, предупреждений {len(report.warnings)}")


def _finish(report: Report) -> None:
    _print(report)
    raise typer.Exit(EXIT_OK if report.ok else EXIT_FOUND_ERRORS)


def _load_profile_arg(target: str, roots: list[Path] | None) -> RobotProfile:
    """Accept either a registry name or a path to a file."""
    candidate = Path(target)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_file():
        return load_model(candidate, RobotProfile)
    return _registry(roots).load(target)


# --------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Версия платформы и пути, по которым она ищет данные."""
    from .. import __version__

    typer.echo(f"{APP_TITLE} {__version__}")
    typer.echo(f"данные:  {data_home()}  (переменная {env_var('HOME')})")
    typer.echo(f"реестр:  {registry_root()}  (переменная {env_var('REGISTRY')})")


@app.command()
def capabilities() -> None:
    """Реестр возможностей ядра (ADR-0001)."""
    for key in sorted(CORE_CAPABILITIES):
        typer.echo(key)
    typer.echo(
        f"\nреестр закрыт; возможности сторонних роботов — в домене {EXTENSION_PREFIX}*, "
        f"форма всегда <домен>.<аспект>"
    )


@app.command()
def validate(
    targets: Annotated[list[Path], typer.Argument(help="Файлы профилей и сценариев.")],
    roots: RootsOption = None,
) -> None:
    """Проверить профили и сценарии до запуска (инвариант 6)."""
    registry = _registry(roots)
    report = Report()
    for path in targets:
        try:
            if _looks_like_scenario(path):
                scenario = load_scenario(path)
                report = report.merged(check_scenario(scenario, registry))
            else:
                profile = load_model(path, RobotProfile)
                lock = ContractLock.load(path.parent)
                report = report.merged(check_profile(profile, lock=lock))
        except LoadError as exc:
            typer.echo(f"не читается  {exc}", err=True)
            raise typer.Exit(EXIT_UNREADABLE) from exc
    _finish(report)


def _looks_like_scenario(path: Path) -> bool:
    """A scenario has a world; a profile has a contract. One key decides."""
    data = read_yaml(path)
    return "world" in data and "contract" not in data


# -- profile ---------------------------------------------------------------


@profile_app.command("list")
def profile_list(roots: RootsOption = None) -> None:
    """Профили, видимые на пути поиска."""
    registry = _registry(roots)
    names = registry.names()
    if not names:
        typer.echo(f"профилей нет; искали в: {', '.join(str(r) for r in registry.roots)}")
        raise typer.Exit(EXIT_OK)
    for name in names:
        typer.echo(name)


@profile_app.command("show")
def profile_show(
    name: Annotated[str, typer.Argument(help="Имя профиля или путь к файлу.")],
    roots: RootsOption = None,
) -> None:
    """Сводка профиля: контракт, возможности, адаптеры."""
    try:
        profile = _load_profile_arg(name, roots)
    except (ProfileNotFoundError, LoadError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc

    contract = profile.contract
    typer.echo(f"{profile.name}")
    typer.echo(f"  контракт     {contract.contract_id}  revision {contract.revision}")
    typer.echo(f"  суставы      {contract.dof}: {', '.join(contract.joint_order)}")
    typer.echo(f"  единицы      {contract.angle_unit}")
    typer.echo(f"  частоты      состояние {contract.state_rate_hz:g} Гц, "
               f"команды {contract.command_rate_hz:g} Гц")
    typer.echo(f"  наблюдения   {', '.join(f'{t.type}×{t.scale:g}' for t in contract.observation)}")
    typer.echo(f"  действие     {contract.action.type}"
               + (f", clip {contract.action.clip}" if contract.action.clip else ""))
    typer.echo(f"  возможности  {', '.join(sorted(profile.capabilities))}")
    typer.echo(f"  режим        {profile.control_mode}")
    if profile.access_levels:
        typer.echo(f"  доступ       {', '.join(profile.access_levels)}")
    adapters = ", ".join(f"{k} -> {v.plugin}" for k, v in profile.adapters.items())
    typer.echo(f"  адаптеры     {adapters}")
    if profile.skills:
        typer.echo(f"  навыки       {', '.join(profile.skills)}")
    if profile.image:
        typer.echo(f"  образ        {image_name(profile.image)}")


@profile_app.command("new")
def profile_new(
    name: Annotated[str, typer.Argument(help="Имя нового робота.")],
    into: Annotated[Path, typer.Option("--into", help="Куда положить заготовку.")] = Path("."),
    dof: Annotated[int, typer.Option(help="Число суставов в заготовке.")] = 6,
) -> None:
    """Создать заготовку профиля и скелет адаптера (раздел 9)."""
    try:
        created = write_stub(name=name, into=into, dof=dof)
    except (ValueError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc
    for path in created:
        typer.echo(f"создано  {path}")
    typer.echo("\nзаполните TODO и проверьте:  validate " + str(created[0]))


# -- contract --------------------------------------------------------------


@contract_app.command("id")
def contract_id(
    name: Annotated[str, typer.Argument(help="Имя профиля или путь к файлу.")],
    roots: RootsOption = None,
) -> None:
    """Идентификатор контракта: хеш нормативных полей (ADR-0004)."""
    try:
        profile = _load_profile_arg(name, roots)
    except (ProfileNotFoundError, LoadError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc
    typer.echo(profile.contract.contract_id)


@contract_app.command("diff")
def contract_diff(
    left: Annotated[str, typer.Argument(help="Профиль или файл.")],
    right: Annotated[str, typer.Argument(help="Профиль или файл.")],
    roots: RootsOption = None,
) -> None:
    """Показать различие контрактов по полям, а не по хешам (ADR-0004)."""
    from ..core.contract import diff as contract_fields_diff

    try:
        a = _load_profile_arg(left, roots).contract
        b = _load_profile_arg(right, roots).contract
    except (ProfileNotFoundError, LoadError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc

    if a.contract_id == b.contract_id:
        typer.echo(f"контракты совпадают: {a.contract_id}")
        raise typer.Exit(EXIT_OK)

    typer.echo(f"{left}: {a.contract_id}\n{right}: {b.contract_id}\n")
    for item in contract_fields_diff(a, b):
        typer.echo(item.describe(left, right))
    raise typer.Exit(EXIT_FOUND_ERRORS)


@contract_app.command("lock")
def contract_lock(
    targets: Annotated[list[Path], typer.Argument(help="Файлы профилей.")],
) -> None:
    """Принять текущее состояние контрактов: записать revision и идентификатор.

    Только эта команда пишет lock-файл. Обновляйся он при проверке,
    проверка записывала бы ровно то, что только что увидела (ADR-0004).
    """
    touched: dict[Path, ContractLock] = {}
    for path in targets:
        try:
            profile = load_model(path, RobotProfile)
        except LoadError as exc:
            typer.echo(f"не читается  {exc}", err=True)
            raise typer.Exit(EXIT_UNREADABLE) from exc
        lock = touched.setdefault(path.parent, ContractLock.load(path.parent))
        previous = lock.get(profile.name)
        lock.record(profile.name, profile.contract)
        was = f"{previous.contract_id} rev {previous.revision} -> " if previous else ""
        now = f"{profile.contract.contract_id} rev {profile.contract.revision}"
        typer.echo(f"{profile.name}: {was}{now}")
    for lock in touched.values():
        lock.save()
        typer.echo(f"записано  {lock.path}")


# -- skill -----------------------------------------------------------------


@skill_app.command("list")
def skill_list() -> None:
    """Навыки: встроенные и установленные плагинами."""
    for name in SkillLibrary().names():
        typer.echo(name)


@skill_app.command("check")
def skill_check(
    skill: Annotated[str, typer.Argument(help="Имя навыка.")],
    profile: Annotated[str, typer.Argument(help="Имя профиля или путь к файлу.")],
    roots: RootsOption = None,
) -> None:
    """Может ли робот выполнить навык: возможности, режим, требования."""
    try:
        spec = SkillLibrary().spec(skill)
        target = _load_profile_arg(profile, roots)
    except (SkillNotFoundError, ProfileNotFoundError, LoadError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNREADABLE) from exc
    _finish(check_skill_against_profile(spec, target))


if __name__ == "__main__":  # pragma: no cover
    app()
