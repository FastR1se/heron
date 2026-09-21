"""Run locally what CI would run on Linux, in CI's order, stopping at the first failure.

Development is on Windows, the tests are on Linux, and the round trip through
GitHub is the slowest loop in this project. Most of what CI checks needs neither
ROS 2 nor a simulator, so it runs here: the linter, the types, the suite, the
packages, the CLI chain on dummy, the shell scripts.

What stays on Linux is named at the end rather than silently skipped.

    python .claude/skills/preflight/run.py            # всё
    python .claude/skills/preflight/run.py --quick    # линтер, типы, тесты ядра
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BIN = REPO / (".venv/Scripts" if os.name == "nt" else ".venv/bin")
SUFFIX = ".exe" if os.name == "nt" else ""


def tool(name: str) -> str:
    return str(BIN / f"{name}{SUFFIX}")


CLI = tool("heron")
REGISTRY = ["--registry", "examples/robots"]
GENERATED = REPO / "generated"


class Step:
    """One command CI runs, with what its outcome has to be."""

    def __init__(self, name, argv, *, expect_failure=False, contains=None, needs=None):
        self.name = name
        self.argv = argv
        self.expect_failure = expect_failure
        self.contains = contains
        self.needs = needs


def steps(quick: bool) -> list[Step]:
    g1 = REPO / "examples/robots/g1_23dof/vendor"
    core = [
        Step("Линтер", [tool("ruff"), "check", "src", "tests", "examples/plugins"]),
        Step("Типы — ядро", [tool("mypy")]),
        Step("Типы — пакет G1", [tool("mypy"), "examples/plugins/unitree_g1/src"]),
        Step("Типы — пакет RC", [tool("mypy"), "examples/plugins/robopro_rc/src"]),
        Step("Тесты ядра", [tool("pytest"), "-q"]),
    ]
    if quick:
        return core
    return core + [
        Step("Тесты пакетов роботов", [
            tool("pytest"), "-q",
            "examples/plugins/unitree_g1/tests", "examples/plugins/robopro_rc/tests",
        ]),
        Step("Тесты пакетов задач", [
            tool("pytest"), "-q",
            "examples/plugins/arm6_reach/tests", "examples/plugins/g1_upper_reach/tests",
            "examples/plugins/g1_stand/tests",
        ]),
        Step("CLI отвечает", [CLI, "version"]),
        Step("Профили проверяются", [
            CLI, "validate", "examples/robots/arm6.yaml", "examples/robots/rest_arm.yaml",
        ]),
        Step("Сцены проверяются", [
            CLI, "validate", "examples/scenarios/table.yaml", *REGISTRY,
        ]),
        Step("Среды порождаются", [
            CLI, "gen", "examples/scenarios/table.yaml",
            "--into", "generated/preflight", *REGISTRY,
        ]),
        Step("Порождение детерминировано", [
            CLI, "gen", "examples/scenarios/table.yaml",
            "--into", "generated/preflight", "--check", *REGISTRY,
        ]),
        Step("Командный класс на dummy", [
            CLI, "run", "examples/scenarios/rest_bench.yaml", *REGISTRY,
        ]),
        Step("Прогон на железе без шкалы не планируется", [
            CLI, "run", "examples/scenarios/rest_bench.yaml", *REGISTRY,
            "--target", "real", "--dry-run",
        ], expect_failure=True),
        Step("Потоковый навык на командном RC5 отвергается", [
            CLI, "skill", "check", "move_joints", "examples/robots/rc5/profile.yaml",
        ], expect_failure=True),
        Step("Миссия комплекса на dummy", [
            CLI, "mission", "run",
            "examples/scenarios/mixed_bench.yaml", "examples/missions/mixed_bench.yaml",
            *REGISTRY, "--into", "generated/preflight-mission",
        ], contains=None),
        Step("Вендорский робот проверяется", [
            CLI, "validate", "examples/robots/g1_23dof/profile.yaml",
        ], needs=g1),
        Step("Сцена комплекса проверяется", [
            CLI, "validate", "examples/scenarios/complex.yaml", *REGISTRY,
        ], needs=g1),
    ]


def shell_scripts() -> list[Path]:
    found: list[Path] = []
    for pattern in ("scripts/*.sh", "scripts/lib/*.sh", "scripts/runs/*.sh",
                    "examples/plugins/*/install.sh"):
        found.extend(sorted(REPO.glob(pattern)))
    return found


def check_shell() -> bool:
    """`bash -n` on every installer: CI does it, and Git Bash is here too."""
    bash = shutil.which("bash")
    if not bash:
        print("  — пропущено: bash не найден (на Linux CI проверит)")
        return True
    bad = []
    for script in shell_scripts():
        done = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        if done.returncode != 0:
            bad.append(f"{script.relative_to(REPO).as_posix()}: {done.stderr.strip()}")
    for line in bad:
        print("  " + line)
    return not bad


def run(step: Step) -> bool:
    if step.needs is not None and not step.needs.exists():
        print(f"[ ~ ] {step.name} — пропущено: нет {step.needs.relative_to(REPO).as_posix()}")
        return True
    started = time.monotonic()
    done = subprocess.run(step.argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    seconds = time.monotonic() - started
    failed = (done.returncode == 0) if step.expect_failure else (done.returncode != 0)
    output = (done.stdout or "") + (done.stderr or "")
    if not failed and step.contains and step.contains not in output:
        failed = True
        output += f"\nв выводе нет «{step.contains}»"
    mark = "!" if failed else "+"
    print(f"[ {mark} ] {step.name} ({seconds:.1f} с)")
    if failed:
        print("      " + " ".join(step.argv))
        print("\n".join("      " + line for line in output.splitlines()[-40:]))
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="линтер, типы, тесты ядра")
    args = parser.parse_args()

    if not BIN.exists():
        print(f"нет окружения {BIN}: scripts\\dev-setup.ps1", file=sys.stderr)
        return 1

    for step in steps(args.quick):
        if not run(step):
            print("\nостановлено на первом отказе; это же задание вернёт CI")
            return 1

    if not args.quick:
        print("[ ? ] Скрипты установки")
        if not check_shell():
            return 1

    shutil.rmtree(GENERATED / "preflight", ignore_errors=True)
    shutil.rmtree(GENERATED / "preflight-mission", ignore_errors=True)
    print("\nвсё прошло. На Linux останутся: задания «ядро без роботов» и «ядро без вендоров»")
    print("(чистая установка), заготовка робота пользователя, и всё, чему нужны ROS 2,")
    print("Gazebo, MuJoCo, DDS или Isaac.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
