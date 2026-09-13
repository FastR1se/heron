"""Fetch the kinematic description of the G1 humanoid from its vendor.

The description is an input of the platform, not an output (ADR-0009), and this
one is not ours to copy into the repository: tens of megabytes of meshes, under
the vendor's licence, versioned by the vendor. What the repository keeps instead
is where the input comes from -- a repository, a commit and a directory -- so
that the file every validator run compares against is the same file on every
machine.

The commit is pinned rather than following a branch. A description that moves
under a profile is exactly the drift the validator exists to catch, and a moving
download would make it catch drift nobody introduced.

Nothing is repaired silently. A checkout at another commit is reported, not
overwritten: somebody put it there, and replacing it would hide that.

Usage::

    python examples/robots/g1_23dof/fetch_description.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

REPOSITORY: Final = "https://github.com/unitreerobotics/unitree_ros.git"
COMMIT: Final = "7d6075f7f58588b189b940130e3edab3c839b2df"
"""master on 2026-08-28. Changing it is a change of input: run the validator."""

SUBDIR: Final = "robots/g1_description"
DESCRIPTION: Final = "g1_23dof.urdf"

HERE: Final = Path(__file__).resolve().parent
TARGET: Final = HERE / "vendor" / "unitree_ros"


def git(*args: str, cwd: Path | None = None) -> str:
    """Run git, keeping bytes as the vendor wrote them (no line-ending rewrite)."""
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} завершился с ошибкой:\n{result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    urdf = TARGET / SUBDIR / DESCRIPTION

    if (TARGET / ".git").is_dir():
        head = git("rev-parse", "HEAD", cwd=TARGET)
        if head != COMMIT:
            print(
                f"в {TARGET} уже лежит описание другого коммита: {head}\n"
                f"ожидается {COMMIT}. Скрипт не перезаписывает чужой вход; "
                f"удалите каталог, если замена намеренна.",
                file=sys.stderr,
            )
            return 1
        if urdf.is_file():
            print(f"описание на месте: {urdf}\nкоммит {COMMIT}")
            return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    if not (TARGET / ".git").is_dir():
        git("init", "--quiet", cwd=TARGET)
        git("remote", "add", "origin", REPOSITORY, cwd=TARGET)
    # Only the one robot's directory, and only the one commit: the repository
    # holds every robot the vendor makes, and none of the others is an input.
    git("sparse-checkout", "set", SUBDIR, cwd=TARGET)
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=TARGET)
    git("checkout", "--quiet", "FETCH_HEAD", cwd=TARGET)

    head = git("rev-parse", "HEAD", cwd=TARGET)
    if head != COMMIT or not urdf.is_file():
        print(f"загрузка не дала ожидаемого: HEAD {head}, файл {urdf}", file=sys.stderr)
        return 1
    print(f"описание загружено: {urdf}\nкоммит {COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
