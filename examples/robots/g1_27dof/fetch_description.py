"""Fetch the descriptions of the G1 27-DoF humanoid from its vendor.

A description is an input of the platform, not an output (ADR-0009), and these
are not ours to copy into the repository: tens of megabytes of meshes, under the
vendor's licence, versioned by the vendor. What the repository keeps instead is
where each input comes from -- a repository, a commit and a directory -- so that
the file every validator run compares against is the same file on every machine.

Two inputs, one robot. The URDF from ``unitree_ros`` is what Gazebo is started
from; the MJCF from ``unitree_mujoco`` is what the vendor's MuJoCo twin loads
(ADR-0023), together with the scene that includes it. They are fetched from the
twin's own repository at the twin's own commit, not from ``unitree_ros``, which
ships a file of the same name with a different actuator order (journal
2026-09-14).

Commits are pinned rather than following a branch. A description that moves
under a profile is exactly the drift the validator exists to catch, and a moving
download would make it catch drift nobody introduced.

Nothing is repaired silently. A checkout at another commit is reported, not
overwritten: somebody put it there, and replacing it would hide that.

Usage::

    python examples/robots/g1_27dof/fetch_description.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
VENDOR: Final = HERE / "vendor"


@dataclass(frozen=True)
class Source:
    """One vendor repository, one commit, one directory of it."""

    repository: str
    commit: str
    subdir: str
    files: tuple[str, ...]
    """Files inside ``subdir`` that must be there after the checkout."""

    @property
    def target(self) -> Path:
        return VENDOR / self.repository.rsplit("/", 1)[-1].removesuffix(".git")


# NOTE: Vendor does not ship a 27-DoF variant (mode_machine=6 has full wrists like
# 29-DoF but missing waist roll/pitch). We fetch 29-DoF files; validator will
# report diffs for the two absent waist joints (expected per ADR-0009).
SOURCES: Final = (
    Source(
        repository="https://github.com/unitreerobotics/unitree_ros.git",
        commit="7d6075f7f58588b189b940130e3edab3c839b2df",
        subdir="robots/g1_description",
        files=("g1_29dof_lock_waist.urdf",),
    ),
    Source(
        repository="https://github.com/unitreerobotics/unitree_mujoco.git",
        commit="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d",
        subdir="unitree_robots/g1",
        files=("g1_29dof.xml", "scene_29dof.xml"),
    ),
)


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


def fetch(source: Source) -> int:
    target = source.target
    expected = [target / source.subdir / name for name in source.files]

    if (target / ".git").is_dir():
        head = git("rev-parse", "HEAD", cwd=target)
        if head != source.commit:
            print(
                f"в {target} уже лежит описание другого коммита: {head}\n"
                f"ожидается {source.commit}. Скрипт не перезаписывает чужой вход; "
                f"удалите каталог, если замена намеренна.",
                file=sys.stderr,
            )
            return 1
        if all(path.is_file() for path in expected):
            print(f"на месте: {target / source.subdir}\nкоммит {source.commit}")
            return 0

    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").is_dir():
        git("init", "--quiet", cwd=target)
        git("remote", "add", "origin", source.repository, cwd=target)
    git("sparse-checkout", "set", source.subdir, cwd=target)
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", source.commit,
        cwd=target)
    git("checkout", "--quiet", "FETCH_HEAD", cwd=target)

    head = git("rev-parse", "HEAD", cwd=target)
    absent = [str(path) for path in expected if not path.is_file()]
    if head != source.commit or absent:
        print(f"загрузка не дала ожидаемого: HEAD {head}, нет {absent}", file=sys.stderr)
        return 1
    print(f"загружено: {target / source.subdir}\nкоммит {source.commit}")
    return 0


def main() -> int:
    return max(fetch(source) for source in SOURCES)


if __name__ == "__main__":
    raise SystemExit(main())