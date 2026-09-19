"""Fetch the G1 with its waist locked from its vendor, and write it down mounted (ADR-0063).

Two inputs, one written from the other. The first is the vendor's: the URDF the vendor's
table names for ``mode_machine`` 6 (ADR-0059), fetched at a pinned commit exactly as
``g1_29dof_lock_waist_rev_1_0/fetch_description.py`` fetches it. The second is the robot
this profile describes -- the same body bolted at the pelvis to a stand -- and it is
written here, by a human's script, from the first:

* a link ``world`` and a fixed joint ``world_joint`` from it to ``pelvis``, ``MOUNT_HEIGHT``
  above the stand's foot;
* the robot's name, so the profile is named by its description (ADR-0059, item 3).

Nothing else. Every other byte is the vendor's, mesh references included: they resolve
through ``mesh_dir`` of the profile, back into the vendor's directory. The platform never
writes kinematics (ADR-0009); this is not the platform, it is the input being prepared,
as the download is. Both files stay out of the repository: they are the vendor's content.

The mount stands in for the balance controller the robot has and no world of the platform
does: an ideal one, infinitely stiff. That is the condition of every run of this profile,
and ADR-0063 says what it costs.

Usage::

    python examples/robots/g1_29dof_lock_waist_rev_1_0_mounted/fetch_description.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

HERE: Final = Path(__file__).resolve().parent
VENDOR: Final = HERE / "vendor"

VENDOR_NAME: Final = "g1_29dof_lock_waist_rev_1_0"
MOUNTED_NAME: Final = f"{VENDOR_NAME}_mounted"
MOUNTED: Final = VENDOR / "mounted" / f"{MOUNTED_NAME}.urdf"
"""Where the mounted description is written. Under ``vendor/``: the vendor's content,
ignored by git like the download it is made from."""

MOUNT_HEIGHT: Final = 1.0
"""Height of the pelvis above the stand's foot, metres. Standing, the G1's pelvis is at
0.793 m (``pelvis`` in the scenes of ``unitree_mujoco``), so the feet hang about 0.2 m
clear of the floor and never touch it; Isaac Lab mounts its fixed-base G1
(``G1_INSPIRE_FTP_CFG``) at the same height."""

MOUNT: Final = (
    "\n"
    "  <!-- Mounted: the pelvis bolted to a stand (ADR-0063). Written by\n"
    "       examples/robots/g1_29dof_lock_waist_rev_1_0_mounted/fetch_description.py;\n"
    "       everything else in this file is the vendor's. -->\n"
    '  <link name="world"/>\n'
    '  <joint name="world_joint" type="fixed">\n'
    f'    <origin xyz="0 0 {MOUNT_HEIGHT:g}" rpy="0 0 0"/>\n'
    '    <parent link="world"/>\n'
    '    <child link="pelvis"/>\n'
    "  </joint>\n"
)


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


SOURCE: Final = Source(
    repository="https://github.com/unitreerobotics/unitree_ros.git",
    # The commit of g1_29dof_lock_waist_rev_1_0 and of the table that names it (ADR-0059).
    # Changing it is a change of input: run the validator.
    commit="7d6075f7f58588b189b940130e3edab3c839b2df",
    subdir="robots/g1_description",
    files=(f"{VENDOR_NAME}.urdf", "README.md"),
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
    # Only the one robot's directory, and only the one commit (meshes included: the
    # mounted description reads them from here).
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


def mount(text: str) -> str:
    """The vendor's description with the stand added and the robot renamed, nothing else.

    Refuses a description it does not recognise rather than mounting something else: a
    changed file under the same name is a changed input, and the validator is where that
    gets looked at, not here.
    """
    root = ET.fromstring(text)
    if root.get("name") != VENDOR_NAME:
        raise SystemExit(f"ожидалось описание {VENDOR_NAME!r}, в файле {root.get('name')!r}")
    if any(link.get("name") == "world" for link in root.iter("link")):
        raise SystemExit("в описании производителя уже есть звено world: крепить нечего")
    if not any(link.get("name") == "pelvis" for link in root.iter("link")):
        raise SystemExit("в описании производителя нет звена pelvis: крепить не к чему")
    opening = f'<robot name="{VENDOR_NAME}">'
    if text.count(opening) != 1:
        raise SystemExit(f"не найден ровно один {opening}")
    return text.replace(opening, f'<robot name="{MOUNTED_NAME}">{MOUNT}', 1)


def write_mounted(source: Source) -> int:
    vendor = source.target / source.subdir / f"{VENDOR_NAME}.urdf"
    mounted = mount(vendor.read_bytes().decode("utf-8"))
    MOUNTED.parent.mkdir(parents=True, exist_ok=True)
    if MOUNTED.is_file() and MOUNTED.read_bytes().decode("utf-8") == mounted:
        print(f"на месте: {MOUNTED}")
        return 0
    # Bytes, not text: the vendor's line endings survive, whatever the platform's are.
    MOUNTED.write_bytes(mounted.encode("utf-8"))
    print(f"записано: {MOUNTED}\nтаз закреплён на {MOUNT_HEIGHT:g} м над основанием стойки")
    return 0


def main() -> int:
    status = fetch(SOURCE)
    return status if status else write_mounted(SOURCE)


if __name__ == "__main__":
    raise SystemExit(main())
