"""Read-only probe of an RC-series robot controller -- phase 5, stage 0.

Answers the facts of ``docs/plans/phase5-rc5.md`` that no document states: which
ports the vendor API opens, how often real-time data changes, what the robot calls
itself, whether a reading carries a controller timestamp. It sends no motion
command, sets no controller state and changes no setting.

Two modes, chosen on the command line:

``read_only`` (default)  RTD socket only, parallel with the vendor GUI.
``--full``               also the control socket, for ``get_robot_info``. It still
                         moves nothing, but it takes control from the GUI: the GUI
                         must not be driving the robot.

The vendor API is not a package on an index; it ships as a zip archive. Point
``--api-dir`` at the directory that contains the ``API`` package.

    python scripts/phase5-probe.py --host 192.168.10.10 --api-dir ~/robopro_api
    python scripts/phase5-probe.py --host 192.168.10.10 --api-dir ~/robopro_api --full

Output is a readable log on stdout and, with ``--json``, the same facts as JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SAMPLE_SECONDS = 2.0


def log(message: str) -> None:
    print(message, flush=True)


def refuse_position_confirmation(data: Any) -> bool:
    """Never confirm a position from a probe, and never wait on a terminal for it."""
    log(f"контроллер просит подтвердить позицию: {data!r} — проба не подтверждает")
    return False


def sockets_of_this_process(host: str) -> list[str]:
    """TCP connections of this process to the robot, as ``ss`` prints them (Linux)."""
    if shutil.which("ss") is None:
        return ["ss не найден: порты не определены"]
    result = subprocess.run(["ss", "-tnp"], capture_output=True, text=True, check=False)
    marker = f"pid={os.getpid()},"
    return [line for line in result.stdout.splitlines() if marker in line and host in line]


def sample_joints(robot: Any, seconds: float) -> dict[str, Any]:
    """Read joint angles as fast as the API answers and count distinct readings."""
    readings: list[tuple[float, tuple[float, ...]]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        joints = robot.motion.joint.get_actual_position(units="deg")
        readings.append((time.monotonic(), tuple(float(v) for v in joints)))

    changes = [
        readings[i][0] for i in range(1, len(readings)) if readings[i][1] != readings[i - 1][1]
    ]
    gaps = [changes[i] - changes[i - 1] for i in range(1, len(changes))]
    return {
        "calls": len(readings),
        "calls_per_s": len(readings) / seconds,
        "distinct_changes": len(changes),
        "changes_per_s": len(changes) / seconds,
        "median_change_gap_ms": statistics.median(gaps) * 1e3 if gaps else None,
        "first_deg": readings[0][1] if readings else None,
        "last_deg": readings[-1][1] if readings else None,
        "note": (
            "неподвижный робот может давать одно и то же показание: частота изменений "
            "тогда меньше частоты RTD, и это надо видеть, а не читать как частоту"
        ),
    }


def rtd_fields(robot: Any) -> dict[str, Any]:
    """Public attributes reachable from the robot object that look like timestamps.

    A guess at names is not a fact: the list is printed so that a human reads the
    API source for the answer, not so that the probe decides it.
    """
    found: dict[str, Any] = {}
    for owner_name in ("", "motion", "safety_status", "controller_state"):
        owner = getattr(robot, owner_name) if owner_name else robot
        for name in dir(owner):
            if name.startswith("__"):
                continue
            if any(key in name.lower() for key in ("time", "stamp", "tick", "rtd")):
                found[f"{owner_name or 'robot'}.{name}"] = type(getattr(owner, name, None)).__name__
    return found


def probe(host: str, full: bool) -> dict[str, Any]:
    from API.rc_api import RobotApi  # type: ignore[import-not-found]

    facts: dict[str, Any] = {
        "host": host,
        "mode": "full" if full else "read_only",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "api_module": str(Path(sys.modules["API"].__file__ or "").parent),
    }
    log(f"подключение к {host} в режиме {facts['mode']}")
    robot = RobotApi(host, read_only=not full, autoconnect=False, timeout=5)
    with contextlib.suppress(Exception):
        robot.controller_state.set_confirm_position_callback(refuse_position_confirmation)

    started = time.monotonic()
    connected = robot.connect(read_only=not full)
    facts["connect"] = {"ok": bool(connected), "seconds": time.monotonic() - started}
    if not connected:
        log("подключение не установлено: адрес, сеть, режим или несовместимость версий (0.6)")
        return facts
    try:
        facts["sockets"] = sockets_of_this_process(host)
        facts["safety_status"] = robot.safety_status.get()
        facts["controller_state"] = robot.controller_state.get()
        facts["motion_mode"] = robot.motion.mode.get()
        facts["warning"] = robot.motion.mode.check_warning_status()
        facts["scale"] = robot.motion.scale_setup.get() if full else "только full"
        if full:
            info = robot.get_robot_info()
            facts["robot_info"] = {
                "robot_model": getattr(info, "robot_model", None),
                "client_version": getattr(info, "client_version", None),
                "dh_model": repr(getattr(info, "dh_model", None)),
            }
        facts["joints"] = sample_joints(robot, SAMPLE_SECONDS)
        facts["timestamp_like_fields"] = rtd_fields(robot)
    finally:
        robot.disconnect()
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="IPv4 контроллера, например 192.168.10.10")
    parser.add_argument("--api-dir", required=True, type=Path, help="каталог с пакетом API")
    parser.add_argument("--full", action="store_true", help="управляющий сокет, без движения")
    parser.add_argument("--json", type=Path, help="куда записать факты")
    args = parser.parse_args()

    if not (args.api_dir / "API").is_dir():
        log(f"в {args.api_dir} нет пакета API: укажите каталог, распакованный из архива")
        return 2
    sys.path.insert(0, str(args.api_dir))

    facts = probe(args.host, args.full)
    text = json.dumps(facts, ensure_ascii=False, indent=2, default=str)
    log(text)
    if args.json is not None:
        args.json.write_text(text, encoding="utf-8")
    return 0 if facts.get("connect", {}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
