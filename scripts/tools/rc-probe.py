"""Read-only probe of an RC-series robot controller -- phase 5, stage 0 and stage 4 ``probe``.

Answers the facts of ``docs/plans/phase5-rc5.md`` that no document states. It sends no
motion command, sets no controller state and changes no setting.

Written first in session 16, before the vendor's source was at hand; rewritten in stage
3 against API 1.4.1 and the package ``robopro_rc``, whose reading of the real-time
packet it uses (ADR-0062), so that the probe exercises on the robot the same copy the
adapter will rely on.

Modes, chosen on the command line:

``read_only`` (default)  RTD socket only, parallel with the vendor GUI.
``--full``               also the control socket: ``get_robot_info`` (0.22) and the
                         scale the controller reports. Moves nothing, but takes control
                         from the GUI: the GUI must not be driving the robot.

Steps, each optional:

``--stamps S``   S seconds of packets: sec/nsec, cycle_time, cycle_duty -- the unit of
                 cycle_time and the monotonicity of the stamps (0.31, 0.47).
``--watch S``    S seconds of every change of safety status, controller state, mode,
                 brakes and connection, while a human presses the emergency button
                 (0.25). With ``ignore_controller_exceptions`` as the adapter runs it.
``--pairs N``    N poses set by a human in the vendor GUI; for each, the probe waits for
                 Enter and records ``act_q`` / ``act_flange_x`` (ADR-0055). Compare with
                 ``python -m robopro_rc.kinematics <json> --urdf examples/robots/rc5/rc5.urdf``.

    python scripts/tools/rc-probe.py --host 192.168.10.10 --api-dir ~/robopro_api --json probe.json
    python scripts/tools/rc-probe.py --host 192.168.10.10 --api-dir ~/robopro_api --full
    python scripts/tools/rc-probe.py --host 192.168.10.10 --api-dir ~/robopro_api --watch 60
    python scripts/tools/rc-probe.py --host 192.168.10.10 --api-dir ~/robopro_api --pairs 5 \\
        --json pairs.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log_sdk = logging.getLogger("rc-probe.sdk")


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


def read(robot: Any) -> Any:
    """One packet, copied as the adapter copies it."""
    from robopro_rc.sdk import copy_rtd

    return copy_rtd(robot._rtd_receiver.get_data())


def stamps(robot: Any, seconds: float) -> dict[str, Any]:
    """Distinct packets for ``seconds``: how the stamp moves and what cycle_time says."""
    seen: list[tuple[int, int, float, float, float]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        data = robot._rtd_receiver.get_data()
        item = (int(data.sec), int(data.nsec), float(data.cycle_time), float(data.cycle_duty),
                time.monotonic())
        if not seen or item[:2] != seen[-1][:2]:
            seen.append(item)
        time.sleep(0.0005)
    steps = [
        (b[0] - a[0]) * 1e9 + (b[1] - a[1]) for a, b in itertools.pairwise(seen)
    ]
    return {
        "packets": len(seen),
        "packets_per_s": len(seen) / seconds,
        "stamp_steps_ns_min": min(steps) if steps else None,
        "stamp_steps_ns_max": max(steps) if steps else None,
        "backwards": sum(1 for step in steps if step <= 0),
        "cycle_time_first": seen[0][2] if seen else None,
        "cycle_time_last": seen[-1][2] if seen else None,
        "cycle_duty_last": seen[-1][3] if seen else None,
        "first_stamp": seen[0][:2] if seen else None,
        "wall_clock_at_first": time.time() - (time.monotonic() - seen[0][4]) if seen else None,
        "note": "шаг меток против числа пакетов в секунду даёт единицу cycle_time (0.31); "
        "backwards > 0 — метки не монотонны (0.47)",
    }


def watch(robot: Any, seconds: float) -> list[dict[str, Any]]:
    """Every change of what the adapter reads, while a human presses the button (0.25)."""
    changes: list[dict[str, Any]] = []
    last: tuple[Any, ...] | None = None
    started = time.monotonic()
    log(f"наблюдение {seconds:g} с: нажмите кнопку аварийного останова, затем отпустите")
    while time.monotonic() - started < seconds:
        connected = robot.is_connected()
        try:
            packet = read(robot)
            now = (connected, packet.safety, packet.controller_state, packet.motion_mode,
                   packet.brakes_released, packet.state_flags)
        except Exception as exc:  # the probe records, it does not decide
            now = (connected, f"ошибка чтения: {exc}")
        if now != last:
            entry = {"t": round(time.monotonic() - started, 3), "state": now}
            changes.append(entry)
            log(f"  {entry}")
            last = now
        time.sleep(0.005)
    return changes


def pairs(robot: Any, count: int) -> list[dict[str, Any]]:
    """Poses set by a human; for each, the angles and the flange the controller computed."""
    recorded: list[dict[str, Any]] = []
    for index in range(count):
        input(f"поза {index + 1}/{count}: выставьте робота в «Пульсе», уберите руки, Enter ")
        packet = read(robot)
        if any(abs(v) > 1e-3 for v in packet.qd):
            log("  робот движется: поза не записана, повторите")
            continue
        recorded.append({"act_q": list(packet.q), "act_flange_x": list(packet.flange),
                         "stamp": [packet.sec, packet.nsec]})
        log(f"  act_q {packet.q}\n  act_flange_x {packet.flange}")
    return recorded


def probe(args: argparse.Namespace) -> dict[str, Any]:
    from API.rc_api import RobotApi  # type: ignore[import-not-found]

    full = args.full
    facts: dict[str, Any] = {
        "host": args.host,
        "mode": "full" if full else "read_only",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "api_module": str(Path(sys.modules["API"].__file__ or "").parent),
    }
    log(f"подключение к {args.host} в режиме {facts['mode']}")
    hook = sys.excepthook
    robot = RobotApi(args.host, ignore_controller_exceptions=True, read_only=not full,
                     autoconnect=False, timeout=5, logger=log_sdk)
    sys.excepthook = hook  # ADR-0053: the SDK installs its shutdown as the process's hook
    robot.controller_state.set_confirm_position_callback(refuse_position_confirmation)
    lost: list[float] = []
    robot.set_disconnection_callback(lambda: lost.append(time.monotonic()))

    started = time.monotonic()
    try:
        connected = robot.connect(read_only=not full)
    except Exception as exc:
        connected = False
        facts["connect_error"] = repr(exc)
    facts["connect"] = {"ok": bool(connected), "seconds": time.monotonic() - started}
    if not connected:
        robot.disconnect()
        log("подключение не установлено: адрес, сеть, режим или «Пульс» держит управление")
        return facts
    try:
        facts["sockets"] = sockets_of_this_process(args.host)
        first = read(robot)
        facts["reading"] = dataclasses.asdict(first)
        facts["safety_status"] = robot.safety_status.get()
        facts["controller_state"] = robot.controller_state.get()
        facts["motion_mode"] = robot.motion.mode.get()
        facts["warning"] = robot.motion.mode.check_warning_status()
        if full:
            info = robot.get_robot_info()
            facts["robot_info"] = {
                "robot_model": info.robot_model,
                "client_version": info.client_version,
                "dh_model": repr(info.dh_model),
                "is_default": info == type(info)(),
            }
            facts["scale_setup"] = robot.motion.scale_setup.get()
            facts["act_vel_scale"] = first.velocity_scale
        if args.stamps:
            facts["stamps"] = stamps(robot, args.stamps)
        if args.watch:
            facts["watch"] = watch(robot, args.watch)
        if args.pairs:
            facts["pairs"] = pairs(robot, args.pairs)
        facts["link_lost_at"] = lost
    finally:
        robot.disconnect()
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--host", required=True, help="IPv4 контроллера, например 192.168.10.10")
    parser.add_argument("--api-dir", type=Path, help="каталог с пакетом API, если он не установлен")
    parser.add_argument("--full", action="store_true", help="управляющий сокет, без движения")
    parser.add_argument("--stamps", type=float, default=2.0, help="секунд серии меток (0 — нет)")
    parser.add_argument("--watch", type=float, default=0.0, help="секунд наблюдения кнопки")
    parser.add_argument("--pairs", type=int, default=0, help="сколько поз записать")
    parser.add_argument("--json", type=Path, help="куда записать факты")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.api_dir is not None:
        if not (args.api_dir / "API").is_dir():
            log(f"в {args.api_dir} нет пакета API: укажите каталог, распакованный из архива")
            return 2
        sys.path.insert(0, str(args.api_dir))

    facts = probe(args)
    text = json.dumps(facts, ensure_ascii=False, indent=2, default=str)
    log(text)
    if args.json is not None:
        args.json.write_text(text, encoding="utf-8")
    return 0 if facts.get("connect", {}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
