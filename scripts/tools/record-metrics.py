#!/usr/bin/env python3
"""Numbers of experiment 0001 (G1 in two twins) from run records and witness windows.

A record is read with the ``mcap`` library directly, not with the platform's reader:
the record is the platform's claim about a run, and it is read by a tool other than
the one that wrote it (the lesson of phase 4, stage 1). Standard library and
``mcap`` only.

    record-metrics.py run RECORD --label mujoco-1 [--hold HOLD.json] --out M.json
    record-metrics.py table M1.json M2.json ...

Every quantity is defined in docs/experiments/0001-g1-dds-twins.md, «Величины», and the
names below follow it. Nothing here knows the robot but what the record says: the joint
order comes from the contract stored in the record, the target from its goal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

NS = 1e9
WATCHDOG_REASON = "состояние не обновлялось"  # executor.py, Executor._trip


def _number(value: Any) -> float:
    return math.nan if value is None else float(value)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _worst(values: list[float], joints: list[str]) -> dict[str, Any]:
    index = max(range(len(values)), key=lambda i: -1.0 if math.isnan(values[i]) else values[i])
    return {"value": values[index], "joint": joints[index]}


def read(path: Path) -> tuple[dict[str, dict[str, str]], list[tuple[int, str, float, float, Any]]]:
    with path.open("rb") as stream:
        reader = make_reader(stream)
        metadata = {item.name: dict(item.metadata) for item in reader.iter_metadata()}
        messages = sorted(
            (message.sequence, channel.topic, message.log_time / NS, message.publish_time / NS,
             json.loads(message.data))
            for _, channel, message in reader.iter_messages(log_time_order=False)
        )
    return metadata, messages


def run_metrics(path: Path, label: str, hold: Path | None) -> dict[str, Any]:
    metadata, messages = read(path)
    run, result = metadata["run"], metadata.get("result", {})
    joints: list[str] = json.loads(run["contract"])["joint_order"]
    goal = json.loads(run["goal"])
    target = [float(v) for v in goal["target"]]

    # (wall, world, q) per state read; the tracking error of a state is its position
    # minus the last command sent before it -- the command in force when it was measured.
    states: list[tuple[float, float, list[float]]] = []
    errors: list[list[float]] = []
    command: list[float] | None = None
    for _, topic, wall, world, data in messages:
        if topic == "/command":
            command = [_number(v) for v in data["values"]]
        elif topic == "/state":
            q = [_number(v) for v in data["joint_position"]]
            states.append((wall, world, q))
            if command is not None:
                errors.append([a - b for a, b in zip(q, command, strict=True)])
    if len(states) < 2:
        raise SystemExit(f"{path}: состояний в записи {len(states)} — считать нечего")

    dof = len(joints)
    rms = [math.sqrt(statistics.fmean(e[j] ** 2 for e in errors)) for j in range(dof)]
    peak = [max(abs(e[j]) for e in errors) for j in range(dof)]
    final = [abs(states[-1][2][j] - target[j]) for j in range(dof)]

    fresh = [states[0]] + [b for a, b in pairwise(states) if b[1] != a[1]]
    world_steps = [(b[1] - a[1]) * 1000 for a, b in pairwise(fresh)]
    periods = [(b[0] - a[0]) * 1000 for a, b in pairwise(states)]
    wall_span = fresh[-1][0] - fresh[0][0]
    world_span = fresh[-1][1] - fresh[0][1]

    # Age of a state beyond the freshest one of the run: wall time of reading against a
    # least-squares line of world time. The slope absorbs the real-time factor, so a slow
    # world is not mistaken for a late link; what remains is how late each reading was.
    xs, ys = [s[1] for s in fresh], [s[0] for s in fresh]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    spread = sum((x - mx) ** 2 for x in xs)
    slope = (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / spread
        if spread
        else math.nan
    )
    residual = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys, strict=True)]
    age = [(r - min(residual)) * 1000 for r in residual]

    numbers: dict[str, Any] = {
        "label": label,
        "record": str(path),
        "status": result.get("status", "нет итога"),
        "reason": result.get("reason", ""),
        "contract_id": run.get("contract_id"),
        "started_utc": run.get("started_utc"),
        "duration_s": goal.get("duration"),
        "states": len(states),
        "commands": sum(1 for m in messages if m[1] == "/command"),
        "stale_reads": len(states) - len(fresh),
        "watchdog_trips": int(result.get("reason", "").startswith(WATCHDOG_REASON)),
        "rtf_run": world_span / wall_span if wall_span > 0 else math.nan,
        "world_step_ms_median": _quantile(world_steps, 0.5),
        "period_ms_median": _quantile(periods, 0.5),
        "period_ms_max": max(periods),
        "age_ms_median": _quantile(age, 0.5),
        "age_ms_p95": _quantile(age, 0.95),
        "age_ms_max": max(age),
        "tracking_rms": _worst(rms, joints),
        "tracking_peak": _worst(peak, joints),
        "final_gap": _worst(final, joints),
        "per_joint": {
            name: {"tracking_rms": rms[j], "tracking_peak": peak[j], "final_gap": final[j]}
            for j, name in enumerate(joints)
        },
    }

    if hold is not None:
        window = json.loads(hold.read_text(encoding="utf-8"))
        steady = [abs(float(window["q_mean"][j]) - target[j]) for j in range(dof)]
        numbers.update(
            {
                "steady_gap": _worst(steady, joints),
                "steady_gap_rms": math.sqrt(statistics.fmean(v**2 for v in steady)),
                "state_rate_hz": window["rate_hz"],
                "state_tick_step_ms_median": window["median_tick_step_ms"],
                "rtf_hold": window["rtf"],
                "hold_samples": window["samples"],
            }
        )
        for j, name in enumerate(joints):
            numbers["per_joint"][name]["steady_gap"] = steady[j]
            numbers["per_joint"][name]["q_std_hold"] = window["q_std"][j]
    return numbers


def _cell(value: Any, digits: int = 3) -> str:
    if isinstance(value, dict):
        return f"{value['value']:.{digits}f} ({value['joint'].removesuffix('_joint')})"
    if isinstance(value, float):
        return "—" if math.isnan(value) else f"{value:.{digits}f}"
    return "—" if value is None else str(value)


ROWS = [
    ("статус", "status", 0),
    ("состояний / команд", None, 0),
    ("несвежих чтений", "stale_reads", 0),
    ("срабатываний сторожевого таймера", "watchdog_trips", 0),
    ("RTF за прогон (мир / стенные)", "rtf_run", 3),
    ("RTF в окне удержания (свидетель)", "rtf_hold", 3),
    ("шаг мира между чтениями, медиана, мс", "world_step_ms_median", 1),
    ("период исполнителя, медиана / макс, мс", None, 1),
    ("возраст состояния сверх наименьшего, медиана / p95 / макс, мс", None, 1),
    ("частота LowState у свидетеля, Гц", "state_rate_hz", 1),
    ("шаг tick у свидетеля, медиана, мс", "state_tick_step_ms_median", 1),
    ("ошибка слежения RMS, худший сустав, рад", "tracking_rms", 3),
    ("ошибка слежения, максимум, рад", "tracking_peak", 3),
    ("до цели в конце навыка, худший сустав, рад", "final_gap", 3),
    ("установившаяся ошибка, худший сустав, рад", "steady_gap", 3),
    ("установившаяся ошибка, RMS по суставам, рад", "steady_gap_rms", 3),
]


def table(paths: list[Path]) -> str:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    lines = ["| Величина | " + " | ".join(r["label"] for r in runs) + " |",
             "|---|" + "---|" * len(runs)]
    for title, key, digits in ROWS:
        cells = []
        for r in runs:
            if title.startswith("состояний"):
                cells.append(f"{r['states']} / {r['commands']}")
            elif title.startswith("период"):
                cells.append(f"{_cell(r['period_ms_median'], 1)} / {_cell(r['period_ms_max'], 1)}")
            elif title.startswith("возраст"):
                cells.append(" / ".join(
                    _cell(r[k], 1) for k in ("age_ms_median", "age_ms_p95", "age_ms_max")))
            else:
                cells.append(_cell(r.get(key), digits))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")

    joints = list(runs[0]["per_joint"])
    worlds = sorted({r["label"].rsplit("-", 1)[0] for r in runs})
    lines += ["", "По суставам — среднее по прогонам мира: ошибка слежения RMS и установившаяся "
              "ошибка, рад", "",
              "| Сустав | " + " | ".join(f"{w} RMS | {w} уст." for w in worlds) + " |",
              "|---|" + "---|---|" * len(worlds)]
    for joint in joints:
        cells = []
        for world in worlds:
            mine = [r["per_joint"][joint] for r in runs if r["label"].rsplit("-", 1)[0] == world]
            for key in ("tracking_rms", "steady_gap"):
                values = [item[key] for item in mine if key in item]
                cells.append(_cell(statistics.fmean(values)) if values else "—")
        lines.append(f"| {joint.removesuffix('_joint')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("record", type=Path)
    one.add_argument("--label", required=True)
    one.add_argument("--hold", type=Path)
    one.add_argument("--out", type=Path, required=True)
    many = commands.add_parser("table")
    many.add_argument("metrics", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "run":
        numbers = run_metrics(args.record, args.label, args.hold)
        args.out.write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{args.label}: {numbers['status']}, состояний {numbers['states']}, "
              f"RTF {_cell(numbers['rtf_run'])}, слежение RMS {_cell(numbers['tracking_rms'])}, "
              f"установившаяся {_cell(numbers.get('steady_gap'))}")
    else:
        print(table(args.metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
