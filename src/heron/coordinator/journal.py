"""The coordinator's part of a mission record (ADR-0070).

Events are appended as they happen, one JSON line each, and flushed: a run cut
short leaves every event up to the cut. The result is written once, at the end;
its absence is how an interrupted run reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from .coordinator import MissionResult
from .plan import EVENTS_FILE, RESULT_FILE


class Journal:
    """``events.jsonl`` and ``result.json`` of one run directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._events: TextIO = (self.directory / EVENTS_FILE).open("a", encoding="utf-8")

    def event(self, record: dict[str, Any]) -> None:
        self._events.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def finish(self, result: MissionResult) -> Path:
        self.close()
        path = self.directory / RESULT_FILE
        path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path

    def close(self) -> None:
        if not self._events.closed:
            self._events.close()


def read_events(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / EVENTS_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
