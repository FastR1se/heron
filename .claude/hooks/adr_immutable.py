"""PreToolUse guard: a decision card is not edited after the fact.

CLAUDE.md states the rule in prose -- "нумерация сквозная, задним числом не
правятся; изменение решения — новая карточка «заменяет» со ссылкой" -- and
nothing enforces it. A card already committed is history: the assistant editing
one rewrites the reasoning a later chapter of the paper cites.

Allowed on a committed card: the ``Статус`` line alone, because superseding a
decision is recorded there and nowhere else.

Reads the hook payload on stdin, writes the refusal on stderr, exits 2 to block.
Standard library only, and any unexpected input lets the edit through: a guard
that breaks the session is worse than the rule it protects.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

CARD = re.compile(r"^\d{4}-.+\.md$")
STATUS = "Статус"


def tracked(path: Path, repo: Path) -> bool:
    """Is the file already in HEAD? An uncommitted draft is still being written."""
    relative = path.relative_to(repo).as_posix()
    done = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative}"],
        cwd=repo,
        capture_output=True,
    )
    return done.returncode == 0


def only_the_status_line(tool_input: dict) -> bool:
    """An Edit that touches the status line and nothing else.

    Cards write it as ``**Статус:** принято``, so the emphasis markers come off
    before the line is recognised.
    """
    old = tool_input.get("old_string")
    new = tool_input.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    lines = [line.strip(" *\t") for line in (old + "\n" + new).splitlines() if line.strip()]
    return bool(lines) and all(line.startswith(STATUS) for line in lines)


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path")
    if not isinstance(raw, str):
        return 0

    path = Path(raw)
    if path.parent.name != "decisions" or not CARD.match(path.name):
        return 0
    if not path.exists():
        return 0

    repo = Path(payload.get("cwd") or Path.cwd()).resolve()
    try:
        path = path.resolve()
        path.relative_to(repo)
    except ValueError:
        return 0
    if not tracked(path, repo):
        return 0
    if payload.get("tool_name") == "Edit" and only_the_status_line(tool_input):
        return 0

    number = path.name[:4]
    sys.stderr.write(
        f"Карточка ADR-{number} принята и коммитом закреплена: задним числом решения не "
        "правятся (CLAUDE.md, «Журналы»).\n"
        "Изменение решения — новая карточка «заменяет» со ссылкой на эту; в этой меняется "
        f"только строка «{STATUS}».\n"
        "Новая карточка со сквозным номером — навык /adr.\n"
    )
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 -- a broken guard must not block work
        sys.stderr.write(f"adr_immutable: хук не отработал ({error}); правка не проверена\n")
        sys.exit(0)
