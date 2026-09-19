"""Open today's session journal, creating the skeleton if it is not there yet.

CLAUDE.md asks for the journal "в момент события, не по памяти задним числом".
Written at the end of a session it turns into a summary of what survived, and
the dead ends -- the material of the chapter that justifies the decisions -- are
exactly what gets forgotten first.

    python .claude/skills/journal/open_today.py                 # путь, создаст при нужде
    python .claude/skills/journal/open_today.py --task "..."    # и запишет задачу

Prints the path. Never overwrites: an existing journal is only read here.
"""

import argparse
import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JOURNAL = REPO / "docs" / "journal"

SECTIONS = ("Задача", "Сделано", "Проблемы и тупики", "Наблюдения", "Дальше")

SKELETON = """# Журнал сессии {date}

## Задача

{task}

## Сделано

## Проблемы и тупики

<!-- Заполняется всегда, даже если от попытки не осталось кода: отрицательный
     результат — материал главы об обосновании решений. -->

## Наблюдения

## Дальше
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="", help="задача сессии одной строкой")
    args = parser.parse_args()

    today = datetime.date.today().isoformat()
    path = JOURNAL / f"{today}.md"
    if not path.exists():
        JOURNAL.mkdir(parents=True, exist_ok=True)
        path.write_text(SKELETON.format(date=today, task=args.task), encoding="utf-8")
        print(f"создан: {path.relative_to(REPO).as_posix()}")
    else:
        print(f"уже есть: {path.relative_to(REPO).as_posix()}")

    text = path.read_text(encoding="utf-8")
    empty = [name for name in SECTIONS if is_empty(text, name)]
    if empty:
        print("пустые разделы: " + ", ".join(empty))
    return 0


def is_empty(text: str, section: str) -> bool:
    marker = f"## {section}"
    if marker not in text:
        return True
    body = text.split(marker, 1)[1].split("\n## ", 1)[0]
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("<!--") and not line.strip().endswith("-->")
    ]
    return not lines


if __name__ == "__main__":
    sys.exit(main())
