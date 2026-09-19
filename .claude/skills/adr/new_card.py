"""Create the next decision card: the number, the file, the required sections.

Numbering is continuous across the project and a number is never reused, so it
is computed from the directory rather than remembered. The template carries the
five sections CLAUDE.md requires; the card is refused later, by review, if
"Альтернативы" holds no reason for each rejection.

    python .claude/skills/adr/new_card.py --slug mission-is-data --title "Миссия — данные"

Prints the path of the created card and the number to put in CLAUDE.md.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DECISIONS = REPO / "docs" / "decisions"
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

TEMPLATE = """# ADR-{number}. {title}

**Статус:** принято
**Дата:** {date}

## Контекст

<!-- Что заставило выбирать: факты, числа, ссылки на журнал и предыдущие карточки. -->

## Решение

<!-- Что выбрано. Нумерованные пункты, каждый проверяемый. -->

## Альтернативы

<!-- Каждая отвергнутая — с причиной отказа. Раздел без причин не считается заполненным. -->

## Последствия

<!-- Что теперь обязано быть правдой: тесты, правки в других местах, что сломается. -->
"""


def next_number() -> str:
    used = [int(p.name[:4]) for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")]
    return f"{max(used, default=0) + 1:04d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="краткое имя файла, English kebab-case")
    parser.add_argument("--title", required=True, help="заголовок карточки, по-русски")
    args = parser.parse_args()

    if not SLUG.match(args.slug):
        print(f"имя файла не kebab-case из латиницы: {args.slug}", file=sys.stderr)
        return 1

    number = next_number()
    path = DECISIONS / f"{number}-{args.slug}.md"
    if path.exists():
        print(f"карточка уже есть: {path}", file=sys.stderr)
        return 1
    for existing in DECISIONS.glob(f"*-{args.slug}.md"):
        print(f"это имя уже занято карточкой {existing.name}", file=sys.stderr)
        return 1

    path.write_text(
        TEMPLATE.format(
            number=number,
            title=args.title,
            date=datetime.date.today().strftime("%d.%m.%Y"),
        ),
        encoding="utf-8",
    )
    print(f"создана: {path.relative_to(REPO).as_posix()}")
    print(f"номер: ADR-{number}")
    print(f"строка в CLAUDE.md: - **{number}** <суть одной строкой>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
