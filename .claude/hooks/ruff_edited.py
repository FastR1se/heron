"""PostToolUse: ruff on the file that was just edited.

The full suite takes about 76 seconds -- too long to run after every edit, and
running it there would teach the assistant to avoid editing. Ruff on one file
costs milliseconds and removes the whole class of CI returns that are nothing
but an unused import or an unsorted import block.

Scope repeats the linter step of .github/workflows/ci.yml exactly: ``src``,
``tests``, ``examples/plugins``. Nothing else is linted there, so nothing else
is linted here.

``--fix`` is applied; whatever ruff cannot fix is written on stderr with exit 2,
which hands the message back to the assistant instead of to a CI log tomorrow.
"""

import json
import subprocess
import sys
from pathlib import Path

LINTED = ("src", "tests", "examples/plugins")


def ruff(repo: Path) -> list[str]:
    """The linter of this project's environment, the PATH one as a fallback."""
    for candidate in (repo / ".venv/Scripts/ruff.exe", repo / ".venv/bin/ruff"):
        if candidate.exists():
            return [str(candidate)]
    return ["ruff"]


def main() -> int:
    payload = json.load(sys.stdin)
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(raw, str) or not raw.endswith(".py"):
        return 0

    repo = Path(payload.get("cwd") or Path.cwd()).resolve()
    path = Path(raw).resolve()
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError:
        return 0
    if not path.exists() or not relative.startswith(LINTED):
        return 0

    command = ruff(repo)
    subprocess.run([*command, "check", "--fix", "--quiet", relative], cwd=repo, capture_output=True)
    done = subprocess.run(
        [*command, "check", "--output-format", "concise", relative],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if done.returncode == 0:
        return 0
    sys.stderr.write("ruff: осталось неисправимое в " + relative + "\n")
    sys.stderr.write(done.stdout or done.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 -- a broken hook must not block work
        sys.stderr.write(f"ruff_edited: хук не отработал ({error})\n")
        sys.exit(0)
