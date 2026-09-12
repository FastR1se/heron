"""Reading strict schemas from YAML.

One place turns a file into a model, so that every loader reports a malformed
document the same way. pydantic's own error text is precise but shaped for
developers; here it becomes a list of "where -- what", which is also the shape
the validator reports its findings in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class SchemaProblem:
    """One rejected field."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


class LoadError(ValueError):
    """A document is unreadable or does not match its schema."""

    def __init__(self, path: Path, problems: list[SchemaProblem]) -> None:
        self.path = path
        self.problems = problems
        detail = "; ".join(str(p) for p in problems)
        super().__init__(f"{path}: {detail}")


def _problems(error: ValidationError) -> list[SchemaProblem]:
    problems: list[SchemaProblem] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<корень>"
        problems.append(SchemaProblem(where=location, message=item["msg"]))
    return problems


def read_yaml(path: Path) -> Any:
    """Parse a YAML document, refusing anything but a mapping at the top."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LoadError(path, [SchemaProblem("<файл>", f"не читается: {exc}")]) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LoadError(path, [SchemaProblem("<синтаксис>", str(exc))]) from exc
    if data is None:
        raise LoadError(path, [SchemaProblem("<корень>", "документ пуст")])
    if not isinstance(data, dict):
        raise LoadError(
            path, [SchemaProblem("<корень>", f"ожидался словарь, получен {type(data).__name__}")]
        )
    return data


def load_model(path: Path, model: type[M]) -> M:
    """Read a YAML file into a strict schema."""
    data = read_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LoadError(path, _problems(exc)) from exc


def parse_model(data: Any, model: type[M], *, source: Path | None = None) -> M:
    """Validate already-parsed data, reporting problems the same way."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LoadError(source or Path("<данные>"), _problems(exc)) from exc
