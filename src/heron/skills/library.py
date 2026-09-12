"""The skill library: names to implementations.

A mission names a skill; it never learns whether that name resolves to code or
to a policy (invariant 4). The library is where that indifference lives, so it
returns a factory and a declaration, never a hint about the kind.

Skills from third-party packages arrive through the same entry-point group as
built-in ones, discovered lazily: importing every installed skill to list names
would drag optional dependencies into a command that only prints a profile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

from ..core.models import SkillSpec
from ..core.plugins import PluginError, available, load
from .hold import HoldSkill

SkillFactory = Callable[[], object]

BUILTIN: dict[str, SkillFactory] = {
    HoldSkill.spec.name: HoldSkill,
}
"""Skills shipped with the core. Nothing vendor-specific may appear here."""


class SkillNotFoundError(LookupError):
    """No skill with this name, built in or installed."""


class SkillLibrary:
    """Lookup over built-in and installed skills."""

    def __init__(self, builtin: dict[str, SkillFactory] | None = None) -> None:
        self._builtin = dict(BUILTIN if builtin is None else builtin)

    def names(self) -> list[str]:
        installed = available("skills")
        return sorted(set(self._builtin) | set(installed))

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def factory(self, name: str) -> SkillFactory:
        """The callable producing a fresh instance of this skill."""
        if name in self._builtin:
            return self._builtin[name]
        try:
            loaded = load("skills", name)
        except PluginError as exc:
            raise SkillNotFoundError(str(exc)) from exc
        if not callable(loaded):
            raise SkillNotFoundError(f"{name!r}: точка входа не вызываема")
        return cast(SkillFactory, loaded)

    def spec(self, name: str) -> SkillSpec:
        """Declaration of a skill, which is what the validator compares."""
        factory = self.factory(name)
        spec = getattr(factory, "spec", None)
        if not isinstance(spec, SkillSpec):
            raise SkillNotFoundError(
                f"{name!r}: у навыка нет объявления spec: SkillSpec. "
                f"Без него валидатор не может сверить его с профилем"
            )
        return spec

    def create(self, name: str) -> object:
        return self.factory(name)()
