"""The skill library: names to implementations.

A mission names a skill; it never learns whether that name resolves to code or
to a policy (invariant 4). The library is where that indifference lives, so it
returns a factory and a declaration, never a hint about the kind.

Which of the two a robot runs is the robot's profile's to say, one line in its
``skills`` (ADR-0046). So a task is resolved through a profile: a skill it binds to
a policy is a ``LearnedSkill`` over that artifact, any other skill it declares comes
from the library, and a skill it does not declare is not this robot's to run.

Skills from third-party packages arrive through the same entry-point group as
built-in ones, discovered lazily: importing every installed skill to list names
would drag optional dependencies into a command that only prints a profile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

from ..core.models import RobotProfile, SkillBinding, SkillSpec
from ..core.plugins import PluginError, available, load
from ..store.policies import PolicyStore
from .hold import HoldSkill
from .learned import LearnedSkill
from .move_joints import MoveJointsSkill

SkillFactory = Callable[[], object]

BUILTIN: dict[str, SkillFactory] = {
    HoldSkill.spec.name: HoldSkill,
    MoveJointsSkill.spec.name: MoveJointsSkill,
}
"""Skills shipped with the core. Nothing vendor-specific may appear here."""


class SkillNotFoundError(LookupError):
    """No skill with this name, built in or installed."""


class SkillNotDeclaredError(SkillNotFoundError):
    """The robot's profile does not declare the skill (``scenario.skill_not_in_profile``)."""


class SkillLibrary:
    """Lookup over built-in and installed skills, and through a profile's bindings."""

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

    # -- through a profile (ADR-0046) ----------------------------------------

    def binding(self, profile: RobotProfile, name: str) -> SkillBinding:
        """How this robot implements the skill; refused if the profile does not declare it."""
        binding = profile.skill(name)
        if binding is None:
            declared = ", ".join(item.name for item in profile.skills) or "—"
            raise SkillNotDeclaredError(
                f"навык {name!r} не объявлен в профиле {profile.name!r}; объявлены: {declared}. "
                f"Навык ищется через профиль: добавьте его в skills (ADR-0046)"
            )
        return binding

    def spec_for(
        self, profile: RobotProfile, name: str, *, store: PolicyStore | None = None
    ) -> SkillSpec:
        """The declaration of the skill as this robot runs it. Reads a manifest, no network."""
        ref = self.binding(profile, name).ref
        if ref is None:
            return self.spec(name)
        policies = store if store is not None else PolicyStore()
        return policies.load(ref).manifest.skill_spec(name)

    def create_for(
        self, profile: RobotProfile, name: str, *, store: PolicyStore | None = None
    ) -> object:
        """A fresh instance of the skill as this robot runs it.

        A policy is loaded and checked here, network included, so that what cannot
        drive the robot is refused before anyone connects (ADR-0047).
        """
        ref = self.binding(profile, name).ref
        if ref is None:
            return self.create(name)
        policies = store if store is not None else PolicyStore()
        return LearnedSkill.load(policies.load(ref), profile, name=name)
