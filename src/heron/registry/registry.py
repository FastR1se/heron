"""Registry of robot profiles.

Profiles are files, found on a search path: explicit roots first, then the
platform data directory read from the environment (section 17). Nothing here
knows a robot name -- it knows how to find YAML.

A profile lives either as ``<root>/<name>.yaml`` or as
``<root>/<name>/profile.yaml``; the second form is what a robot with meshes and
a URDF looks like, and it is what ``new-robot`` generates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core.branding import registry_root
from ..core.loading import LoadError, SchemaProblem, load_model
from ..core.models import RobotProfile
from .lock import ContractLock

PROFILE_SUFFIXES = (".yaml", ".yml")
PROFILE_BASENAME = "profile"


class ProfileNotFoundError(LookupError):
    """No profile with this name on the search path."""


@dataclass(frozen=True, slots=True)
class ProfileSource:
    """A profile together with where it came from."""

    profile: RobotProfile
    path: Path
    root: Path

    @property
    def lock(self) -> ContractLock:
        return ContractLock.load(self.root)


class Registry:
    """Profiles found on a search path."""

    def __init__(self, roots: Iterable[Path] | None = None) -> None:
        explicit = [Path(r).expanduser() for r in (roots or ())]
        default = registry_root()
        self.roots: list[Path] = explicit + ([default] if default not in explicit else [])

    def _candidates(self, root: Path, name: str) -> Iterator[Path]:
        for suffix in PROFILE_SUFFIXES:
            yield root / f"{name}{suffix}"
            yield root / name / f"{PROFILE_BASENAME}{suffix}"

    def locate(self, name: str) -> tuple[Path, Path]:
        """Path of a profile and the root it was found under."""
        for root in self.roots:
            for candidate in self._candidates(root, name):
                if candidate.is_file():
                    return candidate, root
        searched = ", ".join(str(r) for r in self.roots) or "—"
        raise ProfileNotFoundError(f"профиль {name!r} не найден. Искали в: {searched}")

    def load(self, name: str) -> RobotProfile:
        """Load a profile by name, checking that the file matches the name."""
        return self.source(name).profile

    def source(self, name: str) -> ProfileSource:
        path, root = self.locate(name)
        profile = load_model(path, RobotProfile)
        if profile.name != name:
            raise LoadError(
                path,
                [SchemaProblem("name", f"в файле {profile.name!r}, а запрошен {name!r}")],
            )
        return ProfileSource(profile=profile, path=path, root=root)

    def names(self) -> list[str]:
        """Every profile name on the search path, nearest root winning."""
        found: dict[str, None] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                if entry.is_file() and entry.suffix in PROFILE_SUFFIXES:
                    found.setdefault(entry.stem, None)
                elif entry.is_dir() and any(
                    (entry / f"{PROFILE_BASENAME}{s}").is_file() for s in PROFILE_SUFFIXES
                ):
                    found.setdefault(entry.name, None)
        return list(found)

    def load_all(self) -> dict[str, RobotProfile]:
        return {name: self.load(name) for name in self.names()}
