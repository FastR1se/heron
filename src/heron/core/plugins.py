"""Plugin discovery through Python entry points (ARCHITECTURE.md, section 9).

Installing a third-party package must add a robot without a single edit to the
core (invariant 5). Group names come from ``branding`` and are never spelled
out here, so a rename stays one file.

Built-in adapters are registered through the same groups as third-party ones.
The core keeps no private path for its own plugins: if the public mechanism were
not good enough for us, it would not be good enough for anyone.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .branding import ENTRY_POINT_GROUPS


class PluginError(LookupError):
    """A named plugin is not installed, or failed to load."""


def group(kind: str) -> str:
    """Entry-point group for a kind of plugin (``adapters``, ``sim``, ...)."""
    try:
        return ENTRY_POINT_GROUPS[kind]
    except KeyError:
        raise PluginError(
            f"{kind!r}: нет такой группы плагинов. Доступны: "
            f"{', '.join(sorted(ENTRY_POINT_GROUPS))}. Новая точка расширения — "
            f"решение, а не правка (раздел 5)"
        ) from None


def available(kind: str) -> dict[str, str]:
    """Installed plugins of a kind: name -> where it is loaded from."""
    return {ep.name: ep.value for ep in entry_points(group=group(kind))}


def load(kind: str, name: str) -> Any:
    """Load a plugin by entry-point name, or by ``module:Attribute`` path.

    The second form lets a profile point at a plugin that is present on the
    machine but not registered, which is what a freshly generated adapter stub
    looks like before it is packaged.
    """
    for ep in entry_points(group=group(kind)):
        if ep.name == name:
            return ep.load()

    if ":" in name:
        module_name, _, attribute = name.partition(":")
        try:
            module = __import__(module_name, fromlist=[attribute])
            return getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise PluginError(f"{name!r}: не загружается — {exc}") from exc

    installed = sorted(available(kind))
    raise PluginError(
        f"{name!r}: плагин не установлен в группе {group(kind)!r}. "
        f"Доступны: {', '.join(installed) or '—'}"
    )
