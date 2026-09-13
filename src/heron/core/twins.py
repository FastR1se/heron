"""Finding the twin of an adapter in a backend (ADR-0021, ADR-0024).

A twin is found by the pair it declares, not by an entry-point name built from
the pair: a name is a string nobody checks, and a twin announcing a pair it does
not answer for would then be found for it. Every installed twin is loaded and
asked. A twin module therefore imports nothing heavy at import time -- the same
rule adapters follow for ``connect``.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from .enums import TwinKind
from .plugins import group
from .protocols import Twin


class TwinError(LookupError):
    """An installed twin cannot be loaded, or two answer for one pair."""


def installed_twins() -> list[Twin]:
    """Every twin installed in the group, loaded and checked."""
    twins: list[Twin] = []
    for ep in entry_points(group=group("twins")):
        try:
            factory = ep.load()
        except (ImportError, AttributeError) as exc:
            raise TwinError(f"двойник {ep.name!r} не загружается — {exc}") from exc
        twin = factory() if isinstance(factory, type) else factory
        if not isinstance(twin, Twin):
            raise TwinError(
                f"двойник {ep.name!r} ({ep.value}) не объявляет adapter, backend, kind "
                f"и requirements"
            )
        try:
            TwinKind(twin.kind)
        except ValueError:
            raise TwinError(
                f"двойник {ep.name!r}: род {twin.kind!r} неизвестен; "
                f"есть: {', '.join(TwinKind)}"
            ) from None
        twins.append(twin)
    return twins


def find_twin(adapter: str, backend: str) -> Twin | None:
    """The twin answering for this adapter in this backend, if one is installed."""
    matching = [
        twin for twin in installed_twins() if twin.adapter == adapter and twin.backend == backend
    ]
    if len(matching) > 1:
        raise TwinError(
            f"адаптеру {adapter!r} в бэкенде {backend!r} отвечают двойников "
            f"{len(matching)}: {', '.join(type(twin).__name__ for twin in matching)}. "
            f"Сторона связи должна быть одна"
        )
    return matching[0] if matching else None
