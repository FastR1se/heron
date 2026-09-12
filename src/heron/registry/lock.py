"""The contract lock: how "content changed, revision did not" is detected.

ADR-0004 requires the validator to reject a profile whose normative fields moved
while ``revision`` stayed put. That check needs a previous state to compare
against, and a hash computed from the current file cannot provide one.

So the registry keeps a lock file next to its profiles: ``name -> {revision,
contract_id}``, recorded the moment a contract change is accepted. Three cases
follow, and each has exactly one meaning:

* identifier and revision both unchanged -- nothing happened;
* identifier changed together with revision -- a declared change, accepted;
* identifier changed while revision did not -- the silent change ADR-0004 is
  written against, so the validator stops.

The file is written only by an explicit command (``contract lock``). Updating it
automatically during validation would turn the check into a formality: it would
record whatever it just saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..core.contract import Contract

LOCK_FILENAME: Final = "contracts.lock.json"
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class LockEntry:
    """Last accepted state of one profile's contract."""

    revision: int
    contract_id: str


class ContractLock:
    """Recorded contract state of a registry root."""

    def __init__(self, path: Path, entries: dict[str, LockEntry] | None = None) -> None:
        self.path = path
        self.entries: dict[str, LockEntry] = dict(entries or {})

    @classmethod
    def load(cls, root: Path) -> ContractLock:
        """Read the lock of a registry root; a missing file is an empty lock."""
        path = root / LOCK_FILENAME
        if not path.is_file():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = {
            name: LockEntry(revision=int(item["revision"]), contract_id=str(item["contract_id"]))
            for name, item in data.get("contracts", {}).items()
        }
        return cls(path, entries)

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contracts": {
                name: {"revision": entry.revision, "contract_id": entry.contract_id}
                for name, entry in sorted(self.entries.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(self, name: str, contract: Contract) -> None:
        self.entries[name] = LockEntry(revision=contract.revision, contract_id=contract.contract_id)

    def get(self, name: str) -> LockEntry | None:
        return self.entries.get(name)

    def undeclared_change(self, name: str, contract: Contract) -> LockEntry | None:
        """The recorded entry when the contract changed silently, else ``None``."""
        previous = self.get(name)
        if previous is None:
            return None
        changed = previous.contract_id != contract.contract_id
        declared = previous.revision != contract.revision
        return previous if changed and not declared else None
