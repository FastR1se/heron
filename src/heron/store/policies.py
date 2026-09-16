"""Policy artifacts on disk: ``<store>/policies/<name>/<artifact_id>/`` (ADR-0046).

Three files and nothing else cross from the world a policy was trained in to the
world it runs in (invariant 2):

* ``policy.onnx`` -- the network, normalizer inside the graph;
* ``manifest.json`` -- the contract it was trained under, its identifier, what the
  skill it implements demands, when that skill is done, and the graph's signature
  with its hash;
* ``train.json`` -- the conditions of training: seed, versions, machine. Kept for
  the experiment that cites the artifact, outside its identity.

The directory is named by the manifest's content, so a name is checkable: the
identifier a directory claims and the one its manifest hashes to either agree or
the artifact has been edited. Checking that is the validator's (``policy.*``);
this module finds, reads and writes, and says nothing about whether a robot may
run what it found.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..core.branding import store_root
from ..core.loading import LoadError, SchemaProblem, parse_model
from ..core.models import PolicyManifest, PolicyRef

POLICIES_DIR: Final = "policies"
ONNX_FILE: Final = "policy.onnx"
MANIFEST_FILE: Final = "manifest.json"
TRAIN_FILE: Final = "train.json"
PENDING_PREFIX: Final = ".pending-"
"""Where an artifact is assembled before it is renamed into place (ADR-0046, item 6)."""

_CHUNK: Final = 1 << 20


class ArtifactNotFoundError(LookupError):
    """No artifact under this reference in this store."""


class StoreError(RuntimeError):
    """An artifact cannot be written as asked."""


def file_sha256(path: Path) -> str:
    """Full hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyArtifact:
    """An artifact as found: where it is, what its directory claims, what its manifest says."""

    ref: PolicyRef
    """The reference the directory is named by -- not necessarily what the manifest hashes to."""

    directory: Path
    manifest: PolicyManifest

    @property
    def onnx_path(self) -> Path:
        return self.directory / ONNX_FILE

    @property
    def train_path(self) -> Path:
        return self.directory / TRAIN_FILE

    def train(self) -> dict[str, Any]:
        """Conditions of training, or an empty mapping if the file is absent."""
        if not self.train_path.is_file():
            return {}
        data = json.loads(self.train_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}


class PolicyStore:
    """Policy artifacts under one store root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else store_root()

    @property
    def policies(self) -> Path:
        return self.root / POLICIES_DIR

    def directory(self, ref: PolicyRef) -> Path:
        return self.policies / ref.name / ref.artifact_id

    def find(self, ref: PolicyRef) -> PolicyArtifact | None:
        """The artifact under this reference, ``None`` if there is none.

        A directory whose manifest cannot be read raises ``LoadError``: something is
        there, and calling it absent would send a human to look for it elsewhere.
        """
        directory = self.directory(ref)
        if not directory.is_dir():
            return None
        return PolicyArtifact(ref=ref, directory=directory, manifest=read_manifest(directory))

    def load(self, ref: PolicyRef) -> PolicyArtifact:
        artifact = self.find(ref)
        if artifact is None:
            raise ArtifactNotFoundError(
                f"политика {ref} не найдена в хранилище {self.root} "
                f"(искали {self.directory(ref)})"
            )
        return artifact

    def put(self, manifest: PolicyManifest, onnx: Path, train: Mapping[str, Any]) -> PolicyArtifact:
        """Write an artifact, atomically, unless it is already there.

        Assembled in a pending directory inside the store and renamed into place, so
        that a reader never sees half an artifact. The identifier is the manifest's,
        and the manifest carries the network's hash: a file that does not match it
        is refused rather than stored under a name it would contradict.
        """
        actual = file_sha256(onnx)
        if actual != manifest.onnx.sha256:
            raise StoreError(
                f"{onnx}: sha256 {actual}, а манифест записывает {manifest.onnx.sha256}; "
                f"артефакт с таким именем противоречил бы себе"
            )
        ref = manifest.ref()
        target = self.directory(ref)
        if target.is_dir():
            return self.load(ref)

        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.parent / f"{PENDING_PREFIX}{ref.artifact_id}-{uuid.uuid4().hex}"
        pending.mkdir()
        try:
            (pending / MANIFEST_FILE).write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            shutil.copyfile(onnx, pending / ONNX_FILE)
            (pending / TRAIN_FILE).write_text(
                json.dumps(dict(train), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(pending, target)
        except OSError:
            shutil.rmtree(pending, ignore_errors=True)
            if target.is_dir():  # written by someone else meanwhile: the same content
                return self.load(ref)
            raise
        return self.load(ref)

    def __iter__(self) -> Iterator[PolicyRef]:
        """Every reference in the store, by name and identifier."""
        if not self.policies.is_dir():
            return
        for name in sorted(p for p in self.policies.iterdir() if p.is_dir()):
            for directory in sorted(p for p in name.iterdir() if p.is_dir()):
                if directory.name.startswith(PENDING_PREFIX):
                    continue
                try:
                    yield PolicyRef.parse(f"{name.name}@{directory.name}")
                except ValueError:
                    continue


def read_manifest(directory: Path) -> PolicyManifest:
    path = directory / MANIFEST_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LoadError(path, [SchemaProblem("<файл>", f"не читается: {exc}")]) from exc
    except json.JSONDecodeError as exc:
        raise LoadError(path, [SchemaProblem("<синтаксис>", str(exc))]) from exc
    return parse_model(data, PolicyManifest, source=path)
