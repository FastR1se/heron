"""The artifact store and the profile's binding of a skill to a policy (ADR-0046).

No network is built here: what the store does with an artifact does not depend on
what is inside ``policy.onnx``, and the tests that run one live in
``test_learned_skill``. What is checked here is that an artifact is named by its
content, written whole or not at all, and found -- or honestly reported absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import (
    make_contract,
    make_policy_contract,
    make_profile,
    write_profile,
    write_urdf,
)
from heron.core.capabilities import Capability
from heron.core.contract import Contract
from heron.core.loading import LoadError, load_model
from heron.core.models import (
    Completion,
    OnnxSignature,
    PolicyManifest,
    PolicyRef,
    RobotProfile,
    Scenario,
)
from heron.registry.registry import Registry
from heron.skills.library import SkillLibrary, SkillNotDeclaredError
from heron.store.policies import (
    MANIFEST_FILE,
    ONNX_FILE,
    PENDING_PREFIX,
    TRAIN_FILE,
    ArtifactNotFoundError,
    PolicyArtifact,
    PolicyStore,
    StoreError,
    file_sha256,
)
from heron.validator.rules import check_profile, check_scenario

COMPLETION = Completion(tolerance=0.01, hold_s=0.05, horizon_s=2.0)


def network(path: Path, payload: bytes = b"not really a graph") -> Path:
    """A file standing in for the exported network: the store only hashes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def manifest_for(contract: Contract, onnx: Path, **overrides: Any) -> PolicyManifest:
    fields: dict[str, Any] = {
        "schema_version": 1,
        "name": "reach",
        "contract": contract,
        "contract_id": contract.contract_id,
        "capabilities": frozenset({Capability("arm.joint_position")}),
        "completion": COMPLETION,
        "onnx": OnnxSignature(
            sha256=file_sha256(onnx),
            input="obs",
            output="actions",
            observation_size=contract.observation_size or 1,
            action_size=contract.action_size or 1,
            opset=18,
        ),
    }
    fields.update(overrides)
    return PolicyManifest(**fields)


@pytest.fixture
def store(tmp_path: Path) -> PolicyStore:
    return PolicyStore(tmp_path / "store")


@pytest.fixture
def artifact(store: PolicyStore, tmp_path: Path) -> PolicyArtifact:
    onnx = network(tmp_path / "export" / "policy.onnx")
    return store.put(manifest_for(make_policy_contract(), onnx), onnx, {"seed": 7})


# -- the artifact is named by its content ----------------------------------


def test_an_artifact_is_written_under_the_identifier_of_its_manifest(
    artifact: PolicyArtifact,
) -> None:
    assert artifact.directory.name == artifact.manifest.artifact_id
    assert artifact.directory.parent.name == artifact.manifest.name
    assert {item.name for item in artifact.directory.iterdir()} == {
        MANIFEST_FILE,
        ONNX_FILE,
        TRAIN_FILE,
    }
    assert artifact.train() == {"seed": 7}


def test_the_identifier_covers_the_network(store: PolicyStore, tmp_path: Path) -> None:
    """The manifest carries the network's hash, so two networks are two artifacts."""
    contract = make_contract()
    first = store.put(
        manifest_for(contract, network(tmp_path / "a.onnx", b"one")),
        network(tmp_path / "a.onnx", b"one"),
        {},
    )
    second = store.put(
        manifest_for(contract, network(tmp_path / "b.onnx", b"two")),
        network(tmp_path / "b.onnx", b"two"),
        {},
    )
    assert first.ref.artifact_id != second.ref.artifact_id


def test_the_identifier_covers_the_completion(store: PolicyStore, tmp_path: Path) -> None:
    """Hashing the network alone would leave a changed tolerance invisible."""
    onnx = network(tmp_path / "policy.onnx")
    contract = make_contract()
    strict = manifest_for(contract, onnx, completion=COMPLETION)
    loose = manifest_for(
        contract, onnx, completion=Completion(tolerance=0.1, hold_s=0.05, horizon_s=2.0)
    )
    assert strict.artifact_id != loose.artifact_id


def test_a_network_that_does_not_match_the_manifest_is_refused(
    store: PolicyStore, tmp_path: Path
) -> None:
    onnx = network(tmp_path / "policy.onnx")
    manifest = manifest_for(make_contract(), onnx)
    network(tmp_path / "policy.onnx", b"replaced after the manifest was made")

    with pytest.raises(StoreError, match="sha256"):
        store.put(manifest, onnx, {})
    assert not store.directory(manifest.ref()).exists()


def test_writing_the_same_artifact_twice_changes_nothing(
    store: PolicyStore, tmp_path: Path
) -> None:
    onnx = network(tmp_path / "policy.onnx")
    manifest = manifest_for(make_contract(), onnx)
    first = store.put(manifest, onnx, {"seed": 1})
    second = store.put(manifest, onnx, {"seed": 2})

    assert first.directory == second.directory
    assert second.train() == {"seed": 1}, "существующий артефакт не переписывается"
    pending = [p for p in first.directory.parent.iterdir() if p.name.startswith(PENDING_PREFIX)]
    assert not pending, "каталог сборки остался после записи"


def test_capabilities_do_not_shuffle_the_identifier(tmp_path: Path) -> None:
    """A set has no order; an identifier made from one must not depend on it."""
    onnx = network(tmp_path / "policy.onnx")
    contract = make_contract()
    caps = [Capability("arm.joint_position"), Capability("arm.cartesian")]
    one = manifest_for(contract, onnx, capabilities=frozenset(caps))
    other = manifest_for(contract, onnx, capabilities=frozenset(reversed(caps)))
    assert one.artifact_id == other.artifact_id


# -- finding it, or saying it is not here ----------------------------------


def test_an_absent_artifact_is_absent_not_broken(store: PolicyStore) -> None:
    ref = PolicyRef(name="reach", artifact_id="0" * 16)
    assert store.find(ref) is None
    with pytest.raises(ArtifactNotFoundError, match="reach"):
        store.load(ref)


def test_an_unreadable_manifest_is_not_reported_as_absence(
    store: PolicyStore, artifact: PolicyArtifact
) -> None:
    (artifact.directory / MANIFEST_FILE).write_text("{не json", encoding="utf-8")
    with pytest.raises(LoadError):
        store.find(artifact.ref)


def test_the_store_lists_what_it_holds(store: PolicyStore, artifact: PolicyArtifact) -> None:
    (artifact.directory.parent / f"{PENDING_PREFIX}leftover").mkdir()
    assert list(store) == [artifact.ref]


# -- the reference ----------------------------------------------------------


def test_a_reference_is_a_name_and_a_whole_identifier() -> None:
    ref = PolicyRef.parse("arm6_reach@3f2a9c1d0b7e4a55")
    assert (ref.name, ref.artifact_id) == ("arm6_reach", "3f2a9c1d0b7e4a55")
    assert str(ref) == "arm6_reach@3f2a9c1d0b7e4a55"


@pytest.mark.parametrize(
    "text",
    ["arm6_reach", "arm6_reach@3f2a", "arm6_reach@3F2A9C1D0B7E4A55", "@3f2a9c1d0b7e4a55"],
)
def test_a_reference_that_is_not_whole_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="ссылка на политику"):
        PolicyRef.parse(text)


# -- the binding in the profile --------------------------------------------


def test_a_skill_is_a_name_or_a_binding(artifact: PolicyArtifact) -> None:
    profile = make_profile(skills=("hold", {"move_joints": str(artifact.ref)}))
    assert profile.skill("hold") is not None
    assert profile.skill("hold").ref is None  # type: ignore[union-attr]
    bound = profile.skill("move_joints")
    assert bound is not None and bound.ref == artifact.ref
    assert profile.skill("no_such_skill") is None


def test_a_binding_survives_a_round_trip_through_yaml(
    tmp_path: Path, artifact: PolicyArtifact
) -> None:
    """A profile is data: what it is written as is what it is read back as."""
    profile = make_profile(skills=("hold", {"move_joints": str(artifact.ref)}))
    path = write_profile(tmp_path, profile)

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["skills"] == ["hold", {"move_joints": str(artifact.ref)}]
    assert load_model(path, RobotProfile).skills == profile.skills


def test_one_name_gets_one_implementation(artifact: PolicyArtifact) -> None:
    with pytest.raises(ValueError, match="дважды"):
        make_profile(skills=("move_joints", {"move_joints": str(artifact.ref)}))


def test_a_reference_in_a_profile_is_checked_when_it_is_read() -> None:
    with pytest.raises(ValueError, match="ссылка на политику"):
        make_profile(skills=({"move_joints": "reach@short"},))


# -- the validator against the store ---------------------------------------


def test_a_bound_policy_that_is_not_here_is_a_warning(
    store: PolicyStore, artifact: PolicyArtifact
) -> None:
    """The data is right; this machine is not the one it was trained on (ADR-0025)."""
    missing = PolicyRef(name="reach", artifact_id="1" * 16)
    profile = make_profile(skills=({"move_joints": str(missing)},))
    report = check_profile(profile, store=store)

    assert report.ok
    assert report.codes() == ["policy.not_in_store"]


def test_a_bound_policy_that_is_here_is_checked_against_the_profile(
    store: PolicyStore, tmp_path: Path
) -> None:
    onnx = network(tmp_path / "policy.onnx")
    other = make_policy_contract(joint_order=["joint2", "joint1", "joint0"])
    stored = store.put(manifest_for(other, onnx), onnx, {})
    profile = make_profile(
        contract=make_policy_contract(), skills=({"move_joints": str(stored.ref)},)
    )

    report = check_profile(profile, store=store)
    assert not report.ok
    assert "contract.field_mismatch" in report.codes()
    assert [f.detail["field"] for f in report.errors] == ["joint_order"]


def test_a_matching_policy_passes_the_profile(
    store: PolicyStore, artifact: PolicyArtifact
) -> None:
    profile = make_profile(
        contract=make_policy_contract(), skills=({"move_joints": str(artifact.ref)},)
    )
    assert check_profile(profile, store=store).ok


def test_an_edited_manifest_no_longer_matches_its_directory(
    store: PolicyStore, artifact: PolicyArtifact
) -> None:
    edited = artifact.manifest.model_copy(
        update={"completion": Completion(tolerance=0.5, hold_s=0.0, horizon_s=1.0)}
    )
    (artifact.directory / MANIFEST_FILE).write_text(
        edited.model_dump_json(indent=2), encoding="utf-8"
    )
    profile = make_profile(skills=({"move_joints": str(artifact.ref)},))

    report = check_profile(profile, store=store)
    assert "policy.artifact_inconsistent" in report.codes()


# -- the task's skill is looked up through the profile ---------------------


def scenario_asking_for(skill: str) -> Scenario:
    return Scenario(
        schema_version=1,
        name="scene",
        world={"backend": "dummy"},
        robots=[{"id": "arm1", "profile": "test_arm", "role": "scripted"}],
        task={"skill": skill},
    )


def test_a_task_naming_a_skill_the_profile_does_not_declare_is_caught(tmp_path: Path) -> None:
    profile = make_profile(skills=("hold",))
    write_urdf(tmp_path, "test_arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)

    report = check_scenario(scenario_asking_for("move_joints"), Registry([tmp_path]))
    assert not report.ok
    finding = next(f for f in report.errors if f.code == "scenario.skill_not_in_profile")
    assert finding.detail == {
        "skill": "move_joints",
        "profile": "test_arm",
        "declared": ["hold"],
    }


def test_a_declared_skill_passes(tmp_path: Path) -> None:
    profile = make_profile()
    write_urdf(tmp_path, "test_arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)

    assert check_scenario(scenario_asking_for("move_joints"), Registry([tmp_path])).ok


def test_the_library_refuses_a_skill_the_profile_does_not_declare() -> None:
    profile = make_profile(skills=("hold",))
    with pytest.raises(SkillNotDeclaredError, match="move_joints"):
        SkillLibrary().create_for(profile, "move_joints")
