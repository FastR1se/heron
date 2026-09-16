"""A policy artifact against a robot (invariant 2, ADR-0046, ADR-0047).

Everything decidable without running the network: whether the artifact is what its
name says, whether its manifest agrees with itself, whether it was trained under
this robot's contract, and whether the one implementation of a trained skill the
platform has can execute it. The network's signature needs ``onnxruntime`` and is
checked where the network is opened (``skills.learned``).
"""

from __future__ import annotations

from typing import Final

from ..core import capabilities as caps
from ..core.contract import Contract, diff
from ..core.enums import ActionType, ObservationType
from ..core.models import PolicyManifest, RobotProfile
from ..store.policies import ONNX_FILE, PolicyArtifact, file_sha256
from .report import Report, error

UNOBSERVABLE_TERMS: Final = frozenset({ObservationType.BASE_LIN_VEL, ObservationType.BASE_ANG_VEL})
"""Terms a robot state does not carry: it has no base velocities (ADR-0042)."""

EXECUTABLE_ACTIONS: Final = frozenset({ActionType.JOINT_POSITION})
"""Actions a trained skill emits in this phase (ADR-0047)."""


def check_policy_against_profile(
    manifest: PolicyManifest,
    profile: RobotProfile,
    *,
    artifact: PolicyArtifact | None = None,
) -> Report:
    """Whether an exported policy may drive this robot.

    ``artifact`` is where the manifest was read from; with it the directory's name
    and the network file are checked against the manifest too.
    """
    report = Report()
    where = f"policy:{manifest.name} -> profile:{profile.name}"

    if artifact is not None:
        _check_artifact(artifact, report)

    if not manifest.self_consistent:
        report.add(
            error(
                "policy.manifest_inconsistent",
                where,
                f"записанный contract_id {manifest.contract_id} не соответствует "
                f"приложенному контракту ({manifest.contract.contract_id})",
                hint=(
                    "идентификатор вычисляется, а не пишется; "
                    "артефакт повреждён или отредактирован"
                ),
                adr="ADR-0004",
                detail={
                    "recorded": manifest.contract_id,
                    "computed": manifest.contract.contract_id,
                },
            )
        )
    _check_recorded_widths(manifest, report, where)

    report.extend(check_contracts(manifest.contract, profile.contract, where=where).findings)

    absent = caps.missing(manifest.capabilities, profile.capabilities)
    if absent:
        report.add(
            error(
                "policy.capabilities_missing",
                where,
                f"робот не объявляет: {', '.join(sorted(absent))}",
                adr="ADR-0001",
                detail={"missing": sorted(absent)},
            )
        )

    _check_executable(manifest, report, where)
    return report


def check_contracts(left: Contract, right: Contract, *, where: str) -> Report:
    """Compare two contracts, reporting differing fields rather than hashes."""
    report = Report()
    if left.contract_id == right.contract_id:
        return report
    differences = diff(left, right)
    for item in differences:
        report.add(
            error(
                "contract.field_mismatch",
                f"{where}.{item.field}",
                item.describe("у артефакта", "в профиле"),
                hint="переобучать политику или править профиль — решает человек",
                adr="ADR-0004",
                detail={"field": item.field, "artifact": item.left, "profile": item.right},
            )
        )
    if not differences:  # pragma: no cover - would mean a broken canonical form
        report.add(
            error(
                "contract.identity_mismatch",
                where,
                f"идентификаторы различаются ({left.contract_id} и {right.contract_id}) "
                f"при совпадающих нормативных полях",
                hint="сломана каноническая сериализация — это ошибка ядра",
                adr="ADR-0004",
            )
        )
    return report


def _check_artifact(artifact: PolicyArtifact, report: Report) -> None:
    """The directory's name and the network file against the manifest (ADR-0046)."""
    manifest = artifact.manifest
    where = f"policy:{artifact.ref}"
    computed = manifest.ref()
    if computed != artifact.ref:
        report.add(
            error(
                "policy.artifact_inconsistent",
                where,
                f"каталог назван {artifact.ref}, а манифест даёт {computed}",
                hint="идентификатор артефакта — хеш манифеста; манифест правили после записи",
                adr="ADR-0046",
                detail={"directory": str(artifact.ref), "computed": str(computed)},
            )
        )
    if not artifact.onnx_path.is_file():
        report.add(
            error(
                "policy.artifact_inconsistent",
                where,
                f"в каталоге артефакта нет {ONNX_FILE}",
                adr="ADR-0046",
                detail={"file": ONNX_FILE, "recorded": manifest.onnx.sha256, "actual": None},
            )
        )
        return
    actual = file_sha256(artifact.onnx_path)
    if actual != manifest.onnx.sha256:
        report.add(
            error(
                "policy.artifact_inconsistent",
                where,
                f"{ONNX_FILE}: sha256 {actual[:16]}…, манифест записывает "
                f"{manifest.onnx.sha256[:16]}…",
                hint="сеть заменена после экспорта; манифест её не описывает",
                adr="ADR-0046",
                detail={"file": ONNX_FILE, "recorded": manifest.onnx.sha256, "actual": actual},
            )
        )


def _check_recorded_widths(manifest: PolicyManifest, report: Report, where: str) -> None:
    """The signature the manifest records against the widths its contract derives (ADR-0042)."""
    contract = manifest.contract
    wanted = {
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
    }
    recorded = {
        "observation_size": manifest.onnx.observation_size,
        "action_size": manifest.onnx.action_size,
    }
    wrong = {name: {"recorded": recorded[name], "derived": wanted[name]}
             for name in wanted if wanted[name] is not None and wanted[name] != recorded[name]}
    if wrong:
        report.add(
            error(
                "policy.manifest_inconsistent",
                where,
                "ширины сети в манифесте не совпадают с выведенными из его контракта: "
                + ", ".join(
                    f"{name} {item['recorded']} при {item['derived']}"
                    for name, item in wrong.items()
                ),
                hint="ширины выводятся из контракта, а не объявляются",
                adr="ADR-0042",
                detail=wrong,
            )
        )


def _check_executable(manifest: PolicyManifest, report: Report, where: str) -> None:
    """What ``LearnedSkill`` cannot execute, named before anything connects (ADR-0047)."""
    contract = manifest.contract
    if contract.action.type not in EXECUTABLE_ACTIONS:
        report.add(
            error(
                "policy.action_unsupported",
                where,
                f"действие {contract.action.type}: обученный навык исполняет только "
                f"{', '.join(sorted(EXECUTABLE_ACTIONS))}",
                adr="ADR-0047",
                detail={"action": str(contract.action.type)},
            )
        )
    unobservable = [
        str(term.type) for term in contract.observation if term.type in UNOBSERVABLE_TERMS
    ]
    if unobservable:
        report.add(
            error(
                "policy.observation_unavailable",
                where,
                f"термы {', '.join(unobservable)}: в состоянии робота этих величин нет",
                hint="наблюдение собирается из RobotState; скоростей базы в нём нет",
                adr="ADR-0042",
                detail={"terms": unobservable},
            )
        )
    if not any(term.type is ObservationType.TARGET_JOINT_POS for term in contract.observation):
        report.add(
            error(
                "policy.completion_undefined",
                where,
                f"в наблюдении нет терма {ObservationType.TARGET_JOINT_POS}: "
                f"без цели завершение навыка не определено",
                adr="ADR-0043",
            )
        )
