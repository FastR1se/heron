#!/usr/bin/env bash
#
# Phase 6, stage 5: the trained skill executed in Gazebo, through the same executor and the
# same adapter as the coded one (docs/plans/phase6-learned-skill.md, "Этап 5"; ADR-0047,
# ADR-0046, ADR-0045).
#
# The claim this stage has to make good is invariant 4: a mission never learns how a skill is
# made, so swapping a coded skill for a trained one is one line in a profile. So the two runs
# below are the same command, on the same scene, with the same goal, and the only difference
# between them is that line -- which the script makes by copying the profile, not by editing
# the repository's.
#
# Built on the rule of scripts/phase4-stage5.sh: a confirmation is taken by something other
# than the thing under test. Motion is judged from the recorded run (ADR-0038), not from what
# the skill says about itself; the refusals are counted by code, not by wording; nothing stops
# at the first failure.
#
#   env      versions; the artifact present in the store; the scene generated
#   coded    the path of ADR-0019 first, with the coded skill: a torque interface and a
#            trajectory controller at time_from_start 0, before any policy is involved
#   learned  the same scene and the same goal with the profile's line swapped; recorded
#   diverge  six deliberate divergences of the artifact, each refused before `connect`
#   compare  tracking, steady-state error and time to goal, from the two records
#
# Usage:
#   bash scripts/phase6-stage5.sh                     # all steps, in order
#   bash scripts/phase6-stage5.sh env coded           # selected steps
#   GUI=1 bash scripts/phase6-stage5.sh learned       # with the simulator's window
#   REF=arm6_reach@0123456789abcdef bash scripts/phase6-stage5.sh
#
# Prerequisites: scripts/phase0.sh (ROS 2 Jazzy, Gazebo, the execution venv); a finished
# scripts/phase6-stage4.sh, whose store this script reads.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line
ENV_PREFIX="${APP^^}"                      # variables the platform reads, built from the name
readonly ENV_PREFIX

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="$REPO_ROOT/examples/scenarios/table.yaml"
readonly PROFILE="$REPO_ROOT/examples/robots/arm6.yaml"
readonly REGISTRY="$REPO_ROOT/examples/robots"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
readonly ROBOT_ID="arm1"                   # table.yaml, robots[0].id

STAGE4_ROOT="${STAGE4_ROOT:-$HOME/phase6-stage4}"
STAGE_ROOT="${STAGE_ROOT:-$HOME/phase6-stage5}"
STORE="${STORE:-$STAGE4_ROOT/store}"
REF="${REF:-$(cat "$STAGE4_ROOT/artifact.txt" 2>/dev/null)}"
GUI="${GUI:-0}"
WORLD_WAIT="${WORLD_WAIT:-40}"             # seconds for Gazebo and the controllers to come up
RUN_TIMEOUT="${RUN_TIMEOUT:-90}"
readonly STAGE4_ROOT STAGE_ROOT STORE REF GUI WORLD_WAIT RUN_TIMEOUT
readonly GEN="$STAGE_ROOT/gen"
readonly PROFILES="$STAGE_ROOT/registry"   # the copies; the repository's profile is not edited
readonly ALL_STEPS="env coded learned diverge compare"

mkdir -p "$STAGE_ROOT" "$GEN" "$PROFILES"
LOG="$STAGE_ROOT/stage5-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage}" LANG=C.UTF-8 LC_ALL=C.UTF-8
    "TERM=${TERM:-dumb}" "${ENV_PREFIX}_STORE=$STORE"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

clean() { "${BASE_ENV[@]}" bash --noprofile --norc -c "$1"; }
sourced() { "${BASE_ENV[@]}" bash --noprofile --norc -c "source '$ROS_SETUP' >/dev/null 2>&1; $1"; }

STARTED=()
LAST_GROUP=""
start_group() {  # start_group <log> <command>
    setsid "${BASE_ENV[@]}" bash --noprofile --norc -c "$2" > "$1" 2>&1 < /dev/null &
    LAST_GROUP=$!
    STARTED+=("$LAST_GROUP")
}

stop_group() {  # stop_group <pgid> <grace seconds>
    local pgid="$1" grace="$2" waited=0
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    kill -INT -- "-$pgid" 2>/dev/null
    while kill -0 -- "-$pgid" 2>/dev/null && (( waited < grace )); do
        sleep 1; waited=$(( waited + 1 ))
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null
    return 0
}

cleanup() {
    local pgid
    for pgid in "${STARTED[@]}"; do stop_group "$pgid" 15 >/dev/null 2>&1; done
}
trap cleanup EXIT

version_of() {  # version_of <python> <distribution>
    "${BASE_ENV[@]}" "$1" -c "import importlib.metadata as m; print(m.version('$2'))" 2>/dev/null \
        || echo нет
}

# The profile, copied with one line of `skills` rewritten -- which is the whole claim of
# invariant 4 stated as a file operation. The description stays where the copy points at it.
write_profile() {  # write_profile <target.yaml> <policy-ref or empty>
    "${BASE_ENV[@]}" "$EXEC_PY" - "$PROFILE" "$1" "${2:-}" <<'PY'
import sys, yaml
source, target, ref = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
data = yaml.safe_load(open(source, encoding="utf-8"))
skills = []
for item in data.get("skills", []):
    name = item if isinstance(item, str) else next(iter(item))
    skills.append({name: ref} if (ref and name == "move_joints") else name)
data["skills"] = skills
yaml.safe_dump(data, open(target, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PY
}

# A world, a run against it, and the world down again. The record is the evidence (ADR-0038).
run_against_world() {  # run_against_world <name> <registry> <extra run args...>
    local name="$1" registry="$2"; shift 2
    local world_log="$STAGE_ROOT/$name-world.log" run_log="$STAGE_ROOT/$name-run.log"
    local record_file="$STAGE_ROOT/$name.mcap" gui_flag="--headless"
    [[ $GUI == 1 ]] && gui_flag="--gui"

    start_group "$world_log" "$(printf 'source %q >/dev/null 2>&1; %q sim run %q --into %q --registry %q %s --no-rviz' \
        "$ROS_SETUP" "$EXEC_CLI" "$SCENARIO" "$GEN/$name" "$registry" "$gui_flag")"
    local world_pgid="$LAST_GROUP" waited=0
    while (( waited < WORLD_WAIT )); do
        sleep 2; waited=$(( waited + 2 ))
        sourced "ros2 control list_controllers 2>/dev/null" | grep -q 'active' && break
    done
    if ! sourced "ros2 control list_controllers 2>/dev/null" | grep -q 'active'; then
        record "$name: мир" отказ "контроллеры не поднялись за ${WORLD_WAIT} с; см. $world_log"
        stop_group "$world_pgid" 20
        return 1
    fi
    record "$name: мир" ok "контроллеры активны"

    timeout "$RUN_TIMEOUT" "${BASE_ENV[@]}" bash --noprofile --norc -c \
        "$(printf 'source %q >/dev/null 2>&1; %q run %q --robot %q --registry %q --record %q %s' \
            "$ROS_SETUP" "$EXEC_CLI" "$SCENARIO" "$ROBOT_ID" "$registry" "$record_file" "$*")" \
        > "$run_log" 2>&1
    local status=$?
    stop_group "$world_pgid" 20

    if (( status != 0 )); then
        record "$name: прогон" отказ "код $status; см. $run_log"
        tail -n 20 "$run_log" | sed 's/^/      /'
        return 1
    fi
    [[ -s $record_file ]] || { record "$name: запись" отказ "$record_file пуст"; return 1; }
    record "$name: прогон" ok "$(grep -oP 'итог\s+\K\S+' "$run_log" | tail -n 1), запись $record_file"
    return 0
}

# -- env -----------------------------------------------------------------------------------

step_env() {
    step "env: артефакт, окружение, профили двух прогонов"
    local commit
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"

    if [[ -z $REF ]]; then
        record "артефакт" отказ "ссылка не задана: REF=... или завершите scripts/phase6-stage4.sh"
        return
    fi
    if clean "$EXEC_CLI policy check $REF $PROFILE --store $STORE" \
            > "$STAGE_ROOT/policy-check.txt" 2>&1; then
        record "policy check против профиля" ok "$REF"
    else
        record "policy check против профиля" отказ "см. $STAGE_ROOT/policy-check.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-check.txt"
    fi

    ln -sfn "$REGISTRY/urdf" "$PROFILES/urdf"
    write_profile "$PROFILES/arm6.yaml" ""
    mkdir -p "$PROFILES/learned"
    ln -sfn "$REGISTRY/urdf" "$PROFILES/learned/urdf"
    write_profile "$PROFILES/learned/arm6.yaml" "$REF"

    # The one line, shown as a diff: this is the claim of invariant 4 as evidence.
    diff -u "$PROFILES/arm6.yaml" "$PROFILES/learned/arm6.yaml" > "$STAGE_ROOT/one-line.diff"
    local changed
    # Changed lines, not the two file headers. A YAML list item itself begins with "-", so the
    # line removed reads "-- move_joints": the headers are excluded by name, not by the second
    # character.
    changed="$(grep -vE '^(\+\+\+|---) ' "$STAGE_ROOT/one-line.diff" | grep -cE '^[+-]')"
    if (( changed == 2 )); then
        record "замена навыка — одна строка" ok "см. $STAGE_ROOT/one-line.diff"
    else
        record "замена навыка — одна строка" отказ "изменённых строк $changed"
    fi
    sed -n '4,$p' "$STAGE_ROOT/one-line.diff" | sed 's/^/      /'

    {
        echo "# Условия этапа 5 фазы 6 — $(date -Iseconds)"
        echo
        echo "| Что | Значение |"
        echo "|---|---|"
        echo "| Коммит платформы | $commit |"
        echo "| Артефакт | $REF |"
        echo "| Хранилище | $STORE |"
        echo "| Платформа | $(version_of "$EXEC_PY" "$APP") |"
        echo "| onnxruntime | $(version_of "$EXEC_PY" onnxruntime) |"
        echo "| Gazebo | $(command -v gz >/dev/null && gz sim --versions 2>/dev/null | head -n 1 || echo нет) |"
    } > "$STAGE_ROOT/conditions.md"
    record "условия" факт "$STAGE_ROOT/conditions.md"
}

# -- coded ---------------------------------------------------------------------------------

step_coded() {
    step "coded: навык кодом путём ADR-0019 — усилие и траекторный контроллер, до политики"
    if ! clean "$EXEC_CLI gen $SCENARIO --into $GEN/coded --registry $PROFILES" \
            > "$STAGE_ROOT/gen-coded.log" 2>&1; then
        record "coded: генерация" отказ "см. $STAGE_ROOT/gen-coded.log"
        return
    fi

    # The generated scene is read, not assumed: the drive law of ADR-0045 is in these files
    # or it is nowhere.
    local controllers description
    controllers="$(find "$GEN/coded" -name 'controllers.yaml' | head -n 1)"
    description="$(find "$GEN/coded" -name 'description.urdf' | head -n 1)"
    if grep -q 'interpolation_method: *none' "$controllers" 2>/dev/null; then
        record "coded: интерполяция выключена" ok "interpolation_method: none"
    else
        record "coded: интерполяция выключена" отказ "в $controllers этого нет"
    fi
    if grep -q '<command_interface name="effort"' "$description" 2>/dev/null; then
        record "coded: интерфейс усилия" ok "effort в описании"
    else
        record "coded: интерфейс усилия" отказ "в $description его нет"
    fi
    if grep -q 'gains:' "$controllers" 2>/dev/null; then
        record "coded: gains контроллера" ok "kp/kd из контракта"
    else
        record "coded: gains контроллера" отказ "в $controllers нет gains"
    fi

    run_against_world coded "$PROFILES"
}

# -- learned -------------------------------------------------------------------------------

step_learned() {
    step "learned: та же сцена, та же цель, одна строка профиля другая"
    [[ -n $REF ]] || { record "learned" отказ "нет ссылки на артефакт"; return; }
    run_against_world learned "$PROFILES/learned"
}

# -- diverge -------------------------------------------------------------------------------

# Criterion 4 of the phase: a divergence introduced into the artifact is refused before the
# run, by code, not by strange behaviour of a robot. Each divergence is made in a copy of the
# store, so the artifact the other steps use is never touched.
step_diverge() {
    step "diverge: внесённые расхождения артефакта — отказ до подключения"
    [[ -n $REF ]] || { record "diverge" отказ "нет ссылки на артефакт"; return; }
    local name="${REF%@*}" ident="${REF#*@}" caught=0 total=0
    local source="$STORE/policies/$name/$ident"
    [[ -d $source ]] || { record "diverge" отказ "нет каталога $source"; return; }

    local field expected
    for pair in \
        "joint_order|contract.field_mismatch" \
        "state_rate_hz|contract.field_mismatch" \
        "action_scale|contract.field_mismatch" \
        "action_offset|contract.field_mismatch" \
        "gains|contract.field_mismatch" \
        "network|policy.artifact_inconsistent"
    do
        field="${pair%%|*}"; expected="${pair##*|}"
        total=$(( total + 1 ))
        local broken="$STAGE_ROOT/diverge/$field"
        rm -rf "$broken"; mkdir -p "$broken/policies/$name/$ident"
        cp -r "$source/." "$broken/policies/$name/$ident/"
        "${BASE_ENV[@]}" "$EXEC_PY" - "$broken/policies/$name/$ident" "$field" <<'PY'
import json, pathlib, sys
directory, field = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
contract = manifest["contract"]
if field == "joint_order":
    contract["joint_order"] = list(reversed(contract["joint_order"]))
elif field == "state_rate_hz":
    contract["state_rate_hz"] = contract["state_rate_hz"] / 2
elif field == "action_scale":
    scale = contract["action"].get("scale") or [1.0] * len(contract["joint_order"])
    contract["action"]["scale"] = [value * 2 for value in scale]
elif field == "action_offset":
    contract["action"]["offset"] = "zero"
elif field == "gains":
    contract["action"]["kp"] = [value * 3 for value in contract["action"]["kp"]]
elif field == "network":
    # The bytes of the graph, not the document: the manifest still claims their hash.
    onnx = directory / "policy.onnx"
    onnx.write_bytes(onnx.read_bytes() + b"\x00")
if field != "network":
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
        local out="$STAGE_ROOT/diverge/$field.txt"
        clean "$EXEC_CLI policy check $REF $PROFILES/learned/arm6.yaml --store $broken" \
            > "$out" 2>&1
        if grep -q "$expected" "$out"; then
            caught=$(( caught + 1 ))
            record "расхождение: $field" ok "$expected до подключения"
        else
            record "расхождение: $field" отказ "не поймано; см. $out"
        fi
    done
    record "доля пойманных" факт "$caught из $total"
}

# -- compare -------------------------------------------------------------------------------

step_compare() {
    step "compare: слежение, установившаяся ошибка и время до цели по записям"
    local coded="$STAGE_ROOT/coded.mcap" learned="$STAGE_ROOT/learned.mcap"
    for f in "$coded" "$learned"; do
        [[ -s $f ]] || { record "сравнение" отказ "нет записи $f"; return; }
    done
    # Read by the metrics tool of phase 4, which reads a record with the mcap library
    # rather than with the platform's own reader: the record is the platform's claim
    # about a run, and a claim is not checked by the thing that made it.
    local out="$STAGE_ROOT/compare.md" metrics="$REPO_ROOT/scripts/phase4-stage6-metrics.py"
    local files=()
    local label
    for label in coded learned; do
        if clean "$EXEC_PY $metrics run $STAGE_ROOT/$label.mcap --label $label --out $STAGE_ROOT/$label.json"                 > "$STAGE_ROOT/$label-metrics.log" 2>&1; then
            files+=("$STAGE_ROOT/$label.json")
        else
            record "метрики: $label" отказ "см. $STAGE_ROOT/$label-metrics.log"
            sed 's/^/      /' "$STAGE_ROOT/$label-metrics.log"
        fi
    done
    (( ${#files[@]} == 2 )) || { record "сравнение" отказ "метрики посчитаны не для обоих прогонов"; return; }

    if clean "$EXEC_PY $metrics table ${files[*]}" > "$out" 2>&1; then
        record "сравнение навыка кодом и обученного" факт "$out"
        sed 's/^/      /' "$out"
    else
        record "сравнение навыка кодом и обученного" отказ "см. $out"
        sed 's/^/      /' "$out"
    fi
}

# -- main ----------------------------------------------------------------------------------

steps=("$@")
(( ${#steps[@]} )) || read -r -a steps <<< "$ALL_STEPS"
for s in "${steps[@]}"; do
    case "$s" in
        env|coded|learned|diverge|compare) ;;
        *) printf 'неизвестный шаг: %s (доступны: %s)\n' "$s" "$ALL_STEPS"; exit 2 ;;
    esac
done

step "этап 5 фазы 6: ${steps[*]} — лог $LOG"
info "платформа $(git -C "$REPO_ROOT" rev-parse --short HEAD), $APP; артефакт ${REF:-не задан}"
for s in "${steps[@]}"; do "step_$s"; done

step "итог"
for line in "${RESULTS[@]}"; do
    IFS='|' read -r check status note <<< "$line"
    printf '    %-40s %-7s %s\n' "$check" "$status" "$note"
done
info "Прислать: $LOG, $STAGE_ROOT/conditions.md, $STAGE_ROOT/one-line.diff, $STAGE_ROOT/compare.md"
