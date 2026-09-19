#!/usr/bin/env bash
#
# Phase 6, stage 7: the second implementation of the pipeline -- the mounted G1's upper body,
# trained in Isaac Lab and executed in Gazebo through the bridge of its package by the same DDS
# adapter as a coded skill (docs/plans/phase6-learned-skill.md, "Этап 7"; ADR-0040, ADR-P6A,
# ADR-P6B, ADR-0044, ADR-0046, ADR-0047).
#
# Built like scripts/phase6-stage4.sh and phase6-stage5.sh, on the same rule: a claim stands on
# a confirmation taken by something other than the thing under test. The derivation is read off
# the dry run and checked against numbers written here; the mount off Isaac's own body positions,
# not off the description that asks for it; the threshold of ADR-0005 off nvidia-smi and /proc;
# the artifact by `policy check` in the execution venv; motion off the run's record (ADR-0038)
# by the metrics tool of phase 4. Nothing stops at the first failure.
#
#   env      versions and machine into conditions.md, with the run condition of ADR-P6A; the
#            mounted description written; the platform and the task in the training venv, the
#            platform, the G1 package, mcap and onnxruntime in the adapter's venv; rclpy must
#            NOT import in the training venv (invariant 2)
#   plan     validate; the gains stable at 500 Hz on the description's inertias; train env;
#            train run --dry-run and its numbers
#   mount    the mounted description converted by Isaac Lab and stepped: the pelvis must stand
#            at the stand's height and stay there -- the importer's handling of a `world` link
#            is the open question of ADR-0048, and training does not start without its answer
#   train    ENVS environments, ITER iterations; peak VRAM, RAM, swap from outside; the artifact
#   check    policy check of the artifact against the profile in the EXECUTION venv; contract_id
#            on both sides of the boundary
#   replay   the exported graph against the training venv's TorchScript on shared inputs
#   coded    the Gazebo world with the bridge; move_joints; recorded
#   learned  the same world and goal with one line of the profile swapped; recorded
#   diverge  six deliberate divergences of the artifact, each refused before `connect`
#   compare  tracking, steady-state error and time to goal from the two records
#
# Usage:
#   bash scripts/phase6-stage7.sh                            # all steps, in order
#   bash scripts/phase6-stage7.sh env plan mount             # the checks before any GPU hour
#   ENVS=1024 ITER=1500 bash scripts/phase6-stage7.sh train check replay
#   GUI=1 bash scripts/phase6-stage7.sh coded learned compare
#   REF=g1_upper_reach@0123456789abcdef bash scripts/phase6-stage7.sh learned
#
# Prerequisites: scripts/phase0.sh install isaac and exec; scripts/phase6-stage4.sh env passed
# once (the training venv takes the platform); scripts/phase4-probe.sh 0.5 (venv-own-lib:
# CycloneDDS 0.10.2 and the vendor's SDK) and scripts/phase4-stage5.sh env (unitree_hg and the
# bridge built into $HOME/phase4-stage5/ws).
#
# Writes outside $STAGE_ROOT: the mounted description under examples/robots/*/vendor (gitignored);
# `pip install --no-deps -e` of the platform and the task into the training venv, of the platform
# and the G1 package into venv-own-lib, and mcap, onnxruntime there; the Kit cache in $HOME.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line
ENV_PREFIX="${APP^^}"                      # variables the platform reads, built from the name
readonly ENV_PREFIX

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly ROBOT_NAME="g1_29dof_lock_waist_rev_1_0_mounted"
readonly ROBOT_DIR="$REPO_ROOT/examples/robots/$ROBOT_NAME"
readonly PROFILE="$ROBOT_DIR/profile.yaml"
readonly FETCH="$ROBOT_DIR/fetch_description.py"
readonly MOUNTED_URDF="$ROBOT_DIR/vendor/mounted/$ROBOT_NAME.urdf"
readonly MOUNT_HEIGHT="1.0"                # fetch_description.py MOUNT_HEIGHT, ADR-P6A item 1
readonly REGISTRY="$REPO_ROOT/examples/robots"
readonly TRAIN_SCENARIO="$REPO_ROOT/examples/scenarios/g1_mounted_train.yaml"
readonly SCENARIO="$REPO_ROOT/examples/scenarios/g1_mounted.yaml"
readonly ROBOT_ID="g1"                     # both scenarios, robots[0].id
readonly TASK_PACKAGE="$REPO_ROOT/examples/plugins/g1_upper_reach"
readonly G1_PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly TASK="g1_upper_reach"
readonly GAINS="$REPO_ROOT/scripts/phase6-gains-check.py"
readonly URDF_PROBE="$REPO_ROOT/scripts/phase6-probe-urdf.py"
readonly ONNX_PROBE="$REPO_ROOT/scripts/phase6-probe-onnx.py"
readonly METRICS="$REPO_ROOT/scripts/phase4-stage6-metrics.py"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"

TRAIN_ROOT="${TRAIN_ROOT:-$HOME/train-stack}"
STAGE_ROOT="${STAGE_ROOT:-$HOME/phase6-stage7}"
STAGE5_ROOT="${STAGE5_ROOT:-$HOME/phase4-stage5}"        # the colcon workspace of the bridge
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"           # venv-own-lib, the adapter's venv
ENVS="${ENVS:-512}"
ITER="${ITER:-1500}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-21600}"   # ADR-0005 allows six hours; this is the wall of the run
URDF_TIMEOUT="${URDF_TIMEOUT:-900}"
MOUNT_STEPS="${MOUNT_STEPS:-250}"         # physics steps the mounted robot is left to hang
HEARTBEAT="${HEARTBEAT:-60}"
SAMPLE="${SAMPLE:-5}"                     # seconds between memory samples
GUI="${GUI:-0}"
WORLD_WAIT="${WORLD_WAIT:-90}"            # seconds for Gazebo and the bridge to come up
SETTLE="${SETTLE:-5}"                     # seconds the arms hang before the run, as in training's reset
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"
readonly TRAIN_ROOT STAGE_ROOT STAGE5_ROOT PROBE_ROOT ENVS ITER SEED DEVICE TRAIN_TIMEOUT
readonly URDF_TIMEOUT MOUNT_STEPS HEARTBEAT SAMPLE GUI WORLD_WAIT SETTLE RUN_TIMEOUT
readonly TRAIN_VENV="$TRAIN_ROOT/venv"
readonly TRAIN_PY="$TRAIN_VENV/bin/python"
readonly TRAIN_CLI="$TRAIN_VENV/bin/$APP"
readonly ISAACLAB_DIR="$TRAIN_ROOT/IsaacLab"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly OWN_PY="$VENV_OWN/bin/python"
readonly OWN_CLI="$VENV_OWN/bin/$APP"
readonly WS="$STAGE5_ROOT/ws"
readonly RUN_HOME="$STAGE_ROOT/data"      # <PREFIX>_HOME: runs and the USD cache live here
readonly STORE="$STAGE_ROOT/store"
readonly PROFILES="$STAGE_ROOT/registry"  # copies of the profile; the repository's is not edited
readonly ALL_STEPS="env plan mount train check replay coded learned diverge compare"

[[ $ENVS =~ ^[1-9][0-9]*$ ]] || { printf 'ENVS=%s: нужно целое больше нуля\n' "$ENVS"; exit 2; }
[[ $ITER =~ ^[1-9][0-9]*$ ]] || { printf 'ITER=%s: нужно целое больше нуля\n' "$ITER"; exit 2; }

mkdir -p "$STAGE_ROOT" "$RUN_HOME" "$STORE" "$PROFILES"
LOG="$STAGE_ROOT/stage7-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

REF="${REF:-$(cat "$STAGE_ROOT/artifact.txt" 2>/dev/null)}"

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

# Nothing inherited but HOME and a plain PATH: a sourced ROS 2 in the calling shell must not
# decide whether training works (invariant 2), and `run` must not get the world's environment
# from here -- applying it is run's own business (ADR-0039). The data home and the store are
# pointed into the stage, so a run of this script leaves nothing in the user's own.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage7}" LANG=C.UTF-8 LC_ALL=C.UTF-8
    "TERM=${TERM:-dumb}" "${ENV_PREFIX}_HOME=$RUN_HOME" "${ENV_PREFIX}_STORE=$STORE"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

readonly WS_ENV="source '$ROS_SETUP' && source '$WS/install/setup.bash'"

STARTED=()
LAST_GROUP=""
start_group() {  # start_group <log> <command>
    setsid "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$2" > "$1" 2>&1 < /dev/null &
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

# Peak memory is sampled from outside the process under test: a number training reports about
# itself proves nothing about the machine.
PEAK_VRAM=0
PEAK_RAM=0
PEAK_SWAP=0
sample_memory() {
    local vram ram swap
    vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 || echo 0)"
    ram="$(awk '/MemAvailable/{a=$2} /MemTotal/{t=$2} END{printf "%d", (t-a)/1024}' /proc/meminfo)"
    swap="$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d", (t-f)/1024}' /proc/meminfo)"
    [[ $vram =~ ^[0-9]+$ ]] && (( vram > PEAK_VRAM )) && PEAK_VRAM=$vram
    (( ram > PEAK_RAM )) && PEAK_RAM=$ram
    (( swap > PEAK_SWAP )) && PEAK_SWAP=$swap
    return 0
}

# A result is not an exit (phase 6 probe, run 3): with a done-test the wait ends on the result
# and the child gets 60 s to exit on its own.
watch_group() {  # watch_group <pgid> <log> <timeout> [done-test]; child's status, 124 on timeout
    local pgid="$1" log="$2" timeout="$3" done_test="${4:-}" started last line status ticks=0
    local done_at=""
    started="$(date +%s)"; last="$started"
    PEAK_VRAM=0; PEAK_RAM=0; PEAK_SWAP=0
    info "лог процесса: $log (срок ${timeout} с)"
    while kill -0 "$pgid" 2>/dev/null; do
        sleep 1
        ticks=$(( ticks + 1 ))
        (( ticks % SAMPLE == 0 )) && sample_memory
        if [[ -n $done_test && -z $done_at ]] && eval "$done_test"; then
            done_at="$(date +%s)"
            info "итог получен через $(( done_at - started )) с; процессу 60 с на выход"
        fi
        if [[ -n $done_at ]] && (( $(date +%s) - done_at >= 60 )); then
            info "процесс не вышел за 60 с после итога — останавливается"
            stop_group "$pgid" 30
            wait "$pgid" 2>/dev/null
            return 0
        fi
        if (( $(date +%s) - last >= HEARTBEAT )); then
            last="$(date +%s)"
            line="$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 1 | cut -c1-140)"
            info "… $(( last - started )) с, VRAM ${PEAK_VRAM} МиБ, RAM ${PEAK_RAM} МиБ: ${line:-пусто}"
        fi
        if (( $(date +%s) - started >= timeout )); then
            info "срок ${timeout} с истёк — процесс останавливается; хвост лога:"
            tail -n 30 "$log" 2>/dev/null | sed 's/^/      /'
            stop_group "$pgid" 30
            wait "$pgid" 2>/dev/null
            return 124
        fi
    done
    wait "$pgid"; status=$?
    return "$status"
}

version_of() {  # version_of <python> <distribution>
    "${BASE_ENV[@]}" "$1" -c "import importlib.metadata as m; print(m.version('$2'))" 2>/dev/null \
        || echo нет
}

need_train_stack() {
    [[ -x $TRAIN_PY ]] && return 0
    record "стек обучения" отказ "нет $TRAIN_PY (scripts/phase0.sh install isaac)"
    return 1
}

need_adapter_venv() {
    [[ -x $OWN_CLI ]] && return 0
    record "окружение адаптера" отказ "нет $OWN_CLI (scripts/phase4-probe.sh 0.5, затем шаг env)"
    return 1
}

# The profile, copied with one line of `skills` rewritten -- invariant 4 as a file operation.
# The description stays where the copy points at it: the vendor directory is linked, not copied.
write_profile() {  # write_profile <registry dir> <policy-ref or empty>
    local target="$1/$ROBOT_NAME"
    mkdir -p "$target"
    ln -sfn "$ROBOT_DIR/vendor" "$target/vendor"
    "${BASE_ENV[@]}" "$EXEC_PY" - "$PROFILE" "$target/profile.yaml" "${2:-}" <<'PY'
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

# -- env -----------------------------------------------------------------------------------

step_env() {
    step "env: машина, версии, описание, установка платформы, задачи и адаптера"
    local out="$STAGE_ROOT/conditions.md" gpu ram swap cpu os commit dirty

    if clean "cd '$REPO_ROOT' && '$EXEC_PY' '$FETCH'" > "$STAGE_ROOT/fetch.log" 2>&1; then
        record "описание закреплённого G1" ok "$(tail -n 1 "$STAGE_ROOT/fetch.log")"
    else
        record "описание закреплённого G1" отказ "см. $STAGE_ROOT/fetch.log"
        tail -n 10 "$STAGE_ROOT/fetch.log" | sed 's/^/      /'
    fi

    if need_train_stack; then
        # The platform in the training venv: the same code computes contract_id on both sides
        # of the boundary (ADR-0044, item 3); the task is found there by its entry point.
        if clean "'$TRAIN_PY' -m pip install --no-deps -e '$REPO_ROOT' -e '$TASK_PACKAGE'" \
                > "$STAGE_ROOT/install-train.log" 2>&1; then
            record "платформа и задача в окружении обучения" ok "$(version_of "$TRAIN_PY" "$APP")"
        else
            record "платформа и задача в окружении обучения" отказ "см. $STAGE_ROOT/install-train.log"
            tail -n 20 "$STAGE_ROOT/install-train.log" | sed 's/^/      /'
        fi
        if clean "'$TRAIN_PY' -c 'import rclpy'" > /dev/null 2>&1; then
            record "rclpy в окружении обучения" отказ "импортируется — граница миров нарушена (ADR-0003)"
        else
            record "rclpy в окружении обучения" ok "не импортируется (инвариант 2)"
        fi
        if clean "'$TRAIN_CLI' task list" 2>/dev/null | grep -q "^$TASK "; then
            record "задача найдена точкой входа" ok "$TASK"
        else
            record "задача найдена точкой входа" отказ "$TASK не в группе задач"
        fi
    fi

    # The adapter's venv runs `run`: CycloneDDS 0.10.2 and the SDK are there (phase 4); what a
    # learned skill and a record add is onnxruntime and mcap (ADR-0047, ADR-0038).
    if need_adapter_venv; then
        if clean "'$OWN_PY' -m pip install -q --no-cache-dir --timeout 120 --retries 5 pydantic pyyaml typer packaging mcap numpy onnxruntime && '$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$G1_PACKAGE'" \
                > "$STAGE_ROOT/install-own.log" 2>&1; then
            record "окружение адаптера" ok "onnxruntime $(version_of "$OWN_PY" onnxruntime), mcap $(version_of "$OWN_PY" mcap)"
        else
            record "окружение адаптера" отказ "см. $STAGE_ROOT/install-own.log"
            tail -n 20 "$STAGE_ROOT/install-own.log" | sed 's/^/      /'
        fi
    fi
    # The execution venv starts the world: the G1 package's Gazebo twin is found there by its
    # entry point, and it must be this checkout's -- it knows the upper body (ADR-P6B).
    if clean "cd '$REPO_ROOT' && uv pip install --python '$EXEC_PY' --no-deps -e '$G1_PACKAGE'" \
            > "$STAGE_ROOT/install-exec.log" 2>&1; then
        record "пакет G1 в окружении исполнения" ok "$(version_of "$EXEC_PY" unitree-g1)"
    else
        record "пакет G1 в окружении исполнения" отказ "см. $STAGE_ROOT/install-exec.log"
    fi
    [[ -f $WS/install/setup.bash ]] \
        && record "мост G1 собран" ok "$WS" \
        || record "мост G1 собран" отказ "нет $WS/install/setup.bash: bash scripts/phase4-stage5.sh env"

    gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo нет)"
    ram="$(awk '/MemTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    swap="$(awk '/SwapTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    cpu="$(LC_ALL=C lscpu | awk -F: '/Model name/{gsub(/^ +/, "", $2); print $2; exit}')"
    os="$(. /etc/os-release && echo "$PRETTY_NAME"), ядро $(uname -r)"
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    dirty="$(git -C "$REPO_ROOT" status --porcelain | wc -l)"
    {
        echo "# Условия этапа 7 фазы 6 — $(date -Iseconds)"
        echo
        echo "| Что | Значение |"
        echo "|---|---|"
        echo "| **Условие прогона (ADR-P6A)** | таз закреплён описанием на ${MOUNT_HEIGHT} м, ноги без момента; баланс не моделируется |"
        echo "| Робот | $ROBOT_NAME (тело mode_machine 6, верх тела — 15 суставов) |"
        echo "| GPU, драйвер, память | $gpu |"
        echo "| CPU | $cpu |"
        echo "| RAM / своп | $ram / $swap |"
        echo "| ОС | $os |"
        echo "| Коммит платформы | $commit (незакоммиченных файлов: $dirty) |"
        echo "| isaacsim | $(version_of "$TRAIN_PY" isaacsim) |"
        echo "| isaaclab | $(version_of "$TRAIN_PY" isaaclab) |"
        echo "| rsl-rl-lib | $(version_of "$TRAIN_PY" rsl-rl-lib) |"
        echo "| torch | $(version_of "$TRAIN_PY" torch) |"
        echo "| платформа (обучение / исполнение / адаптер) | $(version_of "$TRAIN_PY" "$APP") / $(version_of "$EXEC_PY" "$APP") / $(version_of "$OWN_PY" "$APP") |"
        echo "| onnxruntime (адаптер) | $(version_of "$OWN_PY" onnxruntime) |"
        echo "| cyclonedds (адаптер) | $(version_of "$OWN_PY" cyclonedds) |"
        echo "| Gazebo | $(command -v gz >/dev/null && gz sim --versions 2>/dev/null | head -n 1 || echo нет) |"
        echo "| сред / итераций / зерно | $ENVS / $ITER / $SEED |"
    } > "$out"
    record "условия" факт "$out"
}

# -- plan ----------------------------------------------------------------------------------

step_plan() {
    step "plan: что выводится из контракта — до первой секунды GPU"
    local spec="$STAGE_ROOT/plan.txt"

    if clean "'$EXEC_CLI' validate '$SCENARIO' --registry '$REGISTRY'" > "$STAGE_ROOT/validate-gazebo.txt" 2>&1; then
        record "validate сцены исполнения" ok "$(grep -c 'внимание' "$STAGE_ROOT/validate-gazebo.txt") предупр. (ожидается 1: ноги вне контракта)"
    else
        record "validate сцены исполнения" отказ "см. $STAGE_ROOT/validate-gazebo.txt"
        sed 's/^/      /' "$STAGE_ROOT/validate-gazebo.txt"
    fi

    # ADR-P6A, item 5: the drive law is stable where it is computed, on the inertias both
    # worlds actually have -- the description's, without a rotor.
    if clean "'$EXEC_PY' '$GAINS' '$PROFILE'" > "$STAGE_ROOT/gains.txt" 2>&1; then
        record "коэффициенты устойчивы на 500 Гц" ok "см. $STAGE_ROOT/gains.txt"
    else
        record "коэффициенты устойчивы на 500 Гц" отказ "см. $STAGE_ROOT/gains.txt"
    fi
    sed 's/^/      /' "$STAGE_ROOT/gains.txt"

    # The training scene is validated where its task is installed: the training venv.
    need_train_stack || return
    if clean "'$TRAIN_CLI' validate '$TRAIN_SCENARIO' --registry '$REGISTRY'" \
            > "$STAGE_ROOT/validate-train.txt" 2>&1; then
        record "validate сцены обучения" ok "ошибок нет"
    else
        record "validate сцены обучения" отказ "см. $STAGE_ROOT/validate-train.txt"
        sed 's/^/      /' "$STAGE_ROOT/validate-train.txt"
    fi
    if clean "'$TRAIN_CLI' train env '$TRAIN_SCENARIO'" > "$STAGE_ROOT/env-check.txt" 2>&1; then
        record "окружение обучения" ok "требования бэкенда выполнены"
    else
        record "окружение обучения" отказ "см. $STAGE_ROOT/env-check.txt"
        sed 's/^/      /' "$STAGE_ROOT/env-check.txt"
    fi

    if clean "'$TRAIN_CLI' train run '$TRAIN_SCENARIO' --registry '$REGISTRY' --dry-run" > "$spec" 2>&1; then
        record "сухой прогон" факт "$spec"
        sed 's/^/      /' "$spec"
    else
        record "сухой прогон" отказ "см. $spec"
        sed 's/^/      /' "$spec"
        return
    fi

    # The derivation, checked against numbers written here rather than against the reader that
    # printed it: 500 Hz physics, 50 Hz policy, 15 actions, 60 observations, a fixed base.
    local dt decim obs act base
    dt="$(grep -oP 'физика \K[0-9.]+' "$spec")"
    decim="$(grep -oP 'decimation \K[0-9]+' "$spec")"
    obs="$(grep -oP 'наблюдение.*ширина \K[0-9]+' "$spec")"
    act="$(grep -oP 'действие.*ширина \K[0-9]+' "$spec")"
    base="$(grep -oP 'база\s+\K.*' "$spec")"
    if [[ $dt == "500" && $decim == "10" && $obs == "60" && $act == "15" && $base == "закреплена описанием" ]]; then
        record "вывод среды" ok "физика 500 Гц, decimation 10, действие 15, наблюдение 60, база закреплена"
    else
        record "вывод среды" отказ "физика $dt, decimation $decim, действие $act, наблюдение $obs, база $base"
    fi
}

# -- mount ---------------------------------------------------------------------------------

# ADR-0048 left open how Isaac's URDF importer reads a `world` link under fix_base. For the arm
# it changed nothing visible; for the G1 it decides the height of the pelvis: kept, it is the
# stand's; dropped, the pelvis is bolted where the articulation stands and the arms reach into
# the floor. The probe converts the copy the training process would convert (meshes anchored,
# no control block) and reads where Isaac put the bodies -- after the reset, and after the
# legs have hung for MOUNT_STEPS steps.
step_mount() {
    step "mount: высота таза в Isaac после сброса и после $MOUNT_STEPS шагов (ADR-P6A, ADR-0048)"
    need_train_stack || return
    [[ -x $ISAACLAB_DIR/isaaclab.sh ]] || { record "крепление в Isaac" отказ "нет $ISAACLAB_DIR/isaaclab.sh"; return; }
    [[ -f $MOUNTED_URDF ]] || { record "крепление в Isaac" отказ "нет $MOUNTED_URDF (шаг env)"; return; }

    local copy="$STAGE_ROOT/mount/description.urdf" json="$STAGE_ROOT/mount/facts.json"
    mkdir -p "$STAGE_ROOT/mount"
    rm -f "$json"
    # The same copy training converts: train.spec.description_text, meshes anchored.
    if ! "${BASE_ENV[@]}" "$TRAIN_PY" - "$PROFILE" "$copy" <<'PY'
import sys
from pathlib import Path
import yaml
from heron.core.models import RobotProfile
from heron.train.spec import description_text
profile_path, target = Path(sys.argv[1]), Path(sys.argv[2])
profile = RobotProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
source = profile_path.parent / profile.description.urdf
target.write_text(description_text(profile, source), encoding="utf-8")
print(f"копия описания: {target}")
PY
    then
        record "крепление в Isaac" отказ "копия описания не записана"
        return
    fi

    start_group "$STAGE_ROOT/mount/probe.log" "$(printf "cd %q && source %q && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p %q --headless --urdf %q --usd-dir %q --merge-fixed-joints no --steps %q --out %q" \
        "$ISAACLAB_DIR" "$TRAIN_VENV/bin/activate" "$URDF_PROBE" "$copy" "$STAGE_ROOT/mount/usd" "$MOUNT_STEPS" "$json")"
    watch_group "$LAST_GROUP" "$STAGE_ROOT/mount/probe.log" "$URDF_TIMEOUT" "[[ -s '$json' ]]"
    if [[ ! -s $json ]]; then
        record "крепление в Isaac" отказ "факты не записаны; см. $STAGE_ROOT/mount/probe.log"
        tail -n 20 "$STAGE_ROOT/mount/probe.log" | sed 's/^/      /'
        return
    fi

    local verdict
    verdict="$("${BASE_ENV[@]}" "$TRAIN_PY" - "$json" "$MOUNT_HEIGHT" <<'PY'
import json, sys
facts, height = json.load(open(sys.argv[1], encoding="utf-8")), float(sys.argv[2])
names = facts["body_names"]
pelvis = names.index("pelvis")
before = facts["body_pos_w"][pelvis][2]
after = facts.get("body_pos_w_after", facts["body_pos_w"])[pelvis][2]
feet = [facts.get("body_pos_w_after", facts["body_pos_w"])[names.index(n)][2]
        for n in ("left_ankle_roll_link", "right_ankle_roll_link") if n in names]
ok = abs(before - height) < 0.01 and abs(after - height) < 0.01
print(("ok" if ok else "отказ") + f"|таз {before:.3f} м после сброса, {after:.3f} м после шагов "
      f"(ожидается {height:g}); стопы {', '.join(f'{z:.3f}' for z in feet) or '—'} м; "
      f"корень {facts['root_pos_w']}; суставов {len(facts['joint_names'])}")
PY
)"
    record "крепление в Isaac" "${verdict%%|*}" "${verdict#*|}"
}

# -- train ---------------------------------------------------------------------------------

step_train() {
    step "train: $ENVS сред, $ITER итераций (порог ADR-0005 на задаче G1)"
    need_train_stack || return
    local out="$STAGE_ROOT/train.log" started elapsed status

    started="$(date +%s)"
    start_group "$out" "$(printf '%q train run %q --registry %q --envs %q --iterations %q --seed %q --device %q' \
        "$TRAIN_CLI" "$TRAIN_SCENARIO" "$REGISTRY" "$ENVS" "$ITER" "$SEED" "$DEVICE")"
    watch_group "$LAST_GROUP" "$out" "$TRAIN_TIMEOUT"; status=$?
    elapsed=$(( $(date +%s) - started ))

    if (( status != 0 )); then
        record "обучение" отказ "код $status за ${elapsed} с; см. $out"
        tail -n 30 "$out" | sed 's/^/      /'
        return
    fi
    record "обучение" ok "${elapsed} с (${ENVS} сред, ${ITER} итераций)"

    local vram_ok ram_ok time_ok
    (( PEAK_VRAM <= 7168 )) && vram_ok=ok || vram_ok=отказ
    (( PEAK_RAM <= 13312 )) && ram_ok=ok || ram_ok=отказ
    (( elapsed <= 21600 )) && time_ok=ok || time_ok=отказ
    record "порог: пик VRAM" "$vram_ok" "${PEAK_VRAM} МиБ (предел 7168)"
    record "порог: пик RAM" "$ram_ok" "${PEAK_RAM} МиБ (предел 13312)"
    record "порог: своп" факт "${PEAK_SWAP} МиБ"
    record "порог: время" "$time_ok" "${elapsed} с (предел 21600)"
    (( ENVS >= 512 )) && record "порог: сред" ok "$ENVS (не меньше 512)" \
        || record "порог: сред" отказ "$ENVS меньше 512"

    local ref
    ref="$(grep -oP 'артефакт\s+\K\S+@\S+' "$out" | tail -n 1)"
    if [[ -n $ref ]]; then
        printf '%s\n' "$ref" > "$STAGE_ROOT/artifact.txt"
        REF="$ref"
        record "артефакт" ok "$ref"
    else
        record "артефакт" отказ "ссылка не напечатана; см. $out"
    fi
}

# -- check ---------------------------------------------------------------------------------

step_check() {
    step "check: артефакт против профиля, в окружении ИСПОЛНЕНИЯ"
    [[ -n $REF ]] || { record "проверка артефакта" отказ "нет ссылки (шаг train или REF=...)"; return; }

    if clean "'$EXEC_CLI' policy show '$REF' --store '$STORE'" > "$STAGE_ROOT/policy-show.txt" 2>&1; then
        record "policy show" факт "$STAGE_ROOT/policy-show.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-show.txt"
    else
        record "policy show" отказ "см. $STAGE_ROOT/policy-show.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-show.txt"
        return
    fi
    if clean "'$EXEC_CLI' policy check '$REF' '$PROFILE' --store '$STORE'" \
            > "$STAGE_ROOT/policy-check.txt" 2>&1; then
        record "policy check против профиля" ok "замечаний нет"
    else
        record "policy check против профиля" отказ "см. $STAGE_ROOT/policy-check.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-check.txt"
    fi

    local trained executed
    trained="$(grep -oP 'контракт\s+\K[0-9a-f]{16}' "$STAGE_ROOT/policy-show.txt" | head -n 1)"
    executed="$(clean "'$EXEC_CLI' contract id '$PROFILE'" 2>/dev/null | tr -d '[:space:]')"
    if [[ -n $trained && $trained == "$executed" ]]; then
        record "contract_id по обе стороны границы" ok "$trained"
    else
        record "contract_id по обе стороны границы" отказ "обучение $trained, исполнение $executed"
    fi
}

# -- replay --------------------------------------------------------------------------------

step_replay() {
    step "replay: сеть читается исполняющей стороной так же, как обучающей"
    [[ -n $REF ]] || { record "паритет сети" отказ "нет ссылки на артефакт"; return; }
    local dir="$STORE/policies/${REF%@*}/${REF#*@}"
    [[ -d $dir ]] || { record "паритет сети" отказ "нет каталога $dir"; return; }

    local describe="$STAGE_ROOT/onnx.json" inputs="$STAGE_ROOT/inputs.npz" run_dir pt
    if ! clean "'$EXEC_PY' '$ONNX_PROBE' describe --onnx '$dir/policy.onnx' --out '$describe'"; then
        record "паритет сети" отказ "модель не читается onnxruntime окружения исполнения"
        return
    fi
    record "сигнатура сети" факт "$describe"
    run_dir="$(clean "'$EXEC_PY' -c \"import json;print(json.load(open('$dir/train.json'))['run'])\"" 2>/dev/null)"
    pt="$(find "$run_dir" -name 'policy.pt' 2>/dev/null | head -n 1)"
    if [[ -z $pt ]]; then
        record "паритет TorchScript и ONNX" пропуск "policy.pt в прогоне не найден"
        return
    fi
    if ! clean "'$TRAIN_PY' '$ONNX_PROBE' torch --jit '$pt' --describe '$describe' --seed '$SEED' --out '$inputs'"; then
        record "паритет TorchScript и ONNX" отказ "policy.pt не исполнился в окружении обучения"
        return
    fi
    if clean "'$EXEC_PY' '$ONNX_PROBE' compare --onnx '$dir/policy.onnx' --inputs '$inputs'"; then
        record "паритет TorchScript и ONNX" ok "расхождение ниже допуска"
    else
        record "паритет TorchScript и ONNX" отказ "расхождение выше допуска (числа выше)"
    fi
}

# -- coded, learned: the same world twice -------------------------------------------------

# A world through the bridge, the arms left to hang, a run against it with a record, the world
# down again. Readiness is the bridge's spawner reporting it active in the launch log; the run's
# own link timeout (ADR-0016) is what then says whether LowState really flows.
# The spawner prints "Configured and activated " and the name with a bold escape between them
# (controller_manager/spawner.py, jazzy): the pattern allows escapes, not only a space.
bridge_active() {  # bridge_active <launch log>
    grep -qE 'Configured and activated (.\[[0-9;]*m)*low_level_bridge' "$1" 2>/dev/null
}

run_against_world() {  # run_against_world <name> <registry>
    local name="$1" registry="$2"
    local dir="$STAGE_ROOT/$name" flags="--headless --no-rviz" waited=0 code
    rm -rf "$dir"; mkdir -p "$dir"
    (( GUI )) && flags=""
    need_adapter_venv || return 1
    [[ -f $WS/install/setup.bash ]] || { record "$name: мир" отказ "нет $WS (scripts/phase4-stage5.sh env)"; return 1; }

    start_group "$dir/world.log" "$WS_ENV && cd '$REPO_ROOT' && exec '$EXEC_CLI' sim run '$SCENARIO' --into '$dir/gen' --registry '$registry' $flags"
    local world="$LAST_GROUP"
    while (( waited < WORLD_WAIT )); do
        sleep 2; waited=$(( waited + 2 ))
        bridge_active "$dir/world.log" && break
        kill -0 "$world" 2>/dev/null || break
    done
    if ! bridge_active "$dir/world.log"; then
        record "$name: мир" отказ "мост не активирован за ${WORLD_WAIT} с; см. $dir/world.log"
        tail -n 30 "$dir/world.log" | sed 's/^/      /'
        stop_group "$world" 30
        return 1
    fi
    record "$name: мир" ok "мост активен через ${waited} с; руки висят ${SETTLE} с"
    sleep "$SETTLE"

    clean "cd '$REPO_ROOT' && timeout '$RUN_TIMEOUT' '$OWN_CLI' run '$SCENARIO' --robot '$ROBOT_ID' --registry '$registry' --record '$dir/run.mcap'" \
        > "$dir/run.txt" 2>&1
    code=$?
    stop_group "$world" 30
    sed 's/^/      /' "$dir/run.txt"

    if [[ ! -s $dir/run.mcap ]]; then
        record "$name: прогон" "отказ, код $code" "записи нет; см. $dir/run.txt"
        return 1
    fi
    record "$name: прогон" "$( (( code == 0 )) && echo ok || echo "код $code")" "запись $dir/run.mcap"
    return 0
}

step_coded() {
    step "coded: move_joints на закреплённом G1 в Gazebo через мост"
    write_profile "$PROFILES/coded" ""
    run_against_world coded "$PROFILES/coded"
}

step_learned() {
    step "learned: та же сцена, та же цель, одна строка профиля другая"
    [[ -n $REF ]] || { record "learned" отказ "нет ссылки на артефакт (шаг train или REF=...)"; return; }
    write_profile "$PROFILES/coded" ""
    write_profile "$PROFILES/learned" "$REF"

    # The one line, shown as a diff: invariant 4 as evidence.
    diff -u "$PROFILES/coded/$ROBOT_NAME/profile.yaml" "$PROFILES/learned/$ROBOT_NAME/profile.yaml" \
        > "$STAGE_ROOT/one-line.diff"
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

    if clean "'$EXEC_CLI' policy check '$REF' '$PROFILES/learned/$ROBOT_NAME/profile.yaml' --store '$STORE'" \
            > "$STAGE_ROOT/policy-check-learned.txt" 2>&1; then
        record "learned: policy check" ok "$REF"
    else
        record "learned: policy check" отказ "см. $STAGE_ROOT/policy-check-learned.txt"
    fi
    run_against_world learned "$PROFILES/learned"
}

# -- diverge -------------------------------------------------------------------------------

# Criterion 4 of the phase on the second robot: a divergence introduced into the artifact is
# refused before the run, by code. Each is made in a copy of the store.
step_diverge() {
    step "diverge: внесённые расхождения артефакта — отказ до подключения"
    [[ -n $REF ]] || { record "diverge" отказ "нет ссылки на артефакт"; return; }
    write_profile "$PROFILES/learned" "$REF"
    local name="${REF%@*}" ident="${REF#*@}" caught=0 total=0
    local source="$STORE/policies/$name/$ident" field expected pair
    [[ -d $source ]] || { record "diverge" отказ "нет каталога $source"; return; }

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
    onnx = directory / "policy.onnx"
    onnx.write_bytes(onnx.read_bytes() + b"\x00")
if field != "network":
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
        local out="$STAGE_ROOT/diverge/$field.txt"
        clean "'$EXEC_CLI' policy check '$REF' '$PROFILES/learned/$ROBOT_NAME/profile.yaml' --store '$broken'" \
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
    local out="$STAGE_ROOT/compare.md" files=() label
    for label in coded learned; do
        [[ -s $STAGE_ROOT/$label/run.mcap ]] || { record "метрики: $label" отказ "нет записи"; continue; }
        if clean "'$EXEC_PY' '$METRICS' run '$STAGE_ROOT/$label/run.mcap' --label '$label' --out '$STAGE_ROOT/$label/metrics.json'" \
                > "$STAGE_ROOT/$label/metrics.log" 2>&1; then
            files+=("$STAGE_ROOT/$label/metrics.json")
        else
            record "метрики: $label" отказ "см. $STAGE_ROOT/$label/metrics.log"
            sed 's/^/      /' "$STAGE_ROOT/$label/metrics.log"
        fi
    done
    (( ${#files[@]} == 2 )) || { record "сравнение" отказ "метрики посчитаны не для обоих прогонов"; return; }
    if clean "'$EXEC_PY' '$METRICS' table ${files[*]}" > "$out" 2>&1; then
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
    case " $ALL_STEPS " in
        *" $s "*) ;;
        *) printf 'неизвестный шаг: %s (доступны: %s)\n' "$s" "$ALL_STEPS"; exit 2 ;;
    esac
done

step "этап 7 фазы 6: ${steps[*]} — лог $LOG"
info "платформа $(git -C "$REPO_ROOT" rev-parse --short HEAD), $APP; TRAIN_ROOT=$TRAIN_ROOT"
info "данные прогонов: $RUN_HOME; хранилище артефактов: $STORE; артефакт ${REF:-не задан}"
info "условие прогона: таз закреплён описанием на ${MOUNT_HEIGHT} м, ноги без момента (ADR-P6A)"
for s in "${steps[@]}"; do "step_$s"; done

step "итог"
for line in "${RESULTS[@]}"; do
    IFS='|' read -r check status note <<< "$line"
    printf '    %-40s %-7s %s\n' "$check" "$status" "$note"
done
info "Прислать: $LOG, $STAGE_ROOT/conditions.md, $STAGE_ROOT/plan.txt, $STAGE_ROOT/gains.txt,"
info "$STAGE_ROOT/mount/facts.json, $STAGE_ROOT/policy-show.txt, $STAGE_ROOT/one-line.diff, $STAGE_ROOT/compare.md"
