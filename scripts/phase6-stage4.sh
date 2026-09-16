#!/usr/bin/env bash
#
# Phase 6, stage 4: train the example arm's policy in the environment the platform derives
# from its profile and scenario (docs/plans/phase6-learned-skill.md, "Этап 4"; ADR-0044,
# ADR-0046).
#
# Built like scripts/phase4-stage5.sh and on the same rule: every claim stands on a
# confirmation taken by something other than the thing under test. The derived environment is
# read off spec.json, not off the dry run that printed it; the artifact is checked by
# `policy check` against the profile, not by the exporter that wrote it; the threshold of
# ADR-0005 is measured off nvidia-smi and /proc, not off what training reports about itself.
# Nothing stops at the first failure.
#
#   env     versions and machine into conditions.md; the platform installed into the training
#           venv; the task package installed there; rclpy must NOT import (invariant 2)
#   plan    validate; train env; train run --dry-run -- the derivation, and the refusals a
#           broken contract earns, before a second of GPU time
#   train   train run at ENVS for ITER iterations, with peak VRAM, RAM and swap sampled from
#           outside; the artifact exported
#   check   policy check of the artifact against the profile, in the EXECUTION venv -- the
#           contract_id computed by the side that will run it
#   replay  the exported graph against the training venv's TorchScript on shared inputs, and
#           the policy's own tracking in Isaac: the numbers stage 6 compares Gazebo with
#
# Usage:
#   bash scripts/phase6-stage4.sh                      # all steps, in order
#   bash scripts/phase6-stage4.sh env plan             # selected steps
#   ENVS=1024 ITER=1500 bash scripts/phase6-stage4.sh train check
#
# Prerequisites: scripts/phase0.sh install isaac (TRAIN_ROOT, default ~/train-stack) and exec;
# scripts/phase6-probe.sh passed (ADR-0005 threshold by surrogate, experiment 0002).
#
# Writes outside $STAGE_ROOT: the platform and the task package installed into the training
# venv; runs under the platform's data home (the <PREFIX>_HOME variable, pointed at
# $STAGE_ROOT below); the Kit cache in $HOME.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line
ENV_PREFIX="${APP^^}"                      # variables the platform reads, built from the name
readonly ENV_PREFIX

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="$REPO_ROOT/examples/scenarios/arm6_train.yaml"
readonly PROFILE="$REPO_ROOT/examples/robots/arm6.yaml"
readonly REGISTRY="$REPO_ROOT/examples/robots"
readonly TASK_PACKAGE="$REPO_ROOT/examples/plugins/arm6_reach"
readonly TASK="arm6_reach"
readonly ONNX_PROBE="$REPO_ROOT/scripts/phase6-probe-onnx.py"

TRAIN_ROOT="${TRAIN_ROOT:-$HOME/train-stack}"
STAGE_ROOT="${STAGE_ROOT:-$HOME/phase6-stage4}"
ENVS="${ENVS:-512}"
ITER="${ITER:-1000}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-21600}"   # ADR-0005 allows six hours; this is the wall of the run
HEARTBEAT="${HEARTBEAT:-60}"
SAMPLE="${SAMPLE:-5}"                     # seconds between memory samples
readonly TRAIN_ROOT STAGE_ROOT ENVS ITER SEED DEVICE TRAIN_TIMEOUT HEARTBEAT SAMPLE
readonly TRAIN_VENV="$TRAIN_ROOT/venv"
readonly TRAIN_PY="$TRAIN_VENV/bin/python"
readonly TRAIN_CLI="$TRAIN_VENV/bin/$APP"
readonly RUN_HOME="$STAGE_ROOT/data"      # <PREFIX>_HOME: runs and the USD cache live here
readonly STORE="$STAGE_ROOT/store"
readonly ALL_STEPS="env plan train check replay"

[[ $ENVS =~ ^[1-9][0-9]*$ ]] || { printf 'ENVS=%s: нужно целое больше нуля\n' "$ENVS"; exit 2; }
[[ $ITER =~ ^[1-9][0-9]*$ ]] || { printf 'ITER=%s: нужно целое больше нуля\n' "$ITER"; exit 2; }

mkdir -p "$STAGE_ROOT" "$RUN_HOME" "$STORE"
LOG="$STAGE_ROOT/stage4-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

# Nothing inherited but HOME and a plain PATH: a sourced ROS 2 in the calling shell must not
# decide whether training works -- invariant 2 says it is not there. The platform's data home
# is pointed into the stage, so a run of this script leaves nothing in the user's own.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage}" LANG=C.UTF-8 LC_ALL=C.UTF-8
    "TERM=${TERM:-dumb}" "${ENV_PREFIX}_HOME=$RUN_HOME" "${ENV_PREFIX}_STORE=$STORE"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

clean() { "${BASE_ENV[@]}" bash --noprofile --norc -c "$1"; }

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
    for pgid in "${STARTED[@]}"; do stop_group "$pgid" 10 >/dev/null 2>&1; done
}
trap cleanup EXIT

# Peak memory is sampled from outside the process under test, for the reason the whole script
# is built this way: a number training reports about itself proves nothing about the machine.
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

watch_group() {  # watch_group <pgid> <log> <timeout>; child's status, 124 on timeout
    local pgid="$1" log="$2" timeout="$3" started last line status ticks=0
    started="$(date +%s)"; last="$started"
    PEAK_VRAM=0; PEAK_RAM=0; PEAK_SWAP=0
    info "лог процесса: $log (срок ${timeout} с)"
    while kill -0 "$pgid" 2>/dev/null; do
        sleep 1
        ticks=$(( ticks + 1 ))
        (( ticks % SAMPLE == 0 )) && sample_memory
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

# -- env -----------------------------------------------------------------------------------

step_env() {
    step "env: машина, версии, установка платформы и задачи в окружение обучения"
    need_train_stack || return
    local out="$STAGE_ROOT/conditions.md" gpu ram swap cpu os commit dirty

    gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo нет)"
    ram="$(awk '/MemTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    swap="$(awk '/SwapTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    cpu="$(LC_ALL=C lscpu | awk -F: '/Model name/{gsub(/^ +/, "", $2); print $2; exit}')"
    os="$(. /etc/os-release && echo "$PRETTY_NAME"), ядро $(uname -r)"
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    dirty="$(git -C "$REPO_ROOT" status --porcelain | wc -l)"

    # The platform in the training venv: the same code computes contract_id on both sides
    # of the boundary, which is the whole reason it is installed here (ADR-0044, item 3).
    if clean "$TRAIN_PY -m pip install --no-deps -e $REPO_ROOT -e $TASK_PACKAGE" \
            > "$STAGE_ROOT/install.log" 2>&1; then
        record "платформа и задача в окружении обучения" ok "$(version_of "$TRAIN_PY" "$APP")"
    else
        record "платформа и задача в окружении обучения" отказ "см. $STAGE_ROOT/install.log"
        tail -n 20 "$STAGE_ROOT/install.log" | sed 's/^/      /'
    fi

    if clean "$TRAIN_PY -c 'import rclpy'" > /dev/null 2>&1; then
        record "rclpy в окружении обучения" отказ "импортируется — граница миров нарушена (ADR-0003)"
    else
        record "rclpy в окружении обучения" ok "не импортируется (инвариант 2)"
    fi

    if clean "$TRAIN_CLI task list" 2>/dev/null | grep -q "$TASK"; then
        record "задача найдена точкой входа" ok "$TASK"
    else
        record "задача найдена точкой входа" отказ "$TASK не в группе задач"
    fi

    {
        echo "# Условия этапа 4 фазы 6 — $(date -Iseconds)"
        echo
        echo "| Что | Значение |"
        echo "|---|---|"
        echo "| GPU, драйвер, память | $gpu |"
        echo "| CPU | $cpu |"
        echo "| RAM / своп | $ram / $swap |"
        echo "| ОС | $os |"
        echo "| Коммит платформы | $commit (незакоммиченных файлов: $dirty) |"
        echo "| isaacsim | $(version_of "$TRAIN_PY" isaacsim) |"
        echo "| isaaclab | $(version_of "$TRAIN_PY" isaaclab) |"
        echo "| rsl-rl-lib | $(version_of "$TRAIN_PY" rsl-rl-lib) |"
        echo "| torch | $(version_of "$TRAIN_PY" torch) |"
        echo "| платформа (обучение) | $(version_of "$TRAIN_PY" "$APP") |"
        echo "| платформа (исполнение) | $(version_of "$EXEC_PY" "$APP") |"
        echo "| onnxruntime (исполнение) | $(version_of "$EXEC_PY" onnxruntime) |"
        echo "| сред / итераций / зерно | $ENVS / $ITER / $SEED |"
    } > "$out"
    record "условия" факт "$out"
}

# -- plan ----------------------------------------------------------------------------------

step_plan() {
    step "plan: что выводится из контракта, и чего платформа не выводит"
    local spec="$STAGE_ROOT/plan.txt"

    if clean "$TRAIN_CLI validate $SCENARIO --registry $REGISTRY" > "$STAGE_ROOT/validate.txt" 2>&1; then
        record "validate сцены обучения" ok "ошибок нет"
    else
        record "validate сцены обучения" отказ "см. $STAGE_ROOT/validate.txt"
        sed 's/^/      /' "$STAGE_ROOT/validate.txt"
    fi

    if clean "$TRAIN_CLI train env $SCENARIO" > "$STAGE_ROOT/env-check.txt" 2>&1; then
        record "окружение обучения" ok "требования бэкенда выполнены"
    else
        record "окружение обучения" отказ "см. $STAGE_ROOT/env-check.txt"
        sed 's/^/      /' "$STAGE_ROOT/env-check.txt"
    fi

    if clean "$TRAIN_CLI train run $SCENARIO --registry $REGISTRY --dry-run" > "$spec" 2>&1; then
        record "сухой прогон" факт "$spec"
        sed 's/^/      /' "$spec"
    else
        record "сухой прогон" отказ "см. $spec"
        sed 's/^/      /' "$spec"
        return
    fi

    # The derivation, checked against the contract by a different reader than the one that
    # printed it: what a dry run shows and what the process will read are the same numbers.
    local dt decim obs
    dt="$(grep -oP 'физика \K[0-9.]+' "$spec")"
    decim="$(grep -oP 'decimation \K[0-9]+' "$spec")"
    obs="$(grep -oP 'ширина \K[0-9]+' "$spec" | tail -n 1)"
    if [[ $dt == "100" && $decim == "2" && $obs == "24" ]]; then
        record "вывод среды" ok "физика 100 Гц, decimation 2, наблюдение 24"
    else
        record "вывод среды" отказ "физика $dt, decimation $decim, наблюдение $obs"
    fi

    # A contract an environment cannot be derived from is refused before anything starts.
    # The copy is broken deliberately: the gains that close a position into a torque are
    # removed, and that is exactly what ADR-0045 says a policy may not be trained without.
    local broken="$STAGE_ROOT/arm6-no-gains.yaml"
    "${BASE_ENV[@]}" "$TRAIN_PY" - "$PROFILE" "$broken" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
data["contract"]["action"].pop("kp", None)
data["contract"]["action"].pop("kd", None)
data["contract"]["revision"] += 1
yaml.safe_dump(data, open(sys.argv[2], "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PY
    local broken_dir="$STAGE_ROOT/broken-registry"
    mkdir -p "$broken_dir"
    cp "$broken" "$broken_dir/arm6.yaml"
    ln -sfn "$REGISTRY/urdf" "$broken_dir/urdf"
    if clean "$TRAIN_CLI validate $SCENARIO --registry $broken_dir" 2>&1 \
            | grep -q 'train.gains_missing'; then
        record "контракт без kp/kd отклонён" ok "train.gains_missing до запуска"
    else
        record "контракт без kp/kd отклонён" отказ "валидатор промолчал"
    fi
}

# -- train ---------------------------------------------------------------------------------

step_train() {
    step "train: обучение $ENVS сред, $ITER итераций (порог ADR-0005 на целевой задаче)"
    need_train_stack || return
    local out="$STAGE_ROOT/train.log" started elapsed status

    started="$(date +%s)"
    start_group "$out" "$(printf '%q train run %q --registry %q --envs %q --iterations %q --seed %q --device %q' \
        "$TRAIN_CLI" "$SCENARIO" "$REGISTRY" "$ENVS" "$ITER" "$SEED" "$DEVICE")"
    watch_group "$LAST_GROUP" "$out" "$TRAIN_TIMEOUT"; status=$?
    elapsed=$(( $(date +%s) - started ))

    if (( status != 0 )); then
        record "обучение" отказ "код $status за ${elapsed} с; см. $out"
        tail -n 30 "$out" | sed 's/^/      /'
        return
    fi
    record "обучение" ok "${elapsed} с (${ENVS} сред, ${ITER} итераций)"

    # The threshold of ADR-0005, measured on the target task rather than by surrogate.
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
        record "артефакт" ok "$ref"
    else
        record "артефакт" отказ "ссылка не напечатана; см. $out"
    fi
}

# -- check ---------------------------------------------------------------------------------

step_check() {
    step "check: артефакт против профиля, в окружении ИСПОЛНЕНИЯ"
    local ref
    ref="$(cat "$STAGE_ROOT/artifact.txt" 2>/dev/null)"
    if [[ -z $ref ]]; then
        record "проверка артефакта" отказ "нет $STAGE_ROOT/artifact.txt (шаг train)"
        return
    fi

    # Deliberately the execution venv: a contract_id computed by the side that will run the
    # policy is the only one whose agreement means anything (ADR-0046).
    if clean "$EXEC_CLI policy show $ref --store $STORE" > "$STAGE_ROOT/policy-show.txt" 2>&1; then
        record "policy show" факт "$STAGE_ROOT/policy-show.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-show.txt"
    else
        record "policy show" отказ "см. $STAGE_ROOT/policy-show.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-show.txt"
        return
    fi

    if clean "$EXEC_CLI policy check $ref $PROFILE --store $STORE" \
            > "$STAGE_ROOT/policy-check.txt" 2>&1; then
        record "policy check против профиля" ok "замечаний нет"
    else
        record "policy check против профиля" отказ "см. $STAGE_ROOT/policy-check.txt"
        sed 's/^/      /' "$STAGE_ROOT/policy-check.txt"
    fi

    # The identifier the training side wrote and the one the execution side computes.
    local trained executed
    trained="$(grep -oP 'контракт\s+\K[0-9a-f]{16}' "$STAGE_ROOT/policy-show.txt" | head -n 1)"
    executed="$(clean "$EXEC_CLI contract id $PROFILE" 2>/dev/null | tr -d '[:space:]')"
    if [[ -n $trained && $trained == "$executed" ]]; then
        record "contract_id по обе стороны границы" ok "$trained"
    else
        record "contract_id по обе стороны границы" отказ "обучение $trained, исполнение $executed"
    fi
}

# -- replay --------------------------------------------------------------------------------

step_replay() {
    step "replay: сеть читается исполняющей стороной так же, как обучающей"
    local ref dir
    ref="$(cat "$STAGE_ROOT/artifact.txt" 2>/dev/null)"
    [[ -n $ref ]] || { record "паритет сети" отказ "нет ссылки на артефакт"; return; }
    dir="$STORE/policies/${ref%@*}/${ref#*@}"
    [[ -d $dir ]] || { record "паритет сети" отказ "нет каталога $dir"; return; }

    local describe="$STAGE_ROOT/onnx.json" inputs="$STAGE_ROOT/inputs.npz"
    if ! clean "$EXEC_PY $ONNX_PROBE describe --onnx $dir/policy.onnx --out $describe"; then
        record "паритет сети" отказ "модель не читается onnxruntime окружения исполнения"
        return
    fi
    record "сигнатура сети" факт "$describe"

    # The run directory keeps the TorchScript the framework exports next to the ONNX; if it
    # is there, the two are compared on shared inputs, as the probe compared them (0.18).
    local run_dir pt
    run_dir="$(clean "$EXEC_PY -c \"import json;print(json.load(open('$dir/train.json'))['run'])\"" 2>/dev/null)"
    pt="$(find "$run_dir" -name 'policy.pt' 2>/dev/null | head -n 1)"
    if [[ -z $pt ]]; then
        record "паритет TorchScript и ONNX" пропуск "policy.pt в прогоне не найден"
        return
    fi
    if ! clean "$TRAIN_PY $ONNX_PROBE torch --jit $pt --describe $describe --seed $SEED --out $inputs"; then
        record "паритет TorchScript и ONNX" отказ "policy.pt не исполнился в окружении обучения"
        return
    fi
    if clean "$EXEC_PY $ONNX_PROBE compare --onnx $dir/policy.onnx --inputs $inputs"; then
        record "паритет TorchScript и ONNX" ok "расхождение ниже допуска"
    else
        record "паритет TorchScript и ONNX" отказ "расхождение выше допуска (числа выше)"
    fi
}

# -- main ----------------------------------------------------------------------------------

steps=("$@")
(( ${#steps[@]} )) || read -r -a steps <<< "$ALL_STEPS"
for s in "${steps[@]}"; do
    case "$s" in
        env|plan|train|check|replay) ;;
        *) printf 'неизвестный шаг: %s (доступны: %s)\n' "$s" "$ALL_STEPS"; exit 2 ;;
    esac
done

step "этап 4 фазы 6: ${steps[*]} — лог $LOG"
info "платформа $(git -C "$REPO_ROOT" rev-parse --short HEAD), $APP; TRAIN_ROOT=$TRAIN_ROOT"
info "данные прогонов: $RUN_HOME; хранилище артефактов: $STORE"
for s in "${steps[@]}"; do "step_$s"; done

step "итог"
for line in "${RESULTS[@]}"; do
    IFS='|' read -r check status note <<< "$line"
    printf '    %-40s %-7s %s\n' "$check" "$status" "$note"
done
info "Прислать: $LOG, $STAGE_ROOT/conditions.md, $STAGE_ROOT/plan.txt, $STAGE_ROOT/policy-show.txt"
