#!/usr/bin/env bash
#
# Phase 6, stage 0: facts about the training stack, read off the machine before any code of
# the training backend is written (docs/plans/phase6-learned-skill.md, "Этап 0", facts
# 0.13-0.18). Nothing here trains a platform policy and nothing changes either venv: the
# platform is installed into the training venv only as a dry run, onnxruntime goes into a
# venv of the probe's own.
#
#   env        machine and versions into conditions.md; pip check; rclpy must NOT import in
#              the training venv (invariant 2)
#   core       dry-run install of the platform core into the training venv; onnxruntime wheel
#              for the execution venv's Python (dry run)
#   urdf       what Isaac Lab makes of examples/robots/urdf/arm6.urdf: DOF order against the
#              contract, merged links, limits, default gains -- with and without merging fixed
#              joints
#   threshold  ADR-0005 threshold by surrogate: Isaac-Reach-Franka-v0 at NUM_ENVS for ITER
#              iterations each; peak GPU memory, host memory and swap, seconds per iteration
#   export     play.py on the largest completed run until exported/policy.onnx appears
#   onnx       input/output names, shapes, opset; TorchScript (training venv) against
#              onnxruntime (probe venv) on the same random inputs
#
# Usage:
#   bash scripts/phase6-probe.sh                              # all steps
#   bash scripts/phase6-probe.sh env core urdf
#   NUM_ENVS="512 4096" ITER=100 bash scripts/phase6-probe.sh threshold export onnx
#
# Prerequisites: scripts/phase0.sh install isaac (TRAIN_ROOT, default ~/train-stack) and
# exec; uv on PATH; network for Isaac's assets on the first run. No graphical session needed.
#
# Writes outside $PROBE_ROOT: Isaac Lab's own logs/ in its checkout ($ISAACLAB_DIR/logs), the
# Kit cache in $HOME. Rough time: env/core/urdf a few minutes; threshold 5-15 min per size.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly ARM6_URDF="$REPO_ROOT/examples/robots/urdf/arm6.urdf"
readonly ARM6_PROFILE="$REPO_ROOT/examples/robots/arm6.yaml"
readonly URDF_PROBE="$REPO_ROOT/scripts/phase6-probe-urdf.py"
readonly ONNX_PROBE="$REPO_ROOT/scripts/phase6-probe-onnx.py"
readonly TASK="Isaac-Reach-Franka-v0"
readonly PLAY_TASK="Isaac-Reach-Franka-Play-v0"
readonly EXPERIMENT="franka_reach"         # experiment_name of the task's agent (fact 0.8)

TRAIN_ROOT="${TRAIN_ROOT:-$HOME/train-stack}"     # the same default as scripts/phase0.sh
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase6-probe}"
NUM_ENVS="${NUM_ENVS:-512 1024 2048 4096}"
ITER="${ITER:-50}"
SEED="${SEED:-42}"
EXPORT_TIMEOUT="${EXPORT_TIMEOUT:-900}"
URDF_TIMEOUT="${URDF_TIMEOUT:-900}"       # Kit start plus one conversion; first start compiles shaders
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-3600}"    # per size
HEARTBEAT="${HEARTBEAT:-30}"              # seconds between progress lines of a silent child
readonly TRAIN_ROOT PROBE_ROOT NUM_ENVS ITER SEED EXPORT_TIMEOUT URDF_TIMEOUT TRAIN_TIMEOUT HEARTBEAT
readonly TRAIN_VENV="$TRAIN_ROOT/venv"
readonly TRAIN_PY="$TRAIN_VENV/bin/python"
readonly ISAACLAB_DIR="$TRAIN_ROOT/IsaacLab"
readonly RUNS_DIR="$ISAACLAB_DIR/logs/rsl_rl/$EXPERIMENT"
readonly ORT_VENV="$PROBE_ROOT/ort-venv"
readonly ALL_STEPS="env core urdf threshold export onnx"

[[ $ITER =~ ^[1-9][0-9]*$ ]] || { printf 'ITER=%s: нужно целое больше нуля\n' "$ITER"; exit 2; }
for n in $NUM_ENVS; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || { printf 'NUM_ENVS: %s не целое\n' "$n"; exit 2; }
done

mkdir -p "$PROBE_ROOT"
LOG="$PROBE_ROOT/probe-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

# Nothing inherited but HOME and a plain PATH (phase 4, stage 1): a sourced ROS 2 in the calling
# shell must not decide whether training works -- invariant 2 says it is not there.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-probe}" LANG=C.UTF-8 LC_ALL=C.UTF-8
    "TERM=${TERM:-dumb}"
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

# A child that writes only to its own log looks exactly like a hung one (probe run 1: the urdf
# step showed nothing until it was stopped by hand). So every wait says how long it has been and
# what the child last wrote, and gives up at a deadline with the tail of the log.
watch_group() {  # watch_group <pgid> <log> <timeout seconds>; status of the child, 124 on timeout
    local pgid="$1" log="$2" timeout="$3" started last line
    started="$(date +%s)"; last="$started"
    info "лог процесса: $log (срок ${timeout} с)"
    while kill -0 "$pgid" 2>/dev/null; do
        sleep 1
        if (( $(date +%s) - last >= HEARTBEAT )); then
            last="$(date +%s)"
            line="$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 1 | cut -c1-140)"
            info "… $(( last - started )) с, строк лога $(wc -l < "$log" 2>/dev/null || echo 0): ${line:-пусто}"
        fi
        if (( $(date +%s) - started >= timeout )); then
            info "срок ${timeout} с истёк — процесс останавливается; хвост лога:"
            tail -n 30 "$log" 2>/dev/null | sed 's/^/      /'
            stop_group "$pgid" 15
            wait "$pgid" 2>/dev/null
            return 124
        fi
    done
    wait "$pgid"
}

version_of() {  # version_of <python> <distribution>
    "${BASE_ENV[@]}" "$1" -c "import importlib.metadata as m; print(m.version('$2'))" 2>/dev/null \
        || echo нет
}

# Isaac Lab's launcher from its checkout, as scripts/phase0.sh verify runs it. Arguments are
# quoted once here, so paths with spaces survive the extra shell. Python output unbuffered: into
# a file it is written in blocks, and probe run 2 showed only Kit's own lines for minutes -- a
# silence the probe made, not the child.
isaac_command() {  # isaac_command <script> <args...>
    local script="$1"; shift
    printf "cd %q && source %q && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p %q" \
        "$ISAACLAB_DIR" "$TRAIN_VENV/bin/activate" "$script"
    printf ' %q' "$@"
}

need_train_stack() {
    [[ -x $TRAIN_PY && -x $ISAACLAB_DIR/isaaclab.sh ]] && return 0
    record "стек обучения" отказ "нет $TRAIN_PY или $ISAACLAB_DIR/isaaclab.sh (scripts/phase0.sh install isaac)"
    return 1
}

# -- env ---------------------------------------------------------------------------------------

step_env() {
    step "env: машина и версии (факт 0.13)"
    need_train_stack || return
    local out="$PROBE_ROOT/conditions.md" gpu ram swap cpu os commit dirty
    gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo нет)"
    ram="$(awk '/MemTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    swap="$(awk '/SwapTotal/{printf "%.1f ГиБ", $2/1048576}' /proc/meminfo)"
    cpu="$(LC_ALL=C lscpu | awk -F: '/Model name/{gsub(/^ +/, "", $2); print $2; exit}')"
    os="$(. /etc/os-release && echo "$PRETTY_NAME"), ядро $(uname -r)"
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    dirty="$(git -C "$REPO_ROOT" status --porcelain | wc -l)"
    {
        echo "# Условия пробы фазы 6 — $(date -Iseconds)"
        echo
        echo "| Что | Значение |"
        echo "|---|---|"
        echo "| GPU, драйвер, память | $gpu |"
        echo "| CPU | $cpu |"
        echo "| RAM / своп | $ram / $swap |"
        echo "| ОС | $os |"
        echo "| платформа | \`$commit\`, изменённых файлов в рабочей копии: $dirty |"
        echo "| Python обучения | $("${BASE_ENV[@]}" "$TRAIN_PY" -c 'import sys; print(sys.version.split()[0])') |"
        echo "| Isaac Lab (checkout) | $(git -C "$ISAACLAB_DIR" describe --tags --always 2>/dev/null || echo нет) |"
        local dist
        for dist in isaacsim isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib torch onnx onnxscript \
                    onnxruntime numpy pydantic pyyaml typer packaging gymnasium; do
            echo "| $dist | $(version_of "$TRAIN_PY" "$dist") |"
        done
        echo "| Python исполнения | $("${BASE_ENV[@]}" "$EXEC_PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo нет) |"
    } > "$out"
    cat "$out"
    record "условия" ok "$out"

    if clean "VIRTUAL_ENV='$TRAIN_VENV' uv pip check --python '$TRAIN_PY'" > "$PROBE_ROOT/pip-check.txt" 2>&1; then
        record "pip check (обучение)" ok "конфликтов нет"
    else
        record "pip check (обучение)" факт "есть конфликты: $PROBE_ROOT/pip-check.txt ($(wc -l < "$PROBE_ROOT/pip-check.txt") строк)"
    fi
    sed 's/^/      /' "$PROBE_ROOT/pip-check.txt" | head -20

    if clean "'$TRAIN_PY' -c 'import rclpy'" >/dev/null 2>&1; then
        record "rclpy в окружении обучения" отказ "импортируется — инвариант 2 нарушен"
    else
        record "rclpy в окружении обучения" ok "не импортируется"
    fi
}

# -- core --------------------------------------------------------------------------------------

step_core() {
    step "core: ядро в окружении обучения и onnxruntime в окружении исполнения (факт 0.14)"
    need_train_stack || return
    local out="$PROBE_ROOT/core-dry-run.txt" removed
    if clean "uv pip install --dry-run --python '$TRAIN_PY' -e '$REPO_ROOT'" > "$out" 2>&1; then
        removed="$(grep -c '^ - ' "$out")"
        if (( removed == 0 )); then
            record "ядро в окружение обучения" ok "сухая установка ничего не заменяет"
        else
            record "ядро в окружение обучения" факт "заменило бы $removed пакетов: $out"
        fi
    else
        record "ядро в окружение обучения" отказ "разрешение зависимостей не сошлось: $out"
    fi
    sed 's/^/      /' "$out" | head -30

    out="$PROBE_ROOT/ort-dry-run.txt"
    if [[ ! -x $EXEC_PY ]]; then
        record "onnxruntime (исполнение)" отказ "нет $EXEC_PY (scripts/phase0.sh install exec)"
    elif clean "uv pip install --dry-run --python '$EXEC_PY' onnxruntime" > "$out" 2>&1; then
        record "onnxruntime (исполнение)" ok "$(grep -m1 -o 'onnxruntime==[^ ]*' "$out" || echo 'уже стоит')"
    else
        record "onnxruntime (исполнение)" отказ "колесо не найдено: $out"
    fi
    sed 's/^/      /' "$out" | head -10
}

# -- urdf --------------------------------------------------------------------------------------

step_urdf() {
    step "urdf: arm6 глазами Isaac Lab (факт 0.15)"
    need_train_stack || return
    local merge json order
    order="$("${BASE_ENV[@]}" "$EXEC_PY" -c "import yaml; print(' '.join(yaml.safe_load(open('$ARM6_PROFILE'))['contract']['joint_order']))" 2>/dev/null)"
    info "joint_order контракта: ${order:-не прочитан}"
    for merge in yes no; do
        json="$PROBE_ROOT/urdf-merge-$merge.json"
        rm -f "$json"
        start_group "$PROBE_ROOT/urdf-merge-$merge.log" "$(isaac_command "$URDF_PROBE" \
            --urdf "$ARM6_URDF" --usd-dir "$PROBE_ROOT/usd-merge-$merge" \
            --merge-fixed-joints "$merge" --out "$json" --headless)"
        watch_group "$LAST_GROUP" "$PROBE_ROOT/urdf-merge-$merge.log" "$URDF_TIMEOUT"
        local status=$?
        if [[ ! -s $json ]]; then
            record "urdf merge=$merge" отказ "факты не записаны (код $status, 124 — срок): $PROBE_ROOT/urdf-merge-$merge.log"
            tail -20 "$PROBE_ROOT/urdf-merge-$merge.log" | sed 's/^/      /'
            continue
        fi
        "${BASE_ENV[@]}" "$EXEC_PY" - "$json" "$order" <<'PY'
import json, sys
facts = json.load(open(sys.argv[1]))
contract = sys.argv[2].split()
names = facts["joint_names"]
print(f"      суставы артикуляции ({len(names)}): {' '.join(names)}")
print(f"      звенья ({len(facts['body_names'])}): {' '.join(facts['body_names'])}")
arm = [n for n in names if n in contract]
print(f"      порядок суставов контракта в артикуляции совпадает: {'да' if arm == contract else 'НЕТ'}")
for key in ("joint_pos_limits", "joint_vel_limits", "joint_effort_limits",
            "joint_stiffness", "joint_damping", "default_joint_pos"):
    print(f"      {key}: {facts.get(key)}")
PY
        record "urdf merge=$merge" факт "$json"
    done
}

# -- threshold ---------------------------------------------------------------------------------

sample() {  # sample <pid> <file>: GPU memory, host memory and swap in MiB, once a second
    local pid="$1" out="$2"
    while kill -0 "$pid" 2>/dev/null; do
        printf '%s %s %s %s\n' "$(date +%s)" \
            "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" \
            "$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print int((t-a)/1024)}' /proc/meminfo)" \
            "$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)" \
            >> "$out"
        sleep 1
    done
}

peak() { awk -v c="$2" 'BEGIN{m=0} {if ($c > m) m = $c} END{print m}' "$1"; }

step_threshold() {
    step "threshold: порог ADR-0005 суррогатом — $TASK, $ITER итераций (факт 0.16)"
    need_train_stack || return
    local dir="$PROBE_ROOT/threshold" n log samples base_gpu base_ram base_swap started elapsed status
    local gpu ram swap iters run
    mkdir -p "$dir"
    printf '| сред | итераций | с | с/итер. | пик VRAM, МиБ (фон) | пик RAM, МиБ (фон) | пик свопа, МиБ (фон) | код |\n|---|---|---|---|---|---|---|---|\n' \
        > "$dir/table.md"
    for n in $NUM_ENVS; do
        log="$dir/train-n$n.log"; samples="$dir/samples-n$n.txt"
        rm -f "$samples"
        base_gpu="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
        base_ram="$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print int((t-a)/1024)}' /proc/meminfo)"
        base_swap="$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)"
        info "сред $n: фон VRAM $base_gpu МиБ, RAM $base_ram МиБ, своп $base_swap МиБ"
        started="$(date +%s)"
        start_group "$log" "$(isaac_command "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/train.py" \
            --task "$TASK" --headless --num_envs "$n" --max_iterations "$ITER" --seed "$SEED" \
            --run_name "n$n")"
        sample "$LAST_GROUP" "$samples" &
        watch_group "$LAST_GROUP" "$log" "$TRAIN_TIMEOUT"; status=$?
        wait $! 2>/dev/null   # the sampler ends with the child
        elapsed=$(( $(date +%s) - started ))
        gpu="$(peak "$samples" 2)"; ram="$(peak "$samples" 3)"; swap="$(peak "$samples" 4)"
        iters="$(grep -c 'Learning iteration' "$log")"
        printf '| %s | %s | %s | %s | %s (%s) | %s (%s) | %s (%s) | %s |\n' "$n" "$iters" "$elapsed" \
            "$(awk -v e="$elapsed" -v i="$iters" 'BEGIN{ if (i > 0) printf "%.2f", e/i; else print "—" }')" \
            "$gpu" "$base_gpu" "$ram" "$base_ram" "$swap" "$base_swap" "$status" >> "$dir/table.md"
        run="$(ls -td "$RUNS_DIR"/*_n"$n" 2>/dev/null | head -1)"
        if (( status == 0 )) && [[ -n $run ]] && ls "$run"/model_*.pt >/dev/null 2>&1; then
            record "обучение n=$n" ok "$iters итераций за $elapsed с, VRAM $gpu МиБ, RAM $ram МиБ; $run"
            echo "$run" > "$dir/last_run"
        else
            record "обучение n=$n" отказ "код $status, итераций $iters: $log"
            tail -25 "$log" | sed 's/^/      /'
        fi
    done
    cat "$dir/table.md"
    info "ADR-0005: ≥512 сред, пик VRAM ≤ 7 ГБ (7168 МиБ), пик RAM ≤ 13 ГБ (13312 МиБ), прогон ≤ 6 ч."
    info "Числа — суррогат (задача производителя); в карточку эксперимента — так и называть."
}

# -- export ------------------------------------------------------------------------------------

step_export() {
    step "export: play.py до появления policy.onnx (факты 0.12, 0.17)"
    need_train_stack || return
    local run log waited=0
    run="$(cat "$PROBE_ROOT/threshold/last_run" 2>/dev/null)"
    if [[ -z $run || ! -d $run ]]; then
        record "экспорт" отказ "нет завершённого прогона: сначала шаг threshold"
        return
    fi
    log="$PROBE_ROOT/export.log"
    rm -f "$run/exported/policy.onnx"
    start_group "$log" "$(isaac_command "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/play.py" \
        --task "$PLAY_TASK" --headless --num_envs 1 --load_run "$(basename "$run")")"
    while [[ ! -s $run/exported/policy.onnx ]] && kill -0 "$LAST_GROUP" 2>/dev/null \
            && (( waited < EXPORT_TIMEOUT )); do
        sleep 2; waited=$(( waited + 2 ))
        (( waited % HEARTBEAT == 0 )) && info "… $waited с, policy.onnx ещё нет; лог: $(grep -v '^[[:space:]]*$' "$log" | tail -n 1 | cut -c1-140)"
    done
    sleep 3   # the JIT file is written first, the ONNX file second; let the write finish
    stop_group "$LAST_GROUP" 15
    if [[ -s $run/exported/policy.onnx ]]; then
        record "экспорт" ok "$run/exported: $(ls "$run/exported" | tr '\n' ' ')через ${waited} с"
        echo "$run/exported" > "$PROBE_ROOT/exported"
    else
        record "экспорт" отказ "policy.onnx не появился за ${waited} с: $log"
        tail -25 "$log" | sed 's/^/      /'
    fi
}

# -- onnx --------------------------------------------------------------------------------------

step_onnx() {
    step "onnx: вход и выход, паритет TorchScript и onnxruntime (факты 0.17, 0.18)"
    local exported inputs
    exported="$(cat "$PROBE_ROOT/exported" 2>/dev/null)"
    if [[ -z $exported || ! -s $exported/policy.onnx ]]; then
        record "onnx" отказ "нет экспортированной политики: сначала шаг export"
        return
    fi
    if [[ ! -x $ORT_VENV/bin/python ]]; then
        clean "uv venv --python 3.12 '$ORT_VENV'" || { record "onnx" отказ "venv пробы не создан"; return; }
    fi
    clean "uv pip install --python '$ORT_VENV/bin/python' onnxruntime onnx numpy" > "$PROBE_ROOT/ort-install.txt" 2>&1 \
        || { record "onnx" отказ "onnxruntime не поставился: $PROBE_ROOT/ort-install.txt"; return; }
    info "onnxruntime $(version_of "$ORT_VENV/bin/python" onnxruntime), onnx $(version_of "$ORT_VENV/bin/python" onnx)"

    inputs="$PROBE_ROOT/parity.npz"
    if ! "${BASE_ENV[@]}" "$ORT_VENV/bin/python" "$ONNX_PROBE" describe \
            --onnx "$exported/policy.onnx" --out "$PROBE_ROOT/onnx.json"; then
        record "onnx: описание" отказ "модель не читается"
        return
    fi
    record "onnx: описание" факт "$PROBE_ROOT/onnx.json"
    if ! "${BASE_ENV[@]}" "$TRAIN_PY" "$ONNX_PROBE" torch --jit "$exported/policy.pt" \
            --describe "$PROBE_ROOT/onnx.json" --seed "$SEED" --out "$inputs"; then
        record "onnx: TorchScript" отказ "policy.pt не исполнился в окружении обучения"
        return
    fi
    if "${BASE_ENV[@]}" "$ORT_VENV/bin/python" "$ONNX_PROBE" compare \
            --onnx "$exported/policy.onnx" --inputs "$inputs"; then
        record "onnx: паритет" ok "расхождение ниже допуска"
    else
        record "onnx: паритет" отказ "расхождение выше допуска (числа выше)"
    fi
}

# -- main --------------------------------------------------------------------------------------

steps=("$@")
(( ${#steps[@]} )) || read -r -a steps <<< "$ALL_STEPS"
for s in "${steps[@]}"; do
    case "$s" in
        env|core|urdf|threshold|export|onnx) ;;
        *) printf 'неизвестный шаг: %s (доступны: %s)\n' "$s" "$ALL_STEPS"; exit 2 ;;
    esac
done

step "проба фазы 6: ${steps[*]} — лог $LOG"
info "платформа $(git -C "$REPO_ROOT" rev-parse --short HEAD), $APP; TRAIN_ROOT=$TRAIN_ROOT"
for s in "${steps[@]}"; do "step_$s"; done

step "итог"
for line in "${RESULTS[@]}"; do
    IFS='|' read -r check status note <<< "$line"
    printf '    %-32s %-7s %s\n' "$check" "$status" "$note"
done
info "Прислать лог целиком: $LOG; и $PROBE_ROOT/conditions.md, $PROBE_ROOT/threshold/table.md"
