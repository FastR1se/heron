#!/usr/bin/env bash
#
# Phase 5, stage 4: the RC5 on the stand, from safe to risky (docs/plans/phase5-rc5.md,
# "Этап 4"; ADR-0049..0057, ADR-P5A, ADR-P5B). Every step is its own call, and a human
# stands at the emergency button through all of them.
#
# Built on the rule of scripts/phase4-stage5.sh: a confirmation is taken by something
# other than the thing under test. Whether the robot moved is read by a separate
# read-only RTD connection before and after each run (pose.py below), not from what the
# skill or the adapter say about themselves; runs are recorded (ADR-0038).
#
#   env     a venv with the platform, the robot package and the vendor's API 1.4.1;
#           env check names what is missing
#   probe   read-only probe: stamps and cycle_time (0.31, 0.47), the emergency button
#           watched (0.25), then --full: model and version (0.22), scale; then pairs
#           act_q / act_flange_x in poses a human sets, compared with the description
#           (ADR-0055). No motion from the platform.
#   reqs    validate; skill check refuses move_joints (criterion 3); env check; dry runs:
#           with a scale the plan is printed, without one the run is refused (ADR-0052);
#           the profile states host and model
#   refuse  a target outside the envelope: refused before sending, the robot does not move
#   move    one joint by MOVE_TARGET (±5° from where it stands) at scale 0.1, recorded
#   stop    a slow motion to STOP_TARGET, interrupted after STOP_AFTER s: the platform's
#           stop (ADR-P5A); where the robot stood (0.27)
#   task    TASK_TARGET at TASK_SCALE, three times, recorded
#
# Usage (the robot's address and archive as variables; targets in degrees, contract order):
#   HOST=192.168.10.10 API_DIR=~/robopro_api bash scripts/phase5-stage4.sh env probe
#   bash scripts/phase5-stage4.sh reqs refuse
#   MOVE_TARGET="[0,-90,90,-90,-90,5]" bash scripts/phase5-stage4.sh move
#   STOP_TARGET="[0,-90,90,-90,-90,40]" STOP_AFTER=3 bash scripts/phase5-stage4.sh stop
#   TASK_TARGET="[...]" TASK_SCALE=0.25 bash scripts/phase5-stage4.sh task
#
# Conditions of every step: the vendor GUI does not control the robot during a run (0.5);
# the controller is put into `run` by a human in the GUI (ADR-0054); after probe 0.22 the
# model is written into examples/robots/rc5/profile.yaml (adapters.real.params.model).

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly PROFILE="$REPO_ROOT/examples/robots/rc5/profile.yaml"
readonly DESCRIPTION="$REPO_ROOT/examples/robots/rc5/rc5.urdf"
readonly SCENARIO="$REPO_ROOT/examples/scenarios/rc5_bench.yaml"
readonly REGISTRY="$REPO_ROOT/examples/robots"
readonly PACKAGE="$REPO_ROOT/examples/plugins/robopro_rc"

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase5-stage4}"
HOST="${HOST:-192.168.10.10}"
API_DIR="${API_DIR:-$HOME/robopro_api}"
PYTHON="${PYTHON:-python3.12}"             # the SDK supports 3.10-3.13, the platform 3.11+
WATCH_S="${WATCH_S:-60}"
PAIRS="${PAIRS:-5}"
MOVE_TARGET="${MOVE_TARGET:-}"
STOP_TARGET="${STOP_TARGET:-}"
STOP_AFTER="${STOP_AFTER:-3}"
TASK_TARGET="${TASK_TARGET:-}"
TASK_SCALE="${TASK_SCALE:-0.25}"
TOLERANCE="${TOLERANCE:-0.5}"              # degrees, as the contract
readonly STAGE_ROOT HOST API_DIR PYTHON WATCH_S PAIRS MOVE_TARGET STOP_TARGET STOP_AFTER \
    TASK_TARGET TASK_SCALE TOLERANCE
readonly VENV="$STAGE_ROOT/venv"
readonly PY="$VENV/bin/python"
readonly CLI="$VENV/bin/$APP"
readonly RECORDS="$STAGE_ROOT/records"
readonly ALL_STEPS="env probe reqs refuse move stop task"

mkdir -p "$STAGE_ROOT" "$RECORDS"
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

cd "$REPO_ROOT" || exit 2

# A pose read by a connection of its own, read_only: the witness of every motion step.
write_helpers() {
    cat > "$STAGE_ROOT/pose.py" <<'PY'
import json, math, sys
from API.rc_api import RobotApi
from robopro_rc.sdk import copy_rtd
hook = sys.excepthook
robot = RobotApi(sys.argv[1], ignore_controller_exceptions=True, read_only=True, autoconnect=False)
sys.excepthook = hook
if not robot.connect(read_only=True):
    print("null"); sys.exit(1)
try:
    packet = copy_rtd(robot._rtd_receiver.get_data())
    print(json.dumps({"deg": [math.degrees(v) for v in packet.q], "safety": packet.safety,
                      "state": packet.controller_state, "mode": packet.motion_mode}))
finally:
    robot.disconnect()
PY
    cat > "$STAGE_ROOT/moved.py" <<'PY'
import json, sys
before, after = json.loads(sys.argv[1]), json.loads(sys.argv[2])
print(max(abs(a - b) for a, b in zip(before["deg"], after["deg"])))
PY
}

pose() { "$PY" "$STAGE_ROOT/pose.py" "$HOST" 2>/dev/null; }
moved() { "$PY" "$STAGE_ROOT/moved.py" "$1" "$2"; }
exceeds() { "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$1" "$2"; }

step_env() {
    step "env: окружение стенда"
    [[ -d $API_DIR/API ]] || { record "env архив API" отказ "нет $API_DIR/API"; return; }
    "$PYTHON" -m venv "$VENV" && "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -e "$REPO_ROOT[record]" -e "$PACKAGE" "$API_DIR" \
        && record "env установка" ok "платформа, пакет RC, API из $API_DIR" \
        || record "env установка" отказ "pip install не прошёл"
    # env check answers for the scenario's world (dummy) and takes no target; the adapter
    # of `real` is checked by run before connect. Here the same three requirements by hand.
    "$PY" - <<'PY' && record "env требования" ok "API 1.4.1, numpy, scipy" \
        || record "env требования" отказ "см. вывод"
import importlib.metadata as m, importlib.util as u, sys
version = m.version("API")
print("API", version, "numpy", m.version("numpy"), "scipy", m.version("scipy"), sys.version)
sys.exit(0 if version == "1.4.1" and u.find_spec("numpy") and u.find_spec("scipy") else 1)
PY
}

step_probe() {
    step "probe: только чтение (0.22, 0.25, 0.31, 0.47, ADR-0055)"
    local out="$STAGE_ROOT/probe"
    mkdir -p "$out"
    "$PY" scripts/phase5-probe.py --host "$HOST" --stamps 5 --json "$out/read_only.json" \
        && record "probe read_only" ok "$out/read_only.json" \
        || record "probe read_only" отказ "подключение RTD не установлено"
    "$PY" scripts/phase5-probe.py --host "$HOST" --full --stamps 0 --json "$out/full.json" \
        && record "probe full (0.22)" ok "$(grep -o '"robot_model": "[^"]*"' "$out/full.json")" \
        || record "probe full (0.22)" отказ "full не подключился: «Пульс» держит управление?"
    info "кнопка: нажмите аварийный останов в течение $WATCH_S с и отпустите (0.25)"
    "$PY" scripts/phase5-probe.py --host "$HOST" --stamps 0 --watch "$WATCH_S" \
        --json "$out/watch.json" && record "probe кнопка (0.25)" ok "$out/watch.json"
    info "позы: $PAIRS поз человеком в «Пульсе», затем Enter"
    "$PY" scripts/phase5-probe.py --host "$HOST" --stamps 0 --pairs "$PAIRS" \
        --json "$out/pairs.json" \
        && "$PY" -m robopro_rc.kinematics "$out/pairs.json" --urdf "$DESCRIPTION" \
        | tee "$out/kinematics.txt" \
        && record "probe пары (ADR-0055)" ok "$out/kinematics.txt: соглашение DH — по строке urdf" \
        || record "probe пары (ADR-0055)" отказ "пары не записаны"
}

step_reqs() {
    step "reqs: проверки до подключения"
    "$CLI" validate "$PROFILE" "$SCENARIO" --registry "$REGISTRY" \
        && record "reqs validate" ok "профиль и сценарий" \
        || record "reqs validate" отказ "см. вывод"
    local check
    check="$("$CLI" skill check move_joints "$PROFILE")"
    [[ $check == *"[skill.control_mode_incompatible]"* ]] \
        && record "reqs критерий 3" ok "move_joints отвергнут до подключения" \
        || record "reqs критерий 3" отказ "$check"
    "$PY" - "$PROFILE" <<'PY' && record "reqs params" ok "host и model заданы" \
        || record "reqs params" отказ "model не задана: пропишите значение пробы 0.22 в профиль"
import sys, yaml
params = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["adapters"]["real"]["params"]
print("params:", params)
sys.exit(0 if params.get("host") and params.get("model") else 1)
PY
    "$CLI" run "$SCENARIO" --registry "$REGISTRY" --target real --speed-scale 0.1 --dry-run \
        && record "reqs план" ok "шкала 0.1, навык goto_joints" \
        || record "reqs план" отказ "сухой прогон не спланирован"
    if "$CLI" run "$SCENARIO" --registry "$REGISTRY" --target real --dry-run; then
        record "reqs без шкалы" отказ "прогон на железе спланирован без шкалы"
    else
        record "reqs без шкалы" ok "отказ (ADR-0052)"
    fi
}

# run_on_robot <label> <scale> <target> [extra args...]: one run, witnessed by pose.py.
run_on_robot() {
    local label="$1" scale="$2" target="$3" before after delta status
    shift 3
    before="$(pose)"
    info "до: $before"
    "$CLI" run "$SCENARIO" --registry "$REGISTRY" --target real --speed-scale "$scale" \
        -g "target=$target" -g "tolerance=$TOLERANCE" "$@"
    status=$?
    after="$(pose)"
    info "после: $after"
    delta="$(moved "$before" "$after" 2>/dev/null || echo "?")"
    LAST_DELTA="$delta"
    LAST_STATUS="$status"
    info "$label: код $status, наибольшее смещение сустава $delta°"
}

step_refuse() {
    step "refuse: цель вне рамки"
    run_on_robot "refuse" 0.1 "[190,0,0,0,0,0]"
    if (( LAST_STATUS != 0 )) && ! exceeds "$LAST_DELTA" 0.05; then
        record "refuse" ok "отказ, смещение $LAST_DELTA°"
    else
        record "refuse" отказ "код $LAST_STATUS, смещение $LAST_DELTA°"
    fi
}

step_move() {
    step "move: один сустав ±5° при шкале 0.1"
    [[ -n $MOVE_TARGET ]] || { record "move" пропущен "задайте MOVE_TARGET от текущей позы"; return; }
    run_on_robot "move" 0.1 "$MOVE_TARGET" --record "$RECORDS/move.mcap"
    (( LAST_STATUS == 0 )) && record "move" ok "смещение $LAST_DELTA°, $RECORDS/move.mcap" \
        || record "move" отказ "код $LAST_STATUS, смещение $LAST_DELTA°"
}

step_stop() {
    step "stop: останов платформы посреди движения (ADR-P5A, 0.27)"
    [[ -n $STOP_TARGET ]] || { record "stop" пропущен "задайте STOP_TARGET"; return; }
    local before after pid
    before="$(pose)"
    "$CLI" run "$SCENARIO" --registry "$REGISTRY" --target real --speed-scale 0.1 \
        -g "target=$STOP_TARGET" -g "tolerance=$TOLERANCE" --record "$RECORDS/stop.mcap" &
    pid=$!
    sleep "$STOP_AFTER"
    kill -INT "$pid"                       # Ctrl+C: the executor stops the robot first
    wait "$pid"
    sleep 1
    after="$(pose)"
    info "до: $before"; info "после останова: $after"
    sleep 2
    local later
    later="$(pose)"
    info "через 2 с: $later"
    record "stop" "см. числа" "сдвиг за 2 с после останова $(moved "$after" "$later")°; $RECORDS/stop.mcap"
}

step_task() {
    step "task: поза навыка, три повтора"
    [[ -n $TASK_TARGET ]] || { record "task" пропущен "задайте TASK_TARGET"; return; }
    local i
    for i in 1 2 3; do
        run_on_robot "task $i" "$TASK_SCALE" "$TASK_TARGET" --record "$RECORDS/task-$i.mcap"
        (( LAST_STATUS == 0 )) && record "task $i" ok "$RECORDS/task-$i.mcap" \
            || record "task $i" отказ "код $LAST_STATUS"
    done
}

summary() {
    step "Итог"
    local row check status note
    printf '\n| Проверка | Результат | Подробности |\n|---|---|---|\n'
    for row in "${RESULTS[@]}"; do
        IFS='|' read -r check status note <<< "$row"
        printf '| %s | %s | %s |\n' "$check" "$status" "$note"
    done
    printf '\nОтправьте целиком файл:\n    %s\n' "$LOG"
}

main() {
    local steps=("$@") s
    (( ${#steps[@]} )) || read -r -a steps <<< "$ALL_STEPS"
    for s in "${steps[@]}"; do
        [[ " $ALL_STEPS " == *" $s "* ]] || { printf 'неизвестный шаг: %s. Доступны: %s\n' "$s" "$ALL_STEPS"; exit 2; }
    done
    write_helpers
    for s in "${steps[@]}"; do
        case "$s" in
            env)    step_env ;;
            probe)  step_probe ;;
            reqs)   step_reqs ;;
            refuse) step_refuse ;;
            move)   step_move ;;
            stop)   step_stop ;;
            task)   step_task ;;
        esac
    done
    summary
}

main "$@"
