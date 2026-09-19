#!/usr/bin/env bash
#
# Phase 8, stage 3: one mission, two robots, one Gazebo world (docs/plans/phase8-coordinator.md,
# "Этап 3"; ADR-0065 - ADR-0070).
#
# The scene of phase 4, stage 7 (examples/scenarios/complex.yaml): arm6 through ros2_control and
# G1 through its vendor's DDS. Phase 4 drove them with two `run` commands in turn; here one
# `mission run` hands out the tasks of examples/missions/complex.yaml and each robot works in a
# process of its own. The processes are started separately (`--no-spawn`), each in its own
# environment, because G1's link and rclpy cannot share one (ADR-0022, ADR-0068):
#
#   coordinator  the execution venv, no ROS 2 sourced;
#   g1           the venv with the vendor's SDK and CycloneDDS 0.10.2, no ROS 2 sourced;
#   arm1         the execution venv with ROS 2 sourced.
#
# Nobody hands a process the world's environment: every robot process applies the one written
# in the plan (ADR-0039, ADR-0068).
#
#   env    the venvs of phase 4, stage 7, with this checkout installed; the bridge's workspace
#   reqs   mission validate in both worlds; a dry run names the assignment, the world's
#          environment and the robot processes
#   scene  sim run; mission run --no-spawn --record with the two robot processes; the result,
#          the overlap of g1_arms and arm_hold, the records; both robots moved (witnesses of their
#          own); then a mission whose arm task is refused while G1 moves: G1's task is cancelled,
#          the mission fails and names the task; the platform stops the world
#
# Usage:
#   bash scripts/phase8-stage3.sh                    # all steps
#   GUI=1 bash scripts/phase8-stage3.sh scene        # with the simulator's window and RViz
#   GUI=1 REC=1 bash scripts/phase8-stage3.sh scene  # and the screen into docs/media
#
# Prerequisites: scripts/phase4-stage7.sh env has been run on this machine (the execution venv,
# the SDK venv and the bridge's workspace of stage 5 are in place). ffmpeg for REC.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="examples/scenarios/complex.yaml"
readonly BENCH_SCENARIO="examples/scenarios/complex_bench.yaml"
readonly MISSION="examples/missions/complex.yaml"
readonly G1_PROFILE="examples/robots/g1_23dof/profile.yaml"
readonly ARM_PROFILE="examples/robots/arm6.yaml"
readonly REGISTRY="examples/robots"
readonly PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly METRICS="$REPO_ROOT/scripts/phase4-stage6-metrics.py"
readonly MEDIA="$REPO_ROOT/docs/media"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
# The goals of the mission file: G1's arms forward, the arm back home at the end.
readonly G1_TARGET="[0,0,0,0,0,0, 0,0,0,0,0,0, 0, -0.5,0.3,0,0.8,0, -0.5,-0.3,0,0.8,0]"
readonly ARM_HOME="[0,0,0,0,0,0]"

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase8-stage3}"
STAGE5_ROOT="${STAGE5_ROOT:-$HOME/phase4-stage5}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
GUI="${GUI:-0}"
REC="${REC:-0}"
readonly STAGE_ROOT STAGE5_ROOT PROBE_ROOT GUI REC
readonly PY_DIR="$STAGE_ROOT/py"
readonly WS="$STAGE5_ROOT/ws"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly OWN_PY="$VENV_OWN/bin/python"
readonly OWN_CLI="$VENV_OWN/bin/$APP"
readonly ALL_STEPS="env reqs scene"

mkdir -p "$STAGE_ROOT" "$PY_DIR"
LOG="$STAGE_ROOT/stage3-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage3}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

readonly ROS_ENV="source '$ROS_SETUP'"
readonly WS_ENV="$ROS_ENV && source '$WS/install/setup.bash'"

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
    for pgid in "${STARTED[@]}"; do stop_group "$pgid" 10 >/dev/null 2>&1; done
}
trap cleanup EXIT

running() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ $state != Z* ]]; }

wait_for() {  # wait_for <pid> <seconds>: 0 when the process ended in time
    local pid="$1" limit="$2" waited=0
    while running "$pid" && (( waited < limit )); do sleep 1; waited=$(( waited + 1 )); done
    ! running "$pid"
}

# ---------------------------------------------------------------------------

write_probes() {
    # Witnesses of their own, as in phase 4, stage 7: the platform's claim is its record, the
    # confirmation comes from another tool.
    cat > "$PY_DIR/window.py" <<'PY'
# LowState from a DDS participant of its own for <window> s, after waiting up to <wait> s.
import json, statistics, sys, time
import yaml
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.qos import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_g1.channel import STATE_TOPIC, LinkConfig
from unitree_g1.slots import slots_of

profile_path, wait, window, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
profile = yaml.safe_load(open(profile_path, encoding="utf-8"))
link = LinkConfig.from_params(profile["adapters"]["sim.gazebo"]["params"])
slots = slots_of(profile["contract"]["joint_order"])
domain = Domain(link.domain, link.cyclonedds_xml())
participant = DomainParticipant(link.domain)
qos = Qos(Policy.History.KeepLast(4096), Policy.Reliability.BestEffort)
reader = DataReader(participant, Topic(participant, STATE_TOPIC, LowState_), qos=qos)
start, first, got = time.monotonic(), None, []
while True:
    now = time.monotonic()
    for sample in reader.take(N=4096):
        if hasattr(sample, "tick"):
            first = first or now
            got.append((sample.tick, [sample.motor_state[s].q for s in slots]))
    if (first is not None and now - first >= window) or (first is None and now - start >= wait):
        break
    time.sleep(0.01)
if len(got) < 2:
    print(f"свидетель G1: LowState принято {len(got)} за {time.monotonic() - start:.0f} с")
    sys.exit(1)
q = [statistics.fmean(s) for s in zip(*(item[1] for item in got))]
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"tick": got[-1][0], "q_mean": q}, handle)
print(f"свидетель G1: LowState {len(got)}, tick {got[-1][0]}")
PY

    cat > "$PY_DIR/arm_state.py" <<'PY'
# One JointState of the arm from a ROS 2 node of its own, in the contract's order.
import json, sys, time
import rclpy
import yaml
from sensor_msgs.msg import JointState

profile, topic, wait, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
order = yaml.safe_load(open(profile, encoding="utf-8"))["contract"]["joint_order"]
rclpy.init()
node = rclpy.create_node("phase8_arm_witness")
got = []
node.create_subscription(JointState, topic, got.append, 10)
deadline = time.monotonic() + wait
while not got and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
if not got:
    print(f"свидетель манипулятора: {topic} молчит {wait:g} с")
    sys.exit(1)
message = got[-1]
index = {name: i for i, name in enumerate(message.name)}
q = [message.position[index[name]] for name in order]
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"q_mean": q}, handle)
print(f"свидетель манипулятора: {topic} " + ", ".join(f"{value:.3f}" for value in q))
PY

    cat > "$PY_DIR/moved.py" <<'PY'
# Whether the joints went toward the target: before.json after.json '<target as JSON>' <RMS limit rad>.
import json, math, sys
before, after = (json.load(open(path, encoding="utf-8"))["q_mean"] for path in sys.argv[1:3])
target, limit = json.loads(sys.argv[3]), float(sys.argv[4])
was = [abs(q - t) for q, t in zip(before, target)]
now = [abs(q - t) for q, t in zip(after, target)]
rms = lambda gaps: math.sqrt(sum(g * g for g in gaps) / len(gaps))
print(f"до цели RMS {rms(was):.3f} -> {rms(now):.3f} рад")
sys.exit(0 if rms(now) <= limit and rms(now) <= 0.5 * rms(was) else 1)
PY

    cat > "$PY_DIR/at_pose.py" <<'PY'
# Whether a pose is within <limit> rad of the target in every joint: state.json '<target>' <limit>.
import json, sys
q = json.load(open(sys.argv[1], encoding="utf-8"))["q_mean"]
target, limit = json.loads(sys.argv[2]), float(sys.argv[3])
worst = max(abs(a - b) for a, b in zip(q, target))
print(f"худшее отклонение от цели {worst:.4f} рад")
sys.exit(0 if worst <= limit else 1)
PY

    cat > "$PY_DIR/outcome.py" <<'PY'
# The mission's result against what was expected: <run dir> '<expected states as JSON>' [overlap A B].
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
expected = json.loads(sys.argv[2])
result = json.loads((run / "result.json").read_text(encoding="utf-8"))
tasks = {t["task"]: t for t in result["tasks"]}
states = {name: t["state"] for name, t in tasks.items()}
print(f"миссия {result['status']}: " + ", ".join(f"{k} {v}" for k, v in states.items()))
if result.get("reason"):
    print(f"причина: {result['reason']}")
ok = states == expected
if len(sys.argv) > 4:
    a, b = tasks[sys.argv[3]], tasks[sys.argv[4]]
    overlap = min(a["finished"], b["finished"]) - max(a["started"], b["started"])
    print(f"{a['task']} и {b['task']} шли вместе {overlap:.2f} с")
    ok = ok and overlap > 0
sys.exit(0 if ok else 1)
PY

    cat > "$PY_DIR/scene_env.py" <<'PY'
# The environment the generated launch sets, as shell exports -- for this script's own witnesses.
import shlex, sys
from xml.etree import ElementTree as ET
body = open(sys.argv[1], encoding="utf-8").read().split("?>", 1)[1]
for item in ET.fromstring(body).iter("set_env"):
    print(f"export {item.get('name')}={shlex.quote(item.get('value') or '')}")
PY
}

# ---------------------------------------------------------------------------

step_env() {
    step "Окружение"
    info "лог: $LOG"
    info "репозиторий: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l); GUI=$GUI REC=$REC"
    local ok=1
    [[ -f $ROS_SETUP ]] || { record env отказ "нет $ROS_SETUP"; ok=0; }
    [[ -f $WS/install/setup.bash ]] || { record env отказ "нет $WS: bash scripts/phase4-stage5.sh env"; ok=0; }
    if [[ -x $EXEC_PY ]]; then
        clean "cd '$REPO_ROOT' && uv sync --extra dev --extra ros2 && uv pip install --python '$EXEC_PY' -e '$PACKAGE'" ||
            { record env отказ "uv sync или установка пакета G1 не прошли"; ok=0; }
    else
        record env отказ "нет $EXEC_VENV: bash scripts/phase4-stage7.sh env"; ok=0
    fi
    if [[ -x $OWN_PY ]] && "$OWN_PY" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; then
        clean "'$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$PACKAGE'" ||
            { record env отказ "установка в $VENV_OWN не прошла"; ok=0; }
        # The G1 process must be free of rclpy (ADR-0022): the command line may not bring it in.
        clean "'$OWN_PY' -c 'import sys, ${APP}.cli.main; sys.exit(\"rclpy\" in sys.modules)'" ||
            { record env отказ "командная строка платформы загружает rclpy в процессе G1"; ok=0; }
    else
        record env отказ "нет $VENV_OWN с SDK: bash scripts/phase4-probe.sh 0.5"; ok=0
    fi
    clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" > /dev/null ||
        { record env отказ "описания производителя не загрузились"; ok=0; }
    (( ok )) && record env ok "предусловия на месте"
}

step_reqs() {
    local out="$STAGE_ROOT/reqs" code
    step "reqs — mission validate: одна миссия, два мира"
    for scene in "$SCENARIO" "$BENCH_SCENARIO"; do
        clean "cd '$REPO_ROOT' && '$EXEC_CLI' mission validate '$scene' '$MISSION' -R '$REGISTRY'" > "$out-validate.txt" 2>&1
        code=$?
        cat "$out-validate.txt"
        if (( code == 0 )) && grep -qE '^  g1_arms +-> g1 ' "$out-validate.txt" && grep -qE '^  arm_home +-> arm1 ' "$out-validate.txt"; then
            record "reqs validate $(basename "$scene")" ok "g1_arms -> g1, остальные -> arm1"
        else
            record "reqs validate $(basename "$scene")" "отказ, код $code" "$(grep -m1 '^ошибка' "$out-validate.txt")"
        fi
    done

    step "reqs — mission run --dry-run --no-spawn: назначение, окружение мира, процессы"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' mission run '$SCENARIO' '$MISSION' -R '$REGISTRY' --into '$STAGE_ROOT/dry' --no-spawn --dry-run" > "$out-dry.txt" 2>&1
    code=$?
    cat "$out-dry.txt"
    if (( code == 0 )) && grep -qE '^окружение .*ROS_DOMAIN_ID=1' "$out-dry.txt" &&
        grep -qF "mission robot $STAGE_ROOT/dry --robot g1" "$out-dry.txt" &&
        grep -qF "mission robot $STAGE_ROOT/dry --robot arm1" "$out-dry.txt"; then
        record "reqs сухой прогон" ok "окружение мира в плане, два процесса роботов"
    else
        record "reqs сухой прогон" "отказ, код $code" "$(tail -2 "$out-dry.txt" | tr '\n' ' ')"
    fi
}

# Start the two robot processes of a run, each in its own environment (ADR-0068).
start_robots() {  # start_robots <run dir> <log prefix>
    local run="$1" prefix="$2" waited=0
    while [[ ! -f "$run/plan.json" ]] && (( waited < 60 )); do sleep 1; waited=$(( waited + 1 )); done
    [[ -f "$run/plan.json" ]] || return 1
    start_group "$prefix-g1.log" "cd '$REPO_ROOT' && exec '$OWN_CLI' mission robot '$run' --robot g1"
    start_group "$prefix-arm1.log" "$ROS_ENV && cd '$REPO_ROOT' && exec '$EXEC_CLI' mission robot '$run' --robot arm1"
}

step_scene() {
    local dir="$STAGE_ROOT/scene" pgid waited=0 flags="" coordinator rec_pid=""
    rm -rf "$dir"; mkdir -p "$dir"
    step "scene — один мир, одна миссия, два робота в своих процессах"
    if pgrep -f 'gz sim' >/dev/null; then record scene пропущено "Gazebo уже запущен"; return; fi
    if (( GUI )); then
        [[ -n ${DISPLAY:-} ]] || { record scene пропущено "GUI=1, а DISPLAY не задан"; return; }
    else
        flags="--headless --no-rviz"
    fi

    start_group "$dir/sim.log" "$WS_ENV && cd '$REPO_ROOT' && exec '$EXEC_CLI' sim run '$SCENARIO' --into '$dir/world' -R '$REGISTRY' $flags"
    pgid=$LAST_GROUP
    while [[ ! -f "$dir/world/bringup.launch.xml" ]] && running "$pgid" && (( waited < 60 )); do
        sleep 1; waited=$(( waited + 1 ))
    done
    if [[ ! -f "$dir/world/bringup.launch.xml" ]]; then
        record "scene sim run" отказ "мир не записан: $(tail -1 "$dir/sim.log")"
        stop_group "$pgid" 30; return
    fi
    "$EXEC_PY" "$PY_DIR/scene_env.py" "$dir/world/bringup.launch.xml" > "$dir/scene.env"
    local witness_env="$WS_ENV && source '$dir/scene.env'"

    # The world is up when G1's state is on the wire; G1 falls before its first command (ADR-0018),
    # so "before" is taken after the fall, as in phase 4.
    clean "cd '$REPO_ROOT' && timeout 220 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 180 1 '$dir/g1-before.json'" > "$dir/g1-before.txt" 2>&1
    cat "$dir/g1-before.txt"
    if [[ ! -s "$dir/g1-before.json" ]]; then
        record "scene состояние G1" отказ "$(tail -1 "$dir/g1-before.txt")"
        tail -60 "$dir/sim.log" | sed 's/^/      /'
        stop_group "$pgid" 30; return
    fi
    sleep 8
    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 30 1 '$dir/g1-before.json'" > "$dir/g1-before.txt" 2>&1
    record "scene мир" ok "$(tail -1 "$dir/g1-before.txt")"

    if (( REC )); then
        local size
        size="$(xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
        if command -v ffmpeg >/dev/null && [[ -n $size && -n ${DISPLAY:-} ]]; then
            start_group "$dir/screen.log" "exec ffmpeg -y -loglevel error -f x11grab -video_size '$size' -framerate 15 -i '$DISPLAY' -t 40 -pix_fmt yuv420p '$dir/screen.mp4'"
            rec_pid=$LAST_GROUP
        else
            record "media запись" невозможна "ffmpeg $(command -v ffmpeg >/dev/null && echo есть || echo нет), размер ${size:-?}, DISPLAY ${DISPLAY:-нет}"
        fi
    fi

    # -- the mission ---------------------------------------------------------------
    local run="$dir/mission"
    start_group "$dir/coordinator.log" "cd '$REPO_ROOT' && exec '$EXEC_CLI' mission run '$SCENARIO' '$MISSION' -R '$REGISTRY' --into '$run' --no-spawn --record"
    coordinator=$LAST_GROUP
    if ! start_robots "$run" "$dir/robot"; then
        record "scene миссия" отказ "план не записан: $(tail -3 "$dir/coordinator.log" | tr '\n' ' ')"
        stop_group "$coordinator" 10; stop_group "$pgid" 30; return
    fi
    if ! wait_for "$coordinator" 240; then
        record "scene миссия" отказ "координатор не закончил за 240 с"
        stop_group "$coordinator" 20
    fi
    cat "$dir/coordinator.log"
    sed 's/^/      g1: /' "$dir/robot-g1.log"
    sed 's/^/      arm1: /' "$dir/robot-arm1.log"
    if "$EXEC_PY" "$PY_DIR/outcome.py" "$run" \
        '{"arm_aside": "succeeded", "g1_arms": "succeeded", "arm_hold": "succeeded", "arm_home": "succeeded"}' \
        g1_arms arm_hold > "$dir/outcome.txt" 2>&1; then
        record "scene миссия" ok "$(tr '\n' ' ' < "$dir/outcome.txt")"
    else
        record "scene миссия" отказ "$(tr '\n' ' ' < "$dir/outcome.txt")"
    fi
    local task missing=""
    for task in arm_aside g1_arms arm_hold arm_home; do [[ -s "$run/$task.mcap" ]] || missing+="$task "; done
    [[ -z $missing ]] && record "scene записи" ok "MCAP на каждую из 4 задач, events.jsonl, result.json" ||
        record "scene записи" отказ "нет записей: $missing"

    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 30 1 '$dir/g1-after.json'" > "$dir/g1-after.txt" 2>&1
    if "$OWN_PY" "$PY_DIR/moved.py" "$dir/g1-before.json" "$dir/g1-after.json" "$G1_TARGET" 1.0 > "$dir/g1-moved.txt" 2>&1; then
        record "scene движение G1" ok "$(tail -1 "$dir/g1-moved.txt")"
    else
        record "scene движение G1" отказ "$(tail -1 "$dir/g1-moved.txt")"
    fi
    clean "$witness_env && timeout 60 '$EXEC_PY' '$PY_DIR/arm_state.py' '$ARM_PROFILE' /arm1/joint_states 20 '$dir/arm-after.json'" > "$dir/arm-after.txt" 2>&1
    if [[ -s "$dir/arm-after.json" ]] && "$EXEC_PY" "$PY_DIR/at_pose.py" "$dir/arm-after.json" "$ARM_HOME" 0.05 > "$dir/arm-home.txt" 2>&1; then
        record "scene манипулятор дома" ok "$(tail -1 "$dir/arm-home.txt")"
    else
        record "scene манипулятор дома" отказ "$(tail -1 "$dir/arm-home.txt" 2>/dev/null || tail -1 "$dir/arm-after.txt")"
    fi
    for task in g1_arms arm_aside; do
        [[ -s "$run/$task.mcap" ]] && "$EXEC_PY" "$METRICS" run "$run/$task.mcap" --label "mission-$task" --out "$dir/$task-metrics.json" 2>&1 | sed 's/^/      /'
    done

    # -- a failure in the middle (ADR-0069) ------------------------------------------------
    # G1 moves for 8 s; the arm's goal lies outside its envelope, so its skill refuses it on start.
    cat > "$dir/failing.yaml" <<YAML
schema_version: 1
name: failing
tasks:
  - {id: g1_long, skill: move_joints, requires: {dof: "==23"}, goal: {target: $G1_TARGET, duration: 8.0}}
  - {id: arm_bad, skill: move_joints, requires: {dof: "==6"}, after: [], goal: {target: [9,0,0,0,0,0], duration: 4.0}}
YAML
    local failing="$dir/failing-run"
    start_group "$dir/failing.log" "cd '$REPO_ROOT' && exec '$EXEC_CLI' mission run '$SCENARIO' '$dir/failing.yaml' -R '$REGISTRY' --into '$failing' --no-spawn --record"
    coordinator=$LAST_GROUP
    if start_robots "$failing" "$dir/failing-robot" && wait_for "$coordinator" 120; then
        cat "$dir/failing.log"
        if "$EXEC_PY" "$PY_DIR/outcome.py" "$failing" '{"g1_long": "cancelled", "arm_bad": "failed"}' > "$dir/failing-outcome.txt" 2>&1 &&
            grep -qF "задача arm_bad (arm1" "$dir/failing-outcome.txt"; then
            record "scene отказ посреди миссии" ok "$(tr '\n' ' ' < "$dir/failing-outcome.txt")"
        else
            record "scene отказ посреди миссии" отказ "$(tr '\n' ' ' < "$dir/failing-outcome.txt")"
        fi
    else
        record "scene отказ посреди миссии" отказ "координатор не закончил: $(tail -3 "$dir/failing.log" | tr '\n' ' ')"
        stop_group "$coordinator" 20
    fi

    if [[ -n $rec_pid ]]; then
        wait_for "$rec_pid" 60
        mkdir -p "$MEDIA"
        if [[ -s "$dir/screen.mp4" ]] && ffmpeg -y -loglevel error -i "$dir/screen.mp4" \
            -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$MEDIA/phase8-mission.gif"; then
            record "media phase8-mission.gif" ok "$(du -h "$MEDIA/phase8-mission.gif" | cut -f1)"
        else
            record "media phase8-mission.gif" отказ "$(tail -3 "$dir/screen.log" 2>/dev/null | tr '\n' ' ')"
        fi
    fi

    kill -INT "$pgid" 2>/dev/null
    if wait_for "$pgid" 40; then
        record "scene остановка" ok "sim run вышел"
    else
        record "scene остановка" отказ "sim run не вышел за 40 с"
    fi
    stop_group "$pgid" 5
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
    write_probes
    for s in "${steps[@]}"; do
        case "$s" in
            env)   step_env ;;
            reqs)  step_reqs ;;
            scene) step_scene ;;
        esac
    done
    summary
}

main "$@"
