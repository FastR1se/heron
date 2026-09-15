#!/usr/bin/env bash
#
# Phase 4, stage 7: the system. arm6 through ros2_control and G1 through its vendor's DDS
# in one Gazebo world, driven one after the other (docs/plans/phase4-dds-bridge.md,
# "Этап 7"; ADR-0002, ADR-0034, ADR-0039).
#
# Built like scripts/phase4-stage5.sh: every check stands on a confirmation taken by a
# different tool than the one under test. The G1 by a DDS witness, the arm by a ROS 2 node
# of its own, the controllers by `ros2 control`, the middleware off /proc maps. The two
# `run` processes get no scene environment from this script: joining the world in its
# environment is the platform's job now (ADR-0039), and that is what is checked.
#
#   env    the execution venv synced, the G1 package and mcap in both venvs; the bridge's
#          workspace from stage 5 found
#   reqs   validate; sim run --dry-run: one launch, two managers, one environment; run
#          --dry-run of both robots names the world's environment, and a process in another
#          domain is refused
#   scene  sim run; controllers of both robots active; CycloneDDS in every process of the
#          world; run G1, then run arm1, each with a record; both moved, and G1 held still
#          while the arm ran; the platform stops the world
#
# Usage:
#   bash scripts/phase4-stage7.sh                    # all steps
#   GUI=1 bash scripts/phase4-stage7.sh scene        # with the simulator's window and RViz
#   GUI=1 REC=1 bash scripts/phase4-stage7.sh scene  # and the screen into docs/media
#
# Prerequisites: scripts/phase4-stage6.sh env (or stage 5's), which leave both venvs and the
# bridge's workspace in place. ffmpeg for REC.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="examples/scenarios/complex.yaml"
readonly G1_PROFILE="examples/robots/g1_23dof/profile.yaml"
readonly ARM_PROFILE="examples/robots/arm6.yaml"
readonly REGISTRY="examples/robots"
readonly PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly METRICS="$REPO_ROOT/scripts/phase4-stage6-metrics.py"
readonly MEDIA="$REPO_ROOT/docs/media"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
# The goals of humanoid.yaml and table.yaml: the scene has no task (complex.yaml).
readonly G1_TARGET="[0,0,0,0,0,0, 0,0,0,0,0,0, 0, -0.5,0.3,0,0.8,0, -0.5,-0.3,0,0.8,0]"
readonly ARM_TARGET="[0,-0.6,1.2,0,0.9,0]"
readonly DURATION=4

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase4-stage7}"
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
LOG="$STAGE_ROOT/stage7-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage7}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
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
    kill -TERM -- "-$pgid" 2>/dev/null
    while kill -0 -- "-$pgid" 2>/dev/null && (( waited < grace )); do
        sleep 1; waited=$(( waited + 1 ))
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null
    return 0
}

cleanup() {
    local pgid
    for pgid in "${STARTED[@]}"; do stop_group "$pgid" 5 >/dev/null 2>&1; done
}
trap cleanup EXIT

running() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ $state != Z* ]]; }
descendants() { local p; echo "$1"; for p in $(pgrep -P "$1"); do descendants "$p"; done; }

rmw_in_tree() {  # rmw_in_tree <pid>
    local p libs
    for p in $(descendants "$1"); do
        libs="$(grep -oE 'librmw_[a-z_]+\.so' "/proc/$p/maps" 2>/dev/null |
            grep -vE '^librmw_(implementation|dds_common)' | sort -u | tr '\n' ' ')"
        [[ -n $libs ]] && printf '%s[%s] %s\n' "$(cat "/proc/$p/comm" 2>/dev/null)" "$p" "$libs"
    done
}

# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/window.py" <<'PY'
# LowState from a DDS participant of its own for <window> s, after waiting up to <wait> s.
# Every contract joint's mean position, in the contract's order.
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
print(f"свидетель G1: LowState {len(got)}, первое через {first - start:.1f} с, tick {got[-1][0]}")
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
node = rclpy.create_node("stage7_arm_witness")
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
# Whether the joints went toward the target: before.json after.json '<target as JSON>' <limit rad>.
import json, sys
before, after = (json.load(open(path, encoding="utf-8"))["q_mean"] for path in sys.argv[1:3])
target, limit = json.loads(sys.argv[3]), float(sys.argv[4])
was = [abs(q - t) for q, t in zip(before, target)]
now = [abs(q - t) for q, t in zip(after, target)]
worst = max(range(len(now)), key=now.__getitem__)
print(f"до цели было {max(was):.3f} рад, стало {now[worst]:.3f} (сустав {worst})")
sys.exit(0 if now[worst] <= limit and now[worst] <= 0.5 * max(was) else 1)
PY

    cat > "$PY_DIR/still.py" <<'PY'
# Whether a robot held still between two windows: a.json b.json <limit rad>.
import json, sys
a, b = (json.load(open(path, encoding="utf-8"))["q_mean"] for path in sys.argv[1:3])
drift = max(abs(x - y) for x, y in zip(a, b))
print(f"наибольший сдвиг сустава {drift:.4f} рад")
sys.exit(0 if drift <= float(sys.argv[3]) else 1)
PY

    cat > "$PY_DIR/scene_env.py" <<'PY'
# The environment the generated launch sets, as shell exports -- for this script's own ROS 2
# witnesses and `ros2 control`, never for `run`.
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
    info "репозиторий: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l); DISPLAY ${DISPLAY:-не задан}; GUI=$GUI REC=$REC"
    local ok=1
    [[ -f $ROS_SETUP ]] || { record env отказ "нет $ROS_SETUP"; ok=0; }
    [[ -f $WS/install/setup.bash ]] || { record env отказ "нет $WS: bash scripts/phase4-stage5.sh env"; ok=0; }
    if [[ -x $EXEC_PY ]]; then
        clean "cd '$REPO_ROOT' && uv sync --extra dev && uv pip install --python '$EXEC_PY' -e '$PACKAGE'" ||
            { record env отказ "uv sync или установка пакета G1 не прошли"; ok=0; }
    else
        record env отказ "нет $EXEC_VENV"; ok=0
    fi
    if [[ -x $OWN_PY ]] && "$OWN_PY" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; then
        clean "'$OWN_PY' -m pip install -q --no-cache-dir pydantic pyyaml typer packaging mcap && '$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$PACKAGE'" ||
            { record env отказ "установка в $VENV_OWN не прошла"; ok=0; }
    else
        record env отказ "нет $VENV_OWN с SDK: bash scripts/phase4-probe.sh 0.5"; ok=0
    fi
    clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" > /dev/null ||
        { record env отказ "описания производителя не загрузились"; ok=0; }
    (( ok )) && record env ok "предусловия на месте"
}

step_reqs() {
    local out="$STAGE_ROOT/reqs" code
    step "reqs — validate сцены комплекса"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' validate '$SCENARIO' -R '$REGISTRY'" > "$out-validate.txt" 2>&1
    code=$?
    cat "$out-validate.txt"
    (( code == 0 )) && record "reqs validate" ok "0 ошибок; $(grep -c '^внимание' "$out-validate.txt") предупреждений" ||
        record "reqs validate" "отказ, код $code" "$(grep -m1 '^ошибка' "$out-validate.txt")"

    step "reqs — sim run --dry-run: один запуск, два менеджера, одно окружение"
    rm -rf "$STAGE_ROOT/dry"
    clean "$WS_ENV && cd '$REPO_ROOT' && '$EXEC_CLI' sim run '$SCENARIO' --into '$STAGE_ROOT/dry' -R '$REGISTRY' --dry-run" > "$out-dry.txt" 2>&1
    code=$?
    cat "$out-dry.txt"
    local launch="$STAGE_ROOT/dry/bringup.launch.xml"
    if (( code == 0 )) && [[ -f $launch ]]; then
        local spawners envs
        spawners=$(grep -c 'exec="spawner"' "$launch"); envs=$(grep -c '<set_env ' "$launch")
        if (( spawners == 4 && envs == 3 )) && grep -qF LowLevelBridge "$STAGE_ROOT/dry/g1/controllers.yaml" &&
            grep -qF '/arm1/controller_manager' "$STAGE_ROOT/dry/arm1/controllers.yaml"; then
            record "reqs sim run --dry-run" ok "спаунеров 4 (по два на робота), set_env 3, менеджеры /g1 и /arm1"
        else
            record "reqs sim run --dry-run" отказ "спаунеров $spawners, set_env $envs"
        fi
    else
        record "reqs sim run --dry-run" "отказ, код $code" "$(tail -1 "$out-dry.txt")"
    fi

    step "reqs — run --dry-run обоих роботов: окружение мира в плане (ADR-0039)"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' run '$SCENARIO' -R '$REGISTRY' --robot arm1 --skill move_joints -g 'target=$ARM_TARGET' -g duration=$DURATION --dry-run" > "$out-arm.txt" 2>&1
    clean "cd '$REPO_ROOT' && '$OWN_CLI' run '$SCENARIO' -R '$REGISTRY' --robot g1 --skill move_joints -g 'target=$G1_TARGET' -g duration=$DURATION --dry-run" > "$out-g1.txt" 2>&1
    cat "$out-arm.txt" "$out-g1.txt"
    if grep -qE '^окружение .*ROS_DOMAIN_ID=1' "$out-arm.txt" && grep -qE '^окружение .*ROS_DOMAIN_ID=1' "$out-g1.txt"; then
        record "reqs run --dry-run" ok "оба плана называют окружение мира"
    else
        record "reqs run --dry-run" отказ "строки окружения нет в плане одного из роботов"
    fi
    "${BASE_ENV[@]}" ROS_DOMAIN_ID=0 bash --noprofile --norc -c "cd '$REPO_ROOT' && '$EXEC_CLI' run '$SCENARIO' -R '$REGISTRY' --robot arm1 --skill move_joints -g 'target=$ARM_TARGET' --dry-run" > "$out-other-domain.txt" 2>&1
    code=$?
    if (( code == 1 )) && grep -qF "ROS_DOMAIN_ID: у процесса '0', у мира '1'" "$out-other-domain.txt"; then
        record "reqs чужой домен" ok "run в домене 0 отказывает до подключения и называет оба значения"
    else
        record "reqs чужой домен" отказ "код $code: $(tail -2 "$out-other-domain.txt" | tr '\n' ' ')"
    fi
}

step_scene() {
    local dir="$STAGE_ROOT/scene" pgid code waited=0 flags="" reader rec_pid=""
    rm -rf "$dir"; mkdir -p "$dir"
    step "scene — один мир, два робота, два разных интерфейса"
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

    start_group "$dir/g1-before.txt" "cd '$REPO_ROOT' && exec timeout 220 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 180 1 '$dir/g1-before.json'"
    reader=$LAST_GROUP
    while running "$reader"; do
        running "$pgid" || { stop_group "$reader" 5; break; }
        sleep 1
    done
    cat "$dir/g1-before.txt"
    if [[ ! -s "$dir/g1-before.json" ]]; then
        record "scene состояние G1" отказ "$(tail -1 "$dir/g1-before.txt")"
        tail -60 "$dir/sim.log" | sed 's/^/      /'
        stop_group "$pgid" 30; return
    fi
    record "scene состояние G1" ok "$(tail -1 "$dir/g1-before.txt")"

    clean "$witness_env && ros2 daemon stop >/dev/null 2>&1; timeout 40 ros2 control list_controllers -c /g1/controller_manager; timeout 40 ros2 control list_controllers -c /arm1/controller_manager" > "$dir/controllers.txt" 2>&1
    sed 's/^/      /' "$dir/controllers.txt"
    local active
    active=$(grep -cE '^(low_level_bridge|joint_state_broadcaster|joint_position_controller)\b.*\bactive\b' "$dir/controllers.txt")
    if (( active == 4 )) && grep -qE '^low_level_bridge\b.*\bactive\b' "$dir/controllers.txt" && grep -qE '^joint_position_controller\b.*\bactive\b' "$dir/controllers.txt"; then
        record "scene контроллеры" ok "g1: low_level_bridge и broadcaster, arm1: joint_position_controller и broadcaster — все active"
    else
        record "scene контроллеры" отказ "активных из четырёх: $active — два менеджера ros2_control в одном мире Gazebo"
        tail -40 "$dir/sim.log" | sed 's/^/      /'
    fi

    rmw_in_tree "$pgid" > "$dir/rmw.txt"
    sed 's/^/      /' "$dir/rmw.txt"
    local total good
    total=$(wc -l < "$dir/rmw.txt"); good=$(grep -c librmw_cyclonedds_cpp "$dir/rmw.txt")
    (( total > 0 && good == total )) && record "scene рантайм" ok "процессов мира с rmw $total, у всех CycloneDDS" ||
        record "scene рантайм" отказ "процессов с rmw $total, на CycloneDDS $good"

    clean "$witness_env && timeout 60 '$EXEC_PY' '$PY_DIR/arm_state.py' '$ARM_PROFILE' /arm1/joint_states 40 '$dir/arm-before.json'" > "$dir/arm-before.txt" 2>&1
    cat "$dir/arm-before.txt"

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

    # G1 first, from the adapter's venv; no scene environment here -- run applies it.
    clean "cd '$REPO_ROOT' && timeout 120 '$OWN_CLI' run '$SCENARIO' -R '$REGISTRY' --robot g1 --skill move_joints -g 'target=$G1_TARGET' -g duration=$DURATION --record '$dir/g1.mcap'" > "$dir/run-g1.txt" 2>&1
    code=$?
    cat "$dir/run-g1.txt"
    (( code == 0 )) && record "scene навык G1" ok "$(grep -m1 '^выполнено' "$dir/run-g1.txt")" ||
        record "scene навык G1" "отказ, код $code" "$(grep -E '^(отказ|причина|g1:)' "$dir/run-g1.txt" | tr '\n' ' ' | grep . || tail -1 "$dir/run-g1.txt")"
    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 30 1 '$dir/g1-after.json'" > "$dir/g1-after.txt" 2>&1
    if "$OWN_PY" "$PY_DIR/moved.py" "$dir/g1-before.json" "$dir/g1-after.json" "$G1_TARGET" 1.5 > "$dir/g1-moved.txt" 2>&1; then
        record "scene движение G1" ok "$(tail -1 "$dir/g1-moved.txt")"
    else
        record "scene движение G1" отказ "$(tail -1 "$dir/g1-moved.txt")"
    fi

    # Then the arm, from the execution venv with ROS 2 sourced and nothing else (ADR-0039).
    clean "$ROS_ENV && cd '$REPO_ROOT' && timeout 120 '$EXEC_CLI' run '$SCENARIO' -R '$REGISTRY' --robot arm1 --skill move_joints -g 'target=$ARM_TARGET' -g duration=$DURATION --record '$dir/arm1.mcap'" > "$dir/run-arm1.txt" 2>&1
    code=$?
    cat "$dir/run-arm1.txt"
    (( code == 0 )) && record "scene навык arm1" ok "$(grep -m1 '^выполнено' "$dir/run-arm1.txt")" ||
        record "scene навык arm1" "отказ, код $code" "$(grep -E '^(отказ|причина|arm1:)' "$dir/run-arm1.txt" | tr '\n' ' ' | grep . || tail -3 "$dir/run-arm1.txt" | tr '\n' ' ')"
    sleep 1
    clean "$witness_env && timeout 60 '$EXEC_PY' '$PY_DIR/arm_state.py' '$ARM_PROFILE' /arm1/joint_states 20 '$dir/arm-after.json'" > "$dir/arm-after.txt" 2>&1
    cat "$dir/arm-after.txt"
    if [[ -s "$dir/arm-before.json" && -s "$dir/arm-after.json" ]] &&
        "$EXEC_PY" "$PY_DIR/moved.py" "$dir/arm-before.json" "$dir/arm-after.json" "$ARM_TARGET" 0.1 > "$dir/arm-moved.txt" 2>&1; then
        record "scene движение arm1" ok "$(tail -1 "$dir/arm-moved.txt")"
    else
        record "scene движение arm1" отказ "$(tail -1 "$dir/arm-moved.txt" 2>/dev/null || tail -1 "$dir/arm-after.txt")"
    fi

    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/window.py' '$G1_PROFILE' 30 1 '$dir/g1-after-arm.json'" > "$dir/g1-after-arm.txt" 2>&1
    if "$OWN_PY" "$PY_DIR/still.py" "$dir/g1-after.json" "$dir/g1-after-arm.json" 0.05 > "$dir/g1-still.txt" 2>&1; then
        record "scene G1 во время arm1" ok "$(tail -1 "$dir/g1-still.txt") — команды манипулятора до G1 не дошли"
    else
        record "scene G1 во время arm1" отказ "$(tail -1 "$dir/g1-still.txt")"
    fi

    local robot
    for robot in g1 arm1; do
        [[ -s "$dir/$robot.mcap" ]] && "$EXEC_PY" "$METRICS" run "$dir/$robot.mcap" --label "complex-$robot" --out "$dir/$robot-metrics.json" 2>&1 | sed 's/^/      /'
    done

    if [[ -n $rec_pid ]]; then
        waited=0
        while running "$rec_pid" && (( waited < 60 )); do sleep 1; waited=$(( waited + 1 )); done
        mkdir -p "$MEDIA"
        if [[ -s "$dir/screen.mp4" ]] && ffmpeg -y -loglevel error -i "$dir/screen.mp4" \
            -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$MEDIA/phase4-complex.gif"; then
            record "media phase4-complex.gif" ok "$(du -h "$MEDIA/phase4-complex.gif" | cut -f1)"
        else
            record "media phase4-complex.gif" отказ "$(tail -3 "$dir/screen.log" 2>/dev/null | tr '\n' ' ')"
        fi
    fi

    local tree left="" p
    tree="$(descendants "$pgid" | tr '\n' ' ')"
    kill -INT "$pgid" 2>/dev/null
    waited=0
    while running "$pgid" && (( waited < 40 )); do sleep 1; waited=$(( waited + 1 )); done
    for p in $tree; do running "$p" && left+="$p "; done
    if ! running "$pgid" && [[ -z $left ]]; then
        record "scene остановка" ok "sim run вышел за $waited с, процессов мира не осталось"
    else
        record "scene остановка" отказ "через $waited с осталось: ${left:-—}"
        # shellcheck disable=SC2086
        [[ -n $left ]] && kill -KILL $left 2>/dev/null
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
