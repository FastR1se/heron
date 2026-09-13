#!/usr/bin/env bash
#
# Phase 4, stage 1: what only the Linux machine can show before the DDS adapter is
# written (docs/plans/phase4-dds-bridge.md, "Этап 1"; journal 2026-09-13, session 8).
#
# Built like scripts/phase4-probe.sh, and on its lesson (session 6): every check
# stands on a confirmation of its own, taken by a different tool than the one
# under test, so a defect in one probe can neither pass nor fail another. Nothing
# stops at the first failure. The table at the end is a summary; the lines above
# it are the evidence, and all of it goes into one log -- that file is what to
# send back.
#
# Steps:
#   env      versions, commit, prerequisites
#   reqs     environment requirements (ADR-0025) on a real machine: `env check`
#            with and without ROS 2 sourced, confirmed by `ros2 pkg prefix` and by
#            imports; a declaration shaped like the G1 package's (the package does
#            not exist yet) against the environments phase4-probe.sh built,
#            confirmed by `dpkg-query`, `pip show` and imports
#   scenes   arm6 (table) and G1 (humanoid) through ros2_control on Fast DDS, then
#            on rmw_cyclonedds_cpp: the runtime change measured apart from new code.
#            Each run confirms that state arrives, which runtime the processes
#            actually loaded (/proc/*/maps), the task, and motion toward the goal
#            read off joint_states rather than taken from the task's own verdict
#   oneside  ADR-0023 p. 3: a cyclonedds 0.10.2 participant with peer 127.0.0.1 and
#            multicast off on lo, against one configured the way the C++ SDK
#            configures itself (NetworkInterface lo, no peers). Both directions,
#            both start orders, and a control with no peers on either side
#   twin     the vendor's unitree_mujoco, built at the pinned commits and started
#            with -r g1 -s scene_23dof.xml -i 1 -n lo; the peered participant reads
#            LowState and moves one joint through LowCmd; the unpeered one is the
#            control. Also the twin-shaped requirements: a program and DISPLAY
#
# Usage:
#   bash scripts/phase4-stage1.sh                 # all steps, in order
#   bash scripts/phase4-stage1.sh oneside twin    # selected steps
#   STAGE_ROOT=/data/stage1 bash scripts/phase4-stage1.sh
#
# Prerequisites: scripts/phase0.sh (ROS 2 Jazzy, Gazebo, the execution venv) and
# scripts/phase4-probe.sh steps 0.8 and 0.5 (unitree_hg, the cyclonedds venvs),
# which this run reuses instead of rebuilding. The twin step needs a graphical
# session: the vendor's simulator has no headless mode (ADR-0023).
#
# Writes outside $STAGE_ROOT: apt packages a build cannot run without, each named
# by the step that needs it; the platform installed into the two probe venvs of
# $PROBE_ROOT (the shape the DDS adapter's environment will have); `uv sync` in the
# repository when the execution venv lacks a new core dependency.
#
# Roughly 1.5 GiB and 20-40 minutes: the scenes run four times, the SDK and the
# simulator are built once.

set -uo pipefail

# ---------------------------------------------------------------------------
# Constants. Vendor commits are the ones ADR-0023 was written against.
# ---------------------------------------------------------------------------

readonly ROS_DISTRO_NAME="jazzy"
readonly ROS_SETUP="/opt/ros/$ROS_DISTRO_NAME/setup.bash"

# The command a user types is the package name (pyproject.toml, [project.scripts]).
# Spelled once, here; the rename procedure of section 17 touches this line.
readonly APP="heron"

readonly SDK2_URL="https://github.com/unitreerobotics/unitree_sdk2.git"
readonly SDK2_COMMIT="9754cd153af3da471b0fe5f3aa535e426fb11db3"
readonly TWIN_URL="https://github.com/unitreerobotics/unitree_mujoco.git"
readonly TWIN_COMMIT="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
readonly MUJOCO_VERSION="3.3.6"                # the version the vendor README names
readonly MUJOCO_URL="https://github.com/google-deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz"

readonly DDS_DOMAIN=1                          # the twin's default; the robot uses 0
readonly SCENE_DOMAIN=42                       # scenes kept apart from the DDS participants
readonly SCENES="table humanoid"
readonly RUNTIMES="rmw_fastrtps_cpp rmw_cyclonedds_cpp"   # the old runtime first: a baseline on this commit
readonly ALL_STEPS="env reqs scenes oneside twin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly CLI="$EXEC_VENV/bin/$APP"
readonly G1_URDF="$REPO_ROOT/examples/robots/g1_23dof/vendor/unitree_ros/robots/g1_description/g1_23dof.urdf"

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase4-stage1}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
readonly STAGE_ROOT PROBE_ROOT
readonly PY_DIR="$STAGE_ROOT/py"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly VENV_NEW="$PROBE_ROOT/venv-cyclonedds11"
readonly PROBE_WS="$PROBE_ROOT/ws"
readonly SDK2_DIR="$STAGE_ROOT/unitree_sdk2"
readonly TWIN_DIR="$STAGE_ROOT/unitree_mujoco"
readonly MUJOCO_DIR="$STAGE_ROOT/mujoco-$MUJOCO_VERSION"
readonly TWIN_BIN="$TWIN_DIR/simulate/build/unitree_mujoco"

mkdir -p "$STAGE_ROOT" "$PY_DIR"
LOG="$STAGE_ROOT/stage1-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------
# Output and bookkeeping. Messages address a human, so they are in Russian.
# ---------------------------------------------------------------------------

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <check> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

# Everything runs in an environment that inherits nothing but HOME and a plain
# PATH: what the calling shell has sourced cannot decide an outcome, and which
# commands see ROS 2 is written in the command.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage1}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean()     { "${BASE_ENV[@]}" bash --noprofile --norc -c "$1"; }
clean_gui() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

readonly ROS_ENV="source '$ROS_SETUP'"

# A long-running process gets a process group of its own, so stopping a scene
# stops the launch and everything it started. setsid in a background job does not
# fork, so the pid it leaves in $! is the group.
STARTED=()
LAST_GROUP=""
start_group() {  # start_group <log> <gui: 0|1> <command>
    if (( $2 )); then
        setsid "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$3" > "$1" 2>&1 < /dev/null &
    else
        setsid "${BASE_ENV[@]}" bash --noprofile --norc -c "$3" > "$1" 2>&1 < /dev/null &
    fi
    LAST_GROUP=$!
    STARTED+=("$LAST_GROUP")
}

stop_group() {  # stop_group <pgid> <grace seconds>
    local pgid="$1" grace="$2" waited=0
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    kill -TERM -- "-$pgid" 2>/dev/null
    while kill -0 -- "-$pgid" 2>/dev/null && (( waited < grace )); do
        sleep 1
        waited=$(( waited + 1 ))
    done
    if kill -0 -- "-$pgid" 2>/dev/null; then
        info "группа $pgid не завершилась за $grace с после TERM — KILL"
        kill -KILL -- "-$pgid" 2>/dev/null
    fi
}

cleanup() {
    local pgid
    for pgid in "${STARTED[@]}"; do stop_group "$pgid" 5 >/dev/null 2>&1; done
}
trap cleanup EXIT

wait_for_line() {  # wait_for_line <file> <text> <seconds> <pid>
    local waited=0
    until grep -qF "$2" "$1" 2>/dev/null; do
        kill -0 "$4" 2>/dev/null || return 1
        (( waited >= $3 )) && return 1
        sleep 1
        waited=$(( waited + 1 ))
    done
}

# Fetch one commit into a directory, reusing a checkout that is already there.
fetch_pinned() {  # fetch_pinned <url> <commit> <dir>
    local url="$1" ref="$2" dir="$3"
    if [[ -d "$dir/.git" ]]; then
        info "уже загружено: $dir @ $(git -C "$dir" rev-parse --short HEAD)"
        if [[ "$(git -C "$dir" rev-parse HEAD)" != "$ref" ]]; then
            info "в каталоге другой коммит, ожидается $ref; удалите каталог, если замена намеренна"
            return 1
        fi
        return 0
    fi
    mkdir -p "$dir" &&
        git -C "$dir" init -q &&
        git -C "$dir" remote add origin "$url" &&
        git -C "$dir" fetch -q --depth 1 origin "$ref" &&
        git -C "$dir" checkout -q FETCH_HEAD &&
        info "загружено: $url @ $(git -C "$dir" rev-parse HEAD)"
}

# Install what a step cannot run without, saying what and why. Never silent.
ensure_apt() {  # ensure_apt <step> <package>...
    local what="$1" missing=() p
    shift
    for p in "$@"; do
        dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    (( ${#missing[@]} )) || return 0
    info "ставятся пакеты, без которых шаг $what не идёт: ${missing[*]}"
    sudo apt-get install -y "${missing[@]}" ||
        { record "$what" отказ "apt не поставил ${missing[*]}"; return 1; }
}

twins_group() {
    "$EXEC_PY" -c "from importlib.metadata import entry_points; print(' '.join(sorted(e.name for e in entry_points(group='$APP.twins'))) or 'пусто')" 2>/dev/null ||
        echo "не читается"
}

# Running, not merely present: a child that has exited stays a zombie until it is
# reaped, and kill -0 answers yes for a zombie.
running() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ $state != Z* ]]; }

have_sdk_venv() { [[ -x "$1/bin/python" ]] && "$1/bin/python" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; }

# The platform inside a probe venv, without its optional groups: the DDS adapter's
# process will be exactly this -- the platform and the 0.10.2 binding, no rclpy.
ensure_platform() {  # ensure_platform <venv>
    "$1/bin/python" -c "import $APP.validator.environment" >/dev/null 2>&1 && return 0
    info "в $1 ставится платформа (зависимости ядра и пакет без групп): так будет устроено окружение DDS-адаптера"
    clean "'$1/bin/pip' install -q --no-cache-dir pydantic pyyaml typer packaging && '$1/bin/pip' install -q --no-cache-dir --no-deps -e '$REPO_ROOT'"
}

# ---------------------------------------------------------------------------
# Python probes, written out once so that no step quotes Python in bash.
# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/declared.py" <<'PY'
# What the built-in sides declare, read from the platform rather than copied here.
import importlib, sys
app, side = sys.argv[1], sys.argv[2]
if side == "twin":
    print(*importlib.import_module(f"{app}.generators.ros2_control").TWIN_PACKAGES)
else:
    print(*importlib.import_module(f"{app}.ros2.names").ADAPTER_PACKAGES)
PY

    cat > "$PY_DIR/reqs_vendor.py" <<'PY'
# Requirements in the shape the G1 package will declare them (ADR-0025,
# consequences). A stand-in: the package does not exist until stage 3, and what
# is checked here is the form and the local machine, not the package.
import importlib, sys
app, side = sys.argv[1], sys.argv[2]
env = importlib.import_module(f"{app}.core.environment")
check = importlib.import_module(f"{app}.validator.environment")
K = env.RequirementKind
DDS = "0.10.2 падает при обнаружении участника 0.11+ (ADR-0022)"
SIDES = {
    "adapter": [
        env.needs(K.PYTHON, "cyclonedds", version="==0.10.2",
                  hint="обвязка 0.10.2 из исходника против своей C-библиотеки (ADR-0022)"),
        env.needs(K.PYTHON, "unitree_sdk2py", hint="pip install --no-deps -e unitree_sdk2_python"),
        env.forbids(K.PYTHON, "cyclonedds", version=">=0.11", because=DDS,
                    hint="pip uninstall cyclonedds, затем обвязка 0.10.2"),
    ],
    "bridge": [
        env.needs(K.ROS2, "unitree_hg",
                  hint="colcon build --packages-select unitree_hg; source install/setup.bash"),
        env.needs(K.ROS2, "rmw_cyclonedds_cpp", hint="sudo apt install ros-jazzy-rmw-cyclonedds-cpp"),
        env.forbids(K.ROS2, "cyclonedds", version=">=0.11", because=DDS,
                    hint="дистрибутив ROS 2 с CycloneDDS 0.10.x (Jazzy)"),
    ],
    "twin": [
        env.needs(K.EXECUTABLE, "unitree_mujoco",
                  hint="сборка unitree_mujoco/simulate; каталог сборки на PATH"),
        env.needs(K.VARIABLE, "DISPLAY", hint="графическая сессия: окно GLFW обязательно (ADR-0023)"),
    ],
}
declared = SIDES[side]
report = check.check_requirements(declared, where=f"stand-in:{side}", machine=check.LocalMachine())
for finding in report:
    print(finding)
print(f"{side}: требований {len(declared)}, замечаний {len(report.findings)}, "
      f"коды {sorted(report.codes())}")
sys.exit(0 if report.ok else 1)
PY

    cat > "$PY_DIR/joints.py" <<'PY'
# Joint positions of the scene's driven robot, in contract order, off joint_states.
# Also how fast states arrive and how much world time passes per wall second: the
# latter from header stamps, so it does not depend on how fast this loop reads.
import importlib, json, sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

app, scenario_path, registry_dir, out = sys.argv[1:5]
window, deadline = float(sys.argv[5]), float(sys.argv[6])
scenario = importlib.import_module(f"{app}.scenario.loader").load_scenario(scenario_path)
registry = importlib.import_module(f"{app}.registry.registry").Registry([registry_dir])
robot = next(item for item in scenario.robots if item.role != "static")
order = list(registry.load(robot.profile).contract.joint_order)
topic = f"/{robot.id}/joint_states"

rclpy.init()
node = rclpy.create_node("stage1_joints")
got = []
node.create_subscription(JointState, topic, lambda m: got.append((time.monotonic(), m)),
                         qos_profile_sensor_data)
start = time.monotonic()
while not got and time.monotonic() - start < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
if not got:
    print(f"{topic}: состояния нет за {deadline:g} с; издателей видно {node.count_publishers(topic)}")
    sys.exit(1)
while time.monotonic() - got[0][0] < window:
    rclpy.spin_once(node, timeout_sec=0.05)
(t0, m0), (t1, m1) = got[0], got[-1]
wall = max(t1 - t0, 1e-9)
stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
index = {name: i for i, name in enumerate(m1.name)}
absent = [name for name in order if name not in index]
if absent:
    print(f"{topic}: в сообщении нет суставов контракта: {absent}")
    sys.exit(1)
target = scenario.task.goal.get("target") if scenario.task else None
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"topic": topic, "order": order, "target": target,
               "positions": [m1.position[index[name]] for name in order]}, handle)
print(f"{topic}: первое через {t0 - start:.1f} с; за {wall:.1f} с принято {len(got)} "
      f"({(len(got) - 1) / wall:.0f} Гц приёма), время мира/стенное {(stamp(m1) - stamp(m0)) / wall:.2f}")
node.destroy_node()
rclpy.shutdown()
PY

    cat > "$PY_DIR/moved.py" <<'PY'
# Whether the robot moved toward the goal: distance to the target before and after.
import json, sys
before, after = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3])
target = before["target"]
if target is None or len(target) != len(before["positions"]):
    print("у сценария нет цели target той же длины, что контракт")
    sys.exit(1)
gap = lambda positions: [abs(p - t) for p, t in zip(positions, target)]
was, now = max(gap(before["positions"])), gap(after["positions"])
worst = max(range(len(now)), key=now.__getitem__)
travel = max(abs(a - b) for a, b in zip(after["positions"], before["positions"]))
print(f"до цели было {was:.3f} рад, стало {now[worst]:.3f} рад ({before['order'][worst]}); "
      f"наибольшее перемещение {travel:.3f} рад")
sys.exit(0 if now[worst] <= 0.5 * was else 1)
PY

    cat > "$PY_DIR/dds_configs.py" <<'PY'
# The two CycloneDDS configurations under test (ADR-0022 p. 5, ADR-0023 p. 3).
PEERS_LO = (
    '<CycloneDDS><Domain Id="any"><General>'
    '<Interfaces><NetworkInterface name="lo"/></Interfaces>'
    '<AllowMulticast>false</AllowMulticast></General>'
    '<Discovery><ParticipantIndex>auto</ParticipantIndex>'
    '<Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
    '</Domain></CycloneDDS>'
)
# Extracted from libunitree_sdk2.a at 9754cd1 (journal, session 7), interface
# substituted: what ChannelFactory::Init(domain, "lo") hands to cyclonedds-cxx.
SDK_CXX_LO = (
    '<CycloneDDS><Domain Id="any"><General><Interfaces>'
    '<NetworkInterface name="lo" priority="default" multicast="default" />'
    '</Interfaces></General></Domain></CycloneDDS>'
)
CONFIGS = {"peers-lo": PEERS_LO, "sdk-cxx-lo": SDK_CXX_LO}
PY

    cat > "$PY_DIR/dds_peer.py" <<'PY'
# One participant of a link: the robot's side writes LowCmd and reads LowState,
# the twin's side the other way round. Exit 0 means something was received. The
# writer's matched count is printed as the other side's witness.
import sys, time
from cyclonedds.core import Listener
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from dds_configs import CONFIGS

role, config, domain_id, duration = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
if role == "robot":
    out_name, out_type, make, in_name, in_type = (
        "rt/lowcmd", LowCmd_, unitree_hg_msg_dds__LowCmd_, "rt/lowstate", LowState_)
else:
    out_name, out_type, make, in_name, in_type = (
        "rt/lowstate", LowState_, unitree_hg_msg_dds__LowState_, "rt/lowcmd", LowCmd_)

matched = {"readers": 0, "writers": 0}
def on_publication_matched(writer, status):
    matched["readers"] = max(matched["readers"], status.current_count)
def on_subscription_matched(reader, status):
    matched["writers"] = max(matched["writers"], status.current_count)

domain = Domain(domain_id, CONFIGS[config])
participant = DomainParticipant(domain_id)
writer = DataWriter(participant, Topic(participant, out_name, out_type),
                    listener=Listener(on_publication_matched=on_publication_matched))
reader = DataReader(participant, Topic(participant, in_name, in_type),
                    listener=Listener(on_subscription_matched=on_subscription_matched))
print(f"{role}/{config}: пишет {out_name}, читает {in_name}, домен {domain_id}, {duration:g} с",
      flush=True)
msg = make()
msg.mode_machine = 4
written = received = 0
end = time.monotonic() + duration
while time.monotonic() < end:
    if role == "robot":
        msg.motor_cmd[15].kp = 40.0
    else:
        msg.tick = written
    writer.write(msg)
    written += 1
    received += len(reader.take(N=64))
    time.sleep(0.01)
print(f"{role}/{config}: записано {written} в {out_name}, читателей сопоставлено {matched['readers']}; "
      f"принято {received} из {in_name}, писателей сопоставлено {matched['writers']}", flush=True)
sys.exit(0 if received else 1)
PY

    cat > "$PY_DIR/twin_link.py" <<'PY'
# The adapter's side against the vendor's twin: listen to LowState, then hold every
# joint where it is and move one. Exit 0: state and motion; 2: state, no motion;
# 1: no state. Motion is the confirmation that LowCmd arrived and was applied.
import sys, time
from cyclonedds.core import Listener
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from dds_configs import CONFIGS

config, domain_id = sys.argv[1], int(sys.argv[2])
LISTEN_S, DRIVE_S = 3.0, 5.0
# Slots of the 23-DoF model in the 29-DoF layout (journal, session 5, fact 0.2), in
# contract order, with the gains of examples/robots/g1_23dof/profile.yaml.
SLOTS = [*range(0, 13), *range(15, 20), *range(22, 27)]
KP = [60, 60, 60, 100, 40, 40, 60, 60, 60, 100, 40, 40, 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
KD = [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
PROBE, OFFSET = 15, 0.6   # left shoulder pitch, radians

matched = {"readers": 0, "writers": 0}
def on_publication_matched(writer, status):
    matched["readers"] = max(matched["readers"], status.current_count)
def on_subscription_matched(reader, status):
    matched["writers"] = max(matched["writers"], status.current_count)

domain = Domain(domain_id, CONFIGS[config])
participant = DomainParticipant(domain_id)
reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_),
                    listener=Listener(on_subscription_matched=on_subscription_matched))
writer = DataWriter(participant, Topic(participant, "rt/lowcmd", LowCmd_),
                    listener=Listener(on_publication_matched=on_publication_matched))
states = []
def drain():
    for sample in reader.take(N=256):
        if hasattr(sample, "tick"):
            states.append((time.monotonic(), sample.tick, sample.mode_machine,
                           [motor.q for motor in sample.motor_state]))

start = time.monotonic()
while time.monotonic() - start < LISTEN_S:
    drain()
    time.sleep(0.005)
if len(states) < 2:
    print(f"{config}: LowState принято {len(states)} за {LISTEN_S:g} с, "
          f"писателей сопоставлено {matched['writers']}", flush=True)
    sys.exit(1)
(t0, k0, mode, _), (t1, k1, _, hold) = states[0], states[-1]
wall = t1 - t0
print(f"{config}: LowState {len(states)} за {wall:.2f} с ({len(states) / wall:.0f} Гц приёма), "
      f"писателей сопоставлено {matched['writers']}; tick +{k1 - k0} мс мира, фактор "
      f"{(k1 - k0) / 1000 / wall:.2f}; mode_machine {mode}", flush=True)

target = hold[PROBE] + OFFSET
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_pr = 0
cmd.mode_machine = mode
for slot, kp, kd in zip(SLOTS, KP, KD):
    motor = cmd.motor_cmd[slot]
    motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd = 1, hold[slot], 0.0, 0.0, float(kp), float(kd)
cmd.motor_cmd[PROBE].q = target
crc = CRC()
marker, sent = len(states), 0
drive_start = time.monotonic()
while time.monotonic() - drive_start < DRIVE_S:
    cmd.crc = crc.Crc(cmd)
    writer.write(cmd)
    sent += 1
    drain()
    time.sleep(0.002)
tail = time.monotonic() - 0.5
late = [q[PROBE] for (t, _, _, q) in states[marker:] if t >= tail]
end_q = sum(late) / len(late) if late else float("nan")
share = (end_q - hold[PROBE]) / OFFSET if late else 0.0
print(f"{config}: LowCmd записано {sent}, читателей сопоставлено {matched['readers']}; "
      f"q[{PROBE}] {hold[PROBE]:+.3f} -> {end_q:+.3f} рад при цели {target:+.3f}, "
      f"пройдено {share:.0%}", flush=True)
sys.exit(0 if share > 0.5 else 2)
PY
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

step_env() {
    step "Окружение"
    info "лог: $LOG"
    info "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}"), ядро $(uname -r), ядер $(nproc)"
    info "репозиторий: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l)"
    local p
    for p in ros-$ROS_DISTRO_NAME-rmw-fastrtps-cpp ros-$ROS_DISTRO_NAME-rmw-cyclonedds-cpp \
             ros-$ROS_DISTRO_NAME-cyclonedds ros-$ROS_DISTRO_NAME-gz-ros2-control; do
        info "$p: $(dpkg-query -W -f='${Version}' "$p" 2>/dev/null || echo 'не установлен')"
    done
    info "DISPLAY: ${DISPLAY:-не задан}"
    info "lo: $(ip -o link show lo 2>/dev/null | sed -E 's/.*<([^>]*)>.*/\1/')"

    git -C "$REPO_ROOT" status --short 2>/dev/null | sed 's/^/      /'

    local ok=1 before after
    [[ -f "$ROS_SETUP" ]] || { record env отказ "нет $ROS_SETUP (scripts/phase0.sh install ros2)"; ok=0; }
    if [[ ! -x "$EXEC_PY" ]]; then
        record env отказ "нет окружения исполнения $EXEC_VENV (scripts/phase0.sh install exec)"
        ok=0
    else
        # Always, not only when a dependency is missing: entry points are written at
        # install time, and run 1 went on an install older than the twin group
        # (journal, session 9). The run tests this commit, not an old install of it.
        before="$(twins_group)"
        info "группа двойников в окружении исполнения до синхронизации: $before"
        info "окружение исполнения синхронизируется с pyproject.toml этого коммита (uv sync, группы sim dev)"
        if clean "cd '$REPO_ROOT' && uv sync --extra sim --extra dev"; then
            after="$(twins_group)"
            info "после: $after"
            if [[ $after == пусто ]]; then
                record env отказ "после uv sync группа двойников пуста — встроенный двойник не установлен"; ok=0
            elif [[ $before != "$after" ]]; then
                record "env установка" "была устаревшей" "группа двойников до синхронизации: $before, после: $after"
            fi
        else
            record env отказ "uv sync не прошёл — причина выше"; ok=0
        fi
    fi
    [[ -x "$CLI" ]] || { record env отказ "нет $CLI"; ok=0; }
    info "из phase4-probe.sh: venv-own-lib $(have_sdk_venv "$VENV_OWN" && echo есть || echo нет), venv-cyclonedds11 $(have_sdk_venv "$VENV_NEW" && echo есть || echo нет), unitree_hg $([[ -f "$PROBE_WS/install/setup.bash" ]] && echo собран || echo нет)"
    (( ok )) && record env ok "предусловия на месте"
}

step_reqs() {
    if [[ ! -x "$CLI" || ! -f "$ROS_SETUP" ]]; then
        record reqs пропущено "нет окружения исполнения или ROS 2 (шаг env)"; return
    fi
    local out code confirm version

    step "reqs — env check сцены table, ROS 2 подключён"
    out="$STAGE_ROOT/reqs-table-sourced.txt"
    clean "$ROS_ENV && cd '$REPO_ROOT' && '$CLI' env check examples/scenarios/table.yaml -R examples/robots" > "$out" 2>&1
    code=$?
    cat "$out"
    # The witness: the ament CLI for packages, the interpreter itself for modules.
    confirm="$(clean "$ROS_ENV && for p in \$('$EXEC_PY' '$PY_DIR/declared.py' $APP twin); do ros2 pkg prefix \$p >/dev/null 2>&1 || echo ros2:\$p; done; for m in \$('$EXEC_PY' '$PY_DIR/declared.py' $APP adapter); do '$EXEC_PY' -c \"import \$m\" >/dev/null 2>&1 || echo python:\$m; done" 2>&1)"
    info "ros2 pkg prefix и импорт: ${confirm:-всё объявленное находится}"
    if (( code == 0 )) && [[ -z $confirm ]]; then
        record "reqs table, ROS 2 есть" ok "env check 0; ros2 pkg prefix и импорт находят всё объявленное"
    elif (( code != 0 )) && [[ -n $confirm ]]; then
        record "reqs table, ROS 2 есть" "совпало: не хватает" "env check $code; сторонние средства тоже не находят: $(echo $confirm)"
    else
        record "reqs table, ROS 2 есть" отказ "env check $code, сторонние средства: ${confirm:-всё находят} — проверка расходится с машиной"
    fi

    step "reqs — env check сцены table, ROS 2 не подключён"
    out="$STAGE_ROOT/reqs-table-unsourced.txt"
    clean "cd '$REPO_ROOT' && '$CLI' env check examples/scenarios/table.yaml -R examples/robots" > "$out" 2>&1
    code=$?
    cat "$out"
    if clean "command -v ros2" >/dev/null 2>&1 || clean "'$EXEC_PY' -c 'import rclpy'" >/dev/null 2>&1; then
        record "reqs table, ROS 2 нет" пропущено "ros2 или rclpy видны без подключения — это не машина без них"
    elif (( code == 1 )) && grep -qF '[env.missing]' "$out" && grep -qF 'controller_manager' "$out" && grep -qF 'source /opt/ros' "$out"; then
        record "reqs table, ROS 2 нет" ok "env check 1: $(grep -cF '[env.missing]' "$out") env.missing, называет controller_manager и source; ros2 и rclpy действительно не видны"
    else
        record "reqs table, ROS 2 нет" отказ "env check $code; ожидались env.missing, controller_manager и подсказка source"
    fi

    if [[ -f "$PROBE_WS/install/setup.bash" ]]; then
        step "reqs — объявление в форме моста G1: unitree_hg, rmw_cyclonedds_cpp, запрет cyclonedds >=0.11"
        out="$STAGE_ROOT/reqs-bridge-sourced.txt"
        clean "$ROS_ENV && source '$PROBE_WS/install/setup.bash' && '$EXEC_PY' '$PY_DIR/reqs_vendor.py' $APP bridge" > "$out" 2>&1
        code=$?
        cat "$out"
        confirm="$(clean "$ROS_ENV && source '$PROBE_WS/install/setup.bash' && ros2 pkg prefix unitree_hg && ros2 pkg prefix rmw_cyclonedds_cpp" 2>&1)"
        version="$(dpkg-query -W -f='${Version}' ros-$ROS_DISTRO_NAME-cyclonedds 2>/dev/null)"
        info "ros2 pkg prefix: $(echo $confirm); dpkg: cyclonedds ${version:-не установлен}"
        if (( code == 0 )) && [[ $(grep -c '^/' <<< "$confirm") -eq 2 && $version == 0.10.* ]]; then
            record "reqs мост, окружение есть" ok "$(tail -1 "$out"); ros2 pkg prefix находит оба, dpkg: cyclonedds $version"
        else
            record "reqs мост, окружение есть" отказ "$(tail -1 "$out"); ros2 pkg prefix: $(echo $confirm); dpkg: ${version:-—}"
        fi

        out="$STAGE_ROOT/reqs-bridge-unsourced.txt"
        clean "'$EXEC_PY' '$PY_DIR/reqs_vendor.py' $APP bridge" > "$out" 2>&1
        code=$?
        cat "$out"
        if (( code == 1 )) && [[ $(grep -cF '[env.missing]' "$out") -eq 2 ]] && ! grep -qF '[env.forbidden]' "$out" && ! clean "command -v ros2" >/dev/null 2>&1; then
            record "reqs мост, ROS 2 нет" ok "два env.missing, запрета нет — запрещённого нечего найти; ros2 не виден"
        else
            record "reqs мост, ROS 2 нет" отказ "$(tail -1 "$out")"
        fi
    else
        record "reqs мост" пропущено "unitree_hg не собран: bash scripts/phase4-probe.sh 0.8"
    fi

    local label venv want
    for label in own-lib cyclonedds11; do
        [[ $label == own-lib ]] && { venv="$VENV_OWN"; want="0.10.2"; } || { venv="$VENV_NEW"; want="11."; }
        if ! have_sdk_venv "$venv"; then
            record "reqs адаптер $label" пропущено "нет $venv: bash scripts/phase4-probe.sh 0.5"; continue
        fi
        step "reqs — объявление в форме адаптера G1 в окружении $label"
        ensure_platform "$venv" || { record "reqs адаптер $label" отказ "платформа не ставится в $venv"; continue; }
        out="$STAGE_ROOT/reqs-adapter-$label.txt"
        clean "'$venv/bin/python' '$PY_DIR/reqs_vendor.py' $APP adapter" > "$out" 2>&1
        code=$?
        cat "$out"
        version="$(clean "'$venv/bin/python' -m pip show cyclonedds" 2>/dev/null | sed -n 's/^Version: //p')"
        info "pip show cyclonedds: ${version:-нет}; импорт SDK: $(clean "'$venv/bin/python' -c 'import unitree_sdk2py'" >/dev/null 2>&1 && echo да || echo нет)"
        if [[ $label == own-lib ]]; then
            if (( code == 0 )) && [[ $version == "$want" ]]; then
                record "reqs адаптер $label" ok "$(tail -1 "$out"); pip show: $version"
            else
                record "reqs адаптер $label" отказ "$(tail -1 "$out"); pip show: ${version:-—}"
            fi
        elif (( code == 1 )) && grep -qF "env.forbidden" <<< "$(tail -1 "$out")" && grep -qF "env.version_unmet" <<< "$(tail -1 "$out")" && [[ $version == "$want"* ]]; then
            record "reqs адаптер $label" ok "запрещено и не та версия: $(tail -1 "$out"); pip show: $version"
        else
            record "reqs адаптер $label" отказ "$(tail -1 "$out"); pip show: ${version:-—}"
        fi
    done
}

# Processes of a group and the ROS 2 middleware each has loaded, read off its maps.
rmw_in_group() {  # rmw_in_group <pgid>
    local p libs
    for p in $(pgrep -g "$1"); do
        libs="$(grep -oE 'librmw_[a-z_]+\.so' "/proc/$p/maps" 2>/dev/null |
            grep -vE '^librmw_(implementation|dds_common)' | sort -u | tr '\n' ' ')"
        [[ -n $libs ]] && printf '%s[%s] %s\n' "$(cat "/proc/$p/comm" 2>/dev/null)" "$p" "$libs"
    done
}

real_time_factor() {  # real_time_factor <env command>
    local topic value
    topic="$(clean "$1 && timeout 15 gz topic -l" 2>/dev/null | grep -m1 -E '^/world/[^/]+/stats$')"
    [[ -n $topic ]] || { echo "темы /world/*/stats не видно"; return; }
    value="$(clean "$1 && timeout 15 gz topic -e -t '$topic' -n 1" 2>/dev/null | grep -m1 real_time_factor)"
    echo "${value:-за 15 с сообщения нет} ($topic)"
}

# Processes of a scene are recognised by a variable only the scene's commands
# carry, not by names: run 1 matched "spawner" in the arguments of the desktop's
# own gvfsd daemons and killed them.
readonly SCENE_MARK="PHASE4_STAGE1_SCENE=$$"

leftovers() {  # leftovers <label>
    local p left=()
    for p in $(pgrep -u "$(id -u)"); do
        grep -qzx "$SCENE_MARK" "/proc/$p/environ" 2>/dev/null && left+=("$p")
    done
    (( ${#left[@]} )) || return 0
    info "после остановки остались процессы сцены — убиваются, чтобы не попасть в следующий прогон:"
    ps -o pid=,args= -p "$(IFS=,; echo "${left[*]}")" | sed 's/^/      /'
    kill -KILL "${left[@]}" 2>/dev/null
    record "scenes $1 остановка" "остались процессы" "${#left[@]} шт. после TERM группы, убиты"
}

scene_once() {  # scene_once <scene> <rmw>
    local scene="$1" rmw="$2" label="$1 ${2#rmw_}" dir="$STAGE_ROOT/scenes/$1-$2"
    local env_cmd="$ROS_ENV && export RMW_IMPLEMENTATION=$rmw ROS_DOMAIN_ID=$SCENE_DOMAIN $SCENE_MARK && cd '$REPO_ROOT'"
    local code pgid reader want other total good bad
    rm -rf "$dir"
    mkdir -p "$dir"
    step "scenes — $scene на $rmw"
    clean "$ROS_ENV && export RMW_IMPLEMENTATION=$rmw ROS_DOMAIN_ID=$SCENE_DOMAIN && ros2 daemon stop" >/dev/null 2>&1

    start_group "$dir/sim.log" 0 "$env_cmd && exec '$CLI' sim run examples/scenarios/$scene.yaml --into '$dir/gen' -R examples/robots --headless --no-rviz"
    pgid=$LAST_GROUP
    info "sim run: группа $pgid, лог $dir/sim.log"

    # The reader waits up to 150 s for state, but not for a world that is gone:
    # run 1 waited the full time four times after sim run had refused the scene.
    start_group "$dir/before.txt" 0 "$env_cmd && exec timeout 200 '$EXEC_PY' '$PY_DIR/joints.py' $APP examples/scenarios/$scene.yaml examples/robots '$dir/before.json' 5 150"
    reader=$LAST_GROUP
    while running "$reader"; do
        if ! running "$pgid"; then
            info "sim run завершился раньше, чем пришло состояние"
            stop_group "$reader" 5
            break
        fi
        sleep 1
    done
    wait "$reader"
    code=$?
    cat "$dir/before.txt"
    if (( code != 0 )); then
        if running "$pgid"; then
            record "scenes $label состояние" отказ "$(tail -1 "$dir/before.txt")"
        else
            record "scenes $label состояние" "отказ: sim run завершился" "$(grep -m1 -E '^\[|не |ошибка' "$dir/sim.log" || tail -1 "$dir/sim.log")"
        fi
        info "хвост лога sim run:"
        tail -40 "$dir/sim.log" | sed 's/^/      /'
        stop_group "$pgid" 30
        leftovers "$label"
        return
    fi
    record "scenes $label состояние" ok "$(tail -1 "$dir/before.txt")"

    # The runtime is read off the processes, not off the variable that asked for it.
    rmw_in_group "$pgid" > "$dir/rmw.txt"
    sed 's/^/      /' "$dir/rmw.txt"
    if [[ $rmw == rmw_cyclonedds_cpp ]]; then want=librmw_cyclonedds_cpp; other=librmw_fastrtps
    else want=librmw_fastrtps_cpp; other=librmw_cyclonedds; fi
    total=$(wc -l < "$dir/rmw.txt")
    good=$(grep -c "$want" "$dir/rmw.txt")
    bad=$(grep -c "$other" "$dir/rmw.txt")
    if (( total > 0 && good == total && bad == 0 )); then
        record "scenes $label рантайм" ok "процессов с rmw $total, у всех $want"
    else
        record "scenes $label рантайм" отказ "процессов с rmw $total: $want у $good, $other у $bad"
    fi

    record "scenes $label RTF" число "$(real_time_factor "$env_cmd")"

    clean "$env_cmd && timeout 120 '$CLI' run examples/scenarios/$scene.yaml -R examples/robots" > "$dir/run.txt" 2>&1
    code=$?
    cat "$dir/run.txt"
    if (( code == 0 )); then
        record "scenes $label навык" ok "$(grep -m1 -E '^выполнено' "$dir/run.txt")"
    else
        record "scenes $label навык" "отказ, код $code" "$(grep -m1 -E '^(отказ|причина)|ошибка|не установлена|не хватает' "$dir/run.txt" || tail -1 "$dir/run.txt")"
    fi

    clean "$env_cmd && timeout 60 '$EXEC_PY' '$PY_DIR/joints.py' $APP examples/scenarios/$scene.yaml examples/robots '$dir/after.json' 2 30" > "$dir/after.txt" 2>&1
    cat "$dir/after.txt"
    if clean "'$EXEC_PY' '$PY_DIR/moved.py' '$dir/before.json' '$dir/after.json'" > "$dir/moved.txt" 2>&1; then
        record "scenes $label движение" ok "$(tail -1 "$dir/moved.txt")"
    else
        record "scenes $label движение" отказ "$(tail -1 "$dir/moved.txt")"
    fi

    stop_group "$pgid" 30
    leftovers "$label"
}

step_scenes() {
    if [[ ! -x "$CLI" || ! -f "$ROS_SETUP" ]]; then
        record scenes пропущено "нет окружения исполнения или ROS 2 (шаг env)"; return
    fi
    ensure_apt scenes "ros-$ROS_DISTRO_NAME-rmw-cyclonedds-cpp" || return
    if pgrep -f 'gz sim' >/dev/null; then
        record scenes пропущено "уже запущен gz sim — остановите его: прогон мерил бы чужой мир"; return
    fi
    if [[ ! -f "$G1_URDF" ]]; then
        info "описание G1 не загружено — загружается скриптом рядом с профилем (ADR-0017)"
        clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" ||
            record "scenes humanoid" отказ "описание G1 не загрузилось — причина выше"
    fi
    local scene rmw
    for scene in $SCENES; do
        for rmw in $RUNTIMES; do
            scene_once "$scene" "$rmw"
        done
    done
    clean "$ROS_ENV && ros2 daemon stop" >/dev/null 2>&1
}

# Each direction is reported by the side that receives it; the sender's count of
# matched readers is the second witness, taken from another process.
record_direction() {  # <label> <direction> <receiver exit> <receiver output> <sender output>
    local status
    if [[ $1 == контроль ]]; then
        (( $3 == 0 )) && status="прошло без пиров" || status="ожидаемый отказ"
    else
        (( $3 == 0 )) && status=ok || status=отказ
    fi
    record "oneside $1: $2" "$status" "получатель: $(tail -1 "$4" | sed 's/^[^;]*; //'); отправитель: $(tail -1 "$5" | sed 's/;.*//')"
}

pair() {  # pair <label> "<role> <config>" "<role> <config>"; the first starts 3 s earlier
    local label="$1" first second pid code_first code_second
    read -r -a first <<< "$2"
    read -r -a second <<< "$3"
    local out="$STAGE_ROOT/oneside-${label// /_}"
    step "oneside — $label: первым ${first[*]}, через 3 с ${second[*]}"
    start_group "$out-first.txt" 0 "exec timeout 40 '$VENV_OWN/bin/python' '$PY_DIR/dds_peer.py' ${first[0]} ${first[1]} $DDS_DOMAIN 16"
    pid=$LAST_GROUP
    sleep 3
    clean "timeout 40 '$VENV_OWN/bin/python' '$PY_DIR/dds_peer.py' ${second[0]} ${second[1]} $DDS_DOMAIN 10" > "$out-second.txt" 2>&1
    code_second=$?
    wait "$pid"
    code_first=$?
    cat "$out-first.txt" "$out-second.txt"
    if [[ ${first[0]} == robot ]]; then
        record_direction "$label" "LowState → адаптер" "$code_first" "$out-first.txt" "$out-second.txt"
        record_direction "$label" "LowCmd → сторона SDK" "$code_second" "$out-second.txt" "$out-first.txt"
    else
        record_direction "$label" "LowState → адаптер" "$code_second" "$out-second.txt" "$out-first.txt"
        record_direction "$label" "LowCmd → сторона SDK" "$code_first" "$out-first.txt" "$out-second.txt"
    fi
}

step_oneside() {
    if ! have_sdk_venv "$VENV_OWN"; then
        record oneside пропущено "нет $VENV_OWN: bash scripts/phase4-probe.sh 0.5"; return
    fi
    if pgrep -x unitree_mujoco >/dev/null; then
        record oneside пропущено "запущен unitree_mujoco в том же домене — пары мерили бы его"; return
    fi
    step "oneside — флаг мультикаста на lo"
    ip -d link show lo
    if ip link show lo | grep -q MULTICAST; then
        info "у lo есть MULTICAST: контроль без пиров может пройти, и тогда пары ничего не говорят о пирах"
    else
        info "у lo нет MULTICAST, как в прогоне 2 сессии 6"
    fi
    pair контроль "robot sdk-cxx-lo" "twin sdk-cxx-lo"
    pair "адаптер первым" "robot peers-lo" "twin sdk-cxx-lo"
    pair "SDK первым" "twin sdk-cxx-lo" "robot peers-lo"
}

build_twin() {
    step "twin — unitree_sdk2 @ ${SDK2_COMMIT:0:7} в $SDK2_DIR/install"
    if [[ -f "$SDK2_DIR/install/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake" ]]; then
        info "уже установлен"
    else
        fetch_pinned "$SDK2_URL" "$SDK2_COMMIT" "$SDK2_DIR/src" ||
            { record "twin сборка" отказ "unitree_sdk2 не загружен"; return 1; }
        # Into a prefix of its own, not the README's /opt/unitree_robotics: no sudo,
        # and nothing a later build could find by accident.
        clean "cmake -S '$SDK2_DIR/src' -B '$SDK2_DIR/build' -DCMAKE_INSTALL_PREFIX='$SDK2_DIR/install' -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF && cmake --build '$SDK2_DIR/build' --target install -j\$(nproc)" ||
            { record "twin сборка" отказ "unitree_sdk2 не собрался — причина выше"; return 1; }
    fi

    step "twin — MuJoCo $MUJOCO_VERSION"
    if [[ ! -f "$MUJOCO_DIR/lib/libmujoco.so" ]]; then
        clean "curl -fsSL --retry 3 -o '$STAGE_ROOT/mujoco.tar.gz' '$MUJOCO_URL' && tar -xzf '$STAGE_ROOT/mujoco.tar.gz' -C '$STAGE_ROOT'" ||
            { record "twin сборка" отказ "MuJoCo $MUJOCO_VERSION не загружен"; return 1; }
    fi
    info "sha256 архива: $(sha256sum "$STAGE_ROOT/mujoco.tar.gz" 2>/dev/null | cut -d' ' -f1)"
    if [[ ! -d "$MUJOCO_DIR/simulate" ]]; then
        record "twin сборка" отказ "в архиве MuJoCo нет каталога simulate, а сборка двойника берёт исходники оттуда"
        return 1
    fi

    step "twin — unitree_mujoco @ ${TWIN_COMMIT:0:7}"
    if [[ ! -x "$TWIN_BIN" ]]; then
        fetch_pinned "$TWIN_URL" "$TWIN_COMMIT" "$TWIN_DIR" ||
            { record "twin сборка" отказ "unitree_mujoco не загружен"; return 1; }
        # The symlink is the vendor README's own step, not an edit of the twin.
        ln -sfn "$MUJOCO_DIR" "$TWIN_DIR/simulate/mujoco"
        clean "cmake -S '$TWIN_DIR/simulate' -B '$TWIN_DIR/simulate/build' -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH='$SDK2_DIR/install' && cmake --build '$TWIN_DIR/simulate/build' --target unitree_mujoco -j\$(nproc)" ||
            { record "twin сборка" отказ "unitree_mujoco не собрался — причина выше"; return 1; }
    fi
    info "config.yaml двойника (файл производителя, не правится):"
    sed 's/^/      /' "$TWIN_DIR/simulate/config.yaml"
    record "twin сборка" ok "$TWIN_BIN"
}

reqs_twin() {
    local out="$STAGE_ROOT/reqs-twin" bin_dir code
    bin_dir="$(dirname "$TWIN_BIN")"
    step "reqs — объявление в форме двойника MuJoCo: программа на PATH и DISPLAY"
    clean_gui "export PATH='$bin_dir':\$PATH && '$EXEC_PY' '$PY_DIR/reqs_vendor.py' $APP twin" > "$out-present.txt" 2>&1
    code=$?
    cat "$out-present.txt"
    if (( code == 0 )) && clean_gui "export PATH='$bin_dir':\$PATH && command -v unitree_mujoco && test -n \"\$DISPLAY\"" >/dev/null 2>&1; then
        record "reqs двойник, всё есть" ok "$(tail -1 "$out-present.txt"); command -v и DISPLAY подтверждают"
    else
        record "reqs двойник, всё есть" отказ "$(tail -1 "$out-present.txt")"
    fi

    clean "'$EXEC_PY' '$PY_DIR/reqs_vendor.py' $APP twin" > "$out-absent.txt" 2>&1
    code=$?
    cat "$out-absent.txt"
    if (( code == 1 )) && [[ $(grep -cF '[env.missing]' "$out-absent.txt") -eq 2 ]] && ! clean "command -v unitree_mujoco" >/dev/null 2>&1; then
        record "reqs двойник, ничего нет" ok "называет программу и DISPLAY; command -v не находит программу"
    else
        record "reqs двойник, ничего нет" отказ "$(tail -1 "$out-absent.txt")"
    fi
}

run_twin() {
    local scene="$TWIN_DIR/unitree_robots/g1/scene_23dof.xml" out="$STAGE_ROOT/twin" pid code
    step "twin — unitree_mujoco -r g1 -s scene_23dof.xml -i $DDS_DOMAIN -n lo"
    if pgrep -x unitree_mujoco >/dev/null; then
        record "twin запуск" пропущено "unitree_mujoco уже запущен"; return
    fi
    start_group "$out-sim.log" 1 "cd '$(dirname "$TWIN_BIN")' && export LD_LIBRARY_PATH='$SDK2_DIR/install/lib:$MUJOCO_DIR/lib' && exec '$TWIN_BIN' -r g1 -s '$scene' -i $DDS_DOMAIN -n lo"
    pid=$LAST_GROUP
    if ! wait_for_line "$out-sim.log" "Mujoco data is prepared" 60 "$pid"; then
        record "twin запуск" отказ "за 60 с нет «Mujoco data is prepared»: $(tail -1 "$out-sim.log")"
        tail -40 "$out-sim.log" | sed 's/^/      /'
        stop_group "$pid" 10
        return
    fi
    sleep 2   # the bridge thread initialises its channel right after that line (main.cc)
    info "libddsc в процессе двойника: $(grep -oE '/[^ ]*libddsc[^ ]*' "/proc/$pid/maps" 2>/dev/null | sort -u | tr '\n' ' ')"
    record "twin запуск" ok "процесс $pid, $(basename "$scene")"

    clean "timeout 60 '$VENV_OWN/bin/python' '$PY_DIR/twin_link.py' peers-lo $DDS_DOMAIN" > "$out-peers.txt" 2>&1
    code=$?
    cat "$out-peers.txt"
    case $code in
        0)  record "twin пиры: LowState → адаптер" ok "$(grep -m1 'LowState [0-9]' "$out-peers.txt")"
            record "twin пиры: LowCmd → двойник" ok "$(tail -1 "$out-peers.txt")" ;;
        2)  record "twin пиры: LowState → адаптер" ok "$(grep -m1 'LowState [0-9]' "$out-peers.txt")"
            record "twin пиры: LowCmd → двойник" отказ "$(tail -1 "$out-peers.txt")" ;;
        *)  record "twin пиры: LowState → адаптер" "отказ, код $code" "$(tail -1 "$out-peers.txt")"
            record "twin пиры: LowCmd → двойник" пропущено "нет состояния — движению нечем подтвердиться" ;;
    esac

    clean "timeout 30 '$VENV_OWN/bin/python' '$PY_DIR/twin_link.py' sdk-cxx-lo $DDS_DOMAIN" > "$out-control.txt" 2>&1
    code=$?
    cat "$out-control.txt"
    if (( code == 1 )); then
        record "twin контроль: без пиров" "ожидаемый отказ" "$(tail -1 "$out-control.txt")"
    else
        record "twin контроль: без пиров" "прошло без пиров" "$(grep -m1 'LowState [0-9]' "$out-control.txt")"
    fi

    stop_group "$pid" 10
}

step_twin() {
    if [[ -z ${DISPLAY:-} ]]; then
        record twin пропущено "DISPLAY не задан: окно GLFW у двойника обязательно (ADR-0023); запускайте из графической сессии"
        return
    fi
    if ! have_sdk_venv "$VENV_OWN"; then
        record twin пропущено "нет $VENV_OWN: bash scripts/phase4-probe.sh 0.5"; return
    fi
    # libboost-program-options-dev, not the README's libboost-all-dev: the simulator
    # links only program_options, and the SDK's headers include no boost, Eigen or
    # spdlog (read at the pinned commits).
    ensure_apt twin build-essential cmake git curl libyaml-cpp-dev \
        libboost-program-options-dev libglfw3-dev libfmt-dev || return
    build_twin || return
    [[ -x "$EXEC_PY" ]] && reqs_twin
    run_twin
}

# ---------------------------------------------------------------------------

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
            env)     step_env ;;
            reqs)    step_reqs ;;
            scenes)  step_scenes ;;
            oneside) step_oneside ;;
            twin)    step_twin ;;
        esac
    done
    summary
}

main "$@"
