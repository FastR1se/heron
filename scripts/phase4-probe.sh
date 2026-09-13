#!/usr/bin/env bash
#
# Phase 4, stage 0: the facts only a Linux machine can establish
# (docs/plans/phase4-dds-bridge.md, facts 0.5-0.8; journal 2026-09-13, session 5).
#
# A probe, not an installer. It writes nothing outside $PROBE_ROOT except the apt
# packages a build step cannot run without -- each named, with its reason, by the
# step that needs it -- and the vendor SDK's own trace file, /tmp/cdds.LOG. It never stops at the first failure: an expected refusal is a
# result, and the point of the run is the table at the end. Everything printed
# goes to one log file -- that file is what to send back.
#
# Every step runs in a clean environment (env -i). Whatever the calling shell has
# sourced cannot decide an outcome; which steps see ROS 2 is written in the step.
#
# Steps:
#   env       versions and prerequisites
#   0.8       vendor messages unitree_hg build under Jazzy
#   0.5       cyclonedds binding on Python 3.12: against Jazzy's library (expected
#             to be refused), against a separately built 0.10.x, and 11.x wheels
#   coexist   rclpy on rmw_cyclonedds_cpp and the vendor SDK in one process
#   wire      SDK <-> ROS 2 over the wire, both directions, on lo and on the
#             default interface
#
# Usage:
#   bash scripts/phase4-probe.sh               # all steps, in order
#   bash scripts/phase4-probe.sh wire          # selected steps; earlier builds are reused
#   PROBE_ROOT=/data/probe bash scripts/phase4-probe.sh
#
# Roughly 1 GiB in $PROBE_ROOT and 10-20 minutes, most of it building CycloneDDS.

set -uo pipefail

# ---------------------------------------------------------------------------
# Constants. The vendor commits are the ones read in session 5, so the run
# checks the same files the journal describes.
# ---------------------------------------------------------------------------

readonly ROS_DISTRO_NAME="jazzy"
readonly ROS_SETUP="/opt/ros/$ROS_DISTRO_NAME/setup.bash"
readonly PY="python3.12"

readonly UNITREE_ROS2_URL="https://github.com/unitreerobotics/unitree_ros2.git"
readonly UNITREE_ROS2_COMMIT="668d1ec5a05d1c38d3306bdca7d59f2ba3581a88"
readonly SDK_URL="https://github.com/unitreerobotics/unitree_sdk2_python.git"
readonly SDK_COMMIT="65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"
readonly CDDS_URL="https://github.com/eclipse-cyclonedds/cyclonedds.git"
readonly CDDS_REFS="0.10.2 releases/0.10.x"   # the robot's version first, the vendor README's branch second
readonly CDDS_PY_PIN="0.10.2"                  # what unitree_sdk2py pins
readonly CDDS_PY_NEW="11.0.1"                  # the first release with wheels for Python 3.12

readonly DOMAIN_ID=1                           # the twin's default; the robot uses 0
readonly ALL_STEPS="env 0.8 0.5 coexist wire"

PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
readonly PROBE_ROOT
readonly PY_DIR="$PROBE_ROOT/py"
readonly WS="$PROBE_ROOT/ws"
readonly SDK_DIR="$PROBE_ROOT/unitree_sdk2_python"
readonly VENV_JAZZY="$PROBE_ROOT/venv-jazzy-lib"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly VENV_NEW="$PROBE_ROOT/venv-cyclonedds11"

mkdir -p "$PROBE_ROOT" "$PY_DIR"
LOG="$PROBE_ROOT/probe-$(date +%Y%m%d-%H%M%S).log"
readonly LOG
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------
# Output and bookkeeping. Messages address a human, so they are in Russian.
# ---------------------------------------------------------------------------

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

RESULTS=()
record() {  # record <fact> <status> <note>
    RESULTS+=("$1|$2|$3")
    printf '    [%s] %s — %s\n' "$2" "$1" "$3"
}

BACKGROUND=()
cleanup() {
    local pid
    for pid in "${BACKGROUND[@]}"; do kill "$pid" 2>/dev/null; done
}
trap cleanup EXIT

# A command in an environment that inherits nothing but HOME and a plain PATH.
clean() {
    env -i HOME="$HOME" USER="${USER:-probe}" LANG=C.UTF-8 TERM="${TERM:-dumb}" \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        bash --noprofile --norc -c "$1"
}

readonly ROS_ENV="source '$ROS_SETUP'"
readonly WS_ENV="$ROS_ENV && source '$WS/install/setup.bash' && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=$DOMAIN_ID"
readonly LO_URI="export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"lo\"/></Interfaces></General></Domain></CycloneDDS>'"

# Fetch one commit or ref into a directory, reusing a checkout that is already there.
fetch_pinned() {  # fetch_pinned <url> <commit-or-ref> <dir>
    local url="$1" ref="$2" dir="$3"
    if [[ -d "$dir/.git" ]]; then
        info "уже загружено: $dir @ $(git -C "$dir" rev-parse --short HEAD)"
        if [[ "$ref" =~ ^[0-9a-f]{40}$ && "$(git -C "$dir" rev-parse HEAD)" != "$ref" ]]; then
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

# Install what a build step cannot run without, saying what and why. Never silent.
ensure_apt() {  # ensure_apt <fact> <package>...
    local fact="$1" missing=() p
    shift
    for p in "$@"; do
        dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    (( ${#missing[@]} )) || return 0
    info "ставятся пакеты, без которых шаг $fact не собирается: ${missing[*]}"
    sudo apt-get install -y "${missing[@]}" ||
        { record "$fact" отказ "apt не поставил ${missing[*]}"; return 1; }
}

have_sdk_venv() { [[ -x "$1/bin/python" ]] && "$1/bin/python" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Python probes, written out once so that no step has to quote Python in bash.
# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/sdk_import.py" <<'PY'
import sys
import cyclonedds
print("python", sys.version.split()[0])
print("cyclonedds", cyclonedds.__version__)
try:
    from cyclonedds.__library__ import library_path
    print("libddsc", library_path)
except Exception as exc:  # the file exists only in source builds
    print("libddsc: путь не записан обвязкой:", exc)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
cmd = unitree_hg_msg_dds__LowCmd_()
print("motor_cmd slots", len(cmd.motor_cmd))
print("crc of an empty LowCmd", CRC().Crc(cmd))
PY

    cat > "$PY_DIR/coexist.py" <<'PY'
import sys
import rclpy
from rclpy.utilities import get_rmw_implementation_identifier
rclpy.init()
node = rclpy.create_node("probe_coexist")
print("rclpy ok, rmw:", get_rmw_implementation_identifier())
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(int(sys.argv[1]), "lo")
publisher = ChannelPublisher("rt/probe_coexist", LowState_)
publisher.Init()
print("sdk channel ok")
libs = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libddsc" in line})
print("libddsc loaded in this process:", len(libs))
for lib in libs:
    print("  ", lib)
node.destroy_node()
rclpy.shutdown()
PY

    cat > "$PY_DIR/wire_sdk_pub.py" <<'PY'
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
domain, iface, duration = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
ChannelFactoryInitialize(domain) if iface == "auto" else ChannelFactoryInitialize(domain, iface)
publisher = ChannelPublisher("rt/lowstate", LowState_)
publisher.Init()
msg = unitree_hg_msg_dds__LowState_()
msg.mode_machine = 4
msg.motor_state[15].q = 0.5
end, tick = time.monotonic() + duration, 0
while time.monotonic() < end:
    msg.tick = tick
    publisher.Write(msg)
    tick += 1
    time.sleep(0.002)
print("sdk: опубликовано LowState:", tick)
PY

    cat > "$PY_DIR/wire_sdk_sub.py" <<'PY'
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
domain, iface, duration = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
ChannelFactoryInitialize(domain) if iface == "auto" else ChannelFactoryInitialize(domain, iface)
seen = {"n": 0, "mode_machine": None, "kp15": None}
def on_cmd(msg):
    seen["n"] += 1
    seen["mode_machine"] = msg.mode_machine
    seen["kp15"] = msg.motor_cmd[15].kp
subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
subscriber.Init(on_cmd, 10)
time.sleep(duration)
print(f"sdk: принято LowCmd {seen['n']}, mode_machine {seen['mode_machine']}, kp[15] {seen['kp15']}")
sys.exit(0 if seen["n"] else 1)
PY

    cat > "$PY_DIR/wire_ros_read.py" <<'PY'
import sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowState
window, deadline = float(sys.argv[1]), float(sys.argv[2])
rclpy.init()
node = rclpy.create_node("probe_wire_read")
got = []
node.create_subscription(LowState, "lowstate", lambda m: got.append((time.monotonic(), m)),
                         qos_profile_sensor_data)
start = time.monotonic()
while not got and time.monotonic() - start < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
types = dict(node.get_topic_names_and_types()).get("/lowstate")
print("ros2: тип /lowstate по обнаружению:", types)
if not got:
    print(f"ros2: LowState не пришёл за {deadline:g} с; издателей видно: {node.count_publishers('lowstate')}")
    sys.exit(1)
first_t, first = got[0]
print(f"ros2: первое LowState через {first_t - start:.2f} с: tick {first.tick}, "
      f"mode_machine {first.mode_machine}, motor_state[15].q {first.motor_state[15].q}")
while time.monotonic() - first_t < window:
    rclpy.spin_once(node, timeout_sec=0.05)
last_t, last = got[-1]
elapsed = last_t - first_t
received = len(got) - 1
ticks = last.tick - first.tick
print(f"ros2: за {elapsed:.2f} с принято {received}, частота приёма {received / elapsed:.1f} Гц, "
      f"частота издателя по tick {ticks / elapsed:.1f} Гц, потеряно {max(ticks - received, 0)}")
node.destroy_node()
rclpy.shutdown()
PY

    cat > "$PY_DIR/wire_ros_write.py" <<'PY'
import sys, time
import rclpy
from unitree_hg.msg import LowCmd
duration = float(sys.argv[1])
rclpy.init()
node = rclpy.create_node("probe_wire_write")
publisher = node.create_publisher(LowCmd, "lowcmd", 10)
msg = LowCmd()
msg.mode_machine = 4
msg.motor_cmd[15].kp = 40.0
end, n = time.monotonic() + duration, 0
while time.monotonic() < end:
    publisher.publish(msg)
    n += 1
    time.sleep(0.02)
print(f"ros2: опубликовано LowCmd {n}, подписчиков видно: {node.count_subscribers('lowcmd')}")
node.destroy_node()
rclpy.shutdown()
PY
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

step_env() {
    step "Окружение"
    info "лог: $LOG"
    info "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}"), ядро $(uname -r), $(uname -m)"
    info "$($PY --version 2>&1)"
    info "$(gcc --version 2>/dev/null | head -1 || echo 'gcc: нет')"
    info "$(cmake --version 2>/dev/null | head -1 || echo 'cmake: нет')"
    info "$(git --version 2>/dev/null || echo 'git: нет')"
    local p
    for p in ros-$ROS_DISTRO_NAME-cyclonedds ros-$ROS_DISTRO_NAME-rmw-cyclonedds-cpp \
             ros-$ROS_DISTRO_NAME-rosidl-generator-dds-idl python3-venv libssl-dev; do
        info "$p: $(dpkg-query -W -f='${Version}' "$p" 2>/dev/null || echo 'не установлен')"
    done
    [[ -n "${ROS_DISTRO:-}" ]] && info "в этом терминале подключён ROS 2 ($ROS_DISTRO): на исход не влияет, шаги идут в чистом окружении"

    local ok=1
    [[ -f "$ROS_SETUP" ]] || { record env отказ "нет $ROS_SETUP"; ok=0; }
    command -v "$PY" >/dev/null || { record env отказ "нет $PY"; ok=0; }
    command -v cmake >/dev/null || { record env отказ "нет cmake (phase0.sh install base)"; ok=0; }
    command -v gcc >/dev/null || { record env отказ "нет gcc (phase0.sh install base)"; ok=0; }
    clean "$ROS_ENV && command -v colcon" >/dev/null 2>&1 || { record env отказ "нет colcon (ros-dev-tools)"; ok=0; }
    clean "$PY -m venv '$PROBE_ROOT/.venv-check' && rm -rf '$PROBE_ROOT/.venv-check'" >/dev/null 2>&1 ||
        { record env отказ "$PY -m venv не работает: sudo apt install python3.12-venv"; ok=0; }
    (( ok )) && record env ok "предусловия на месте"
}

step_08() {
    step "0.8 — unitree_hg собирается под Jazzy"
    ensure_apt 0.8 ros-$ROS_DISTRO_NAME-rmw-cyclonedds-cpp ros-$ROS_DISTRO_NAME-rosidl-generator-dds-idl ||
        return

    fetch_pinned "$UNITREE_ROS2_URL" "$UNITREE_ROS2_COMMIT" "$WS/src/unitree_ros2" ||
        { record 0.8 отказ "не загружен unitree_ros2"; return; }

    if ! clean "$ROS_ENV && cd '$WS' && colcon build --base-paths src/unitree_ros2/cyclonedds_ws/src/unitree --packages-select unitree_hg --event-handlers console_direct+"; then
        record 0.8 отказ "colcon build unitree_hg — причина в выводе выше"
        return
    fi
    if clean "$WS_ENV && ros2 interface show unitree_hg/msg/LowCmd && python3 -c 'from unitree_hg.msg import LowCmd, LowState; print(\"слотов motor_cmd:\", len(LowCmd().motor_cmd), \"motor_state:\", len(LowState().motor_state))'"; then
        record 0.8 ok "unitree_hg собран и импортируется"
    else
        record 0.8 отказ "собран, но не импортируется"
    fi
}

step_05() {
    # bison builds the IDL compiler of the C library; python3-dev gives Python.h to
    # the binding, which on 3.12 compiles from source (no wheels, session 5).
    ensure_apt 0.5 bison python3-dev || return

    step "0.5а — обвязка cyclonedds $CDDS_PY_PIN против CycloneDDS из Jazzy"
    if clean "$PY -m venv '$VENV_JAZZY' && CYCLONEDDS_HOME=/opt/ros/$ROS_DISTRO_NAME '$VENV_JAZZY/bin/pip' install --no-cache-dir 'cyclonedds==$CDDS_PY_PIN'"; then
        record 0.5а "ok, неожиданно" "обвязка собралась против библиотеки Jazzy — вывод этапа 0 пересмотреть"
    else
        record 0.5а "ожидаемый отказ" "поисковик сборки не принимает раскладку /opt/ros/$ROS_DISTRO_NAME"
    fi

    fetch_pinned "$SDK_URL" "$SDK_COMMIT" "$SDK_DIR" || { record 0.5 отказ "не загружен unitree_sdk2_python"; return; }

    step "0.5б — своя C-библиотека CycloneDDS, обвязка $CDDS_PY_PIN и SDK"
    local ref dir built=""
    for ref in $CDDS_REFS; do
        dir="$PROBE_ROOT/cyclonedds-${ref//\//_}"
        if [[ -f "$dir/install/lib/libddsc.so" || -f "$dir/install/lib64/libddsc.so" ]]; then
            record "0.5б C $ref" ok "уже собрана: $dir/install"; built="$dir/install"; break
        fi
        if ! fetch_pinned "$CDDS_URL" "$ref" "$dir/src"; then
            record "0.5б C $ref" отказ "не загружена"; continue
        fi
        if clean "cmake -S '$dir/src' -B '$dir/build' -DCMAKE_INSTALL_PREFIX='$dir/install' -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SHM=OFF && cmake --build '$dir/build' --target install -j\$(nproc)"; then
            record "0.5б C $ref" ok "коммит $(git -C "$dir/src" rev-parse --short HEAD), $dir/install"
            built="$dir/install"
            break
        fi
        record "0.5б C $ref" отказ "сборка C-библиотеки — причина в выводе выше"
    done
    if [[ -z "$built" ]]; then
        record 0.5б отказ "ни одна C-библиотека 0.10.x не собралась"
    elif clean "$PY -m venv '$VENV_OWN' && CYCLONEDDS_HOME='$built' '$VENV_OWN/bin/pip' install --no-cache-dir 'cyclonedds==$CDDS_PY_PIN' numpy && '$VENV_OWN/bin/pip' install --no-deps -e '$SDK_DIR'" &&
         clean "'$VENV_OWN/bin/python' '$PY_DIR/sdk_import.py'"; then
        record 0.5б ok "обвязка $CDDS_PY_PIN на $PY и SDK импортируются, CRC считается"
    else
        record 0.5б отказ "обвязка или SDK — причина в выводе выше"
    fi

    step "0.5в — cyclonedds $CDDS_PY_NEW из колёс (вопреки пину SDK)"
    if clean "$PY -m venv '$VENV_NEW' && '$VENV_NEW/bin/pip' install --no-cache-dir 'cyclonedds==$CDDS_PY_NEW' numpy && '$VENV_NEW/bin/pip' install --no-deps -e '$SDK_DIR'" &&
       clean "'$VENV_NEW/bin/python' '$PY_DIR/sdk_import.py'"; then
        record 0.5в ok "SDK импортируется на cyclonedds $CDDS_PY_NEW"
    else
        record 0.5в отказ "SDK на cyclonedds $CDDS_PY_NEW — причина в выводе выше"
    fi
}

step_coexist() {
    step "0.5г — rclpy на rmw_cyclonedds_cpp и SDK в одном процессе"
    local label venv
    for label in own-lib cyclonedds11; do
        [[ $label == own-lib ]] && venv="$VENV_OWN" || venv="$VENV_NEW"
        if ! have_sdk_venv "$venv"; then
            record "0.5г $label" пропущено "нет окружения $venv (шаг 0.5)"; continue
        fi
        if clean "$ROS_ENV && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=$DOMAIN_ID && '$venv/bin/python' '$PY_DIR/coexist.py' $DOMAIN_ID"; then
            record "0.5г $label" ok "оба в одном процессе; сколько libddsc загружено — в выводе выше"
        else
            record "0.5г $label" отказ "rclpy и SDK не уживаются — причина в выводе выше"
        fi
    done
}

wire_once() {  # wire_once <label> <venv> <iface: lo|auto>
    local label="$1" venv="$2" iface="$3" ros_env="$WS_ENV" out="$PROBE_ROOT/wire-$1-$3"
    [[ $iface == lo ]] && ros_env="$ros_env && $LO_URI"

    step "0.6/0.7 — SDK ($label) → ROS 2, интерфейс $iface"
    clean "'$venv/bin/python' '$PY_DIR/wire_sdk_pub.py' $DOMAIN_ID $iface 30" > "$out-pub.txt" 2>&1 &
    local pub=$!
    BACKGROUND+=("$pub")
    sleep 3
    if clean "$ros_env && python3 '$PY_DIR/wire_ros_read.py' 5 20" | tee "$out-read.txt"; then
        record "wire $label $iface SDK→ROS" ok "$(tail -1 "$out-read.txt")"
        clean "$ros_env && timeout 15 ros2 topic info --no-daemon --spin-time 3 -v /lowstate" ||
            info "ros2 topic info не отработал — не влияет на результат"
    else
        record "wire $label $iface SDK→ROS" отказ "$(tail -1 "$out-read.txt")"
    fi
    wait "$pub"; cat "$out-pub.txt"

    step "0.6/0.7 — ROS 2 → SDK ($label), интерфейс $iface"
    clean "'$venv/bin/python' '$PY_DIR/wire_sdk_sub.py' $DOMAIN_ID $iface 16" > "$out-sub.txt" 2>&1 &
    local sub=$!
    BACKGROUND+=("$sub")
    sleep 3
    clean "$ros_env && python3 '$PY_DIR/wire_ros_write.py' 10"
    if wait "$sub"; then
        record "wire $label $iface ROS→SDK" ok "$(tail -1 "$out-sub.txt")"
    else
        record "wire $label $iface ROS→SDK" отказ "$(tail -1 "$out-sub.txt")"
    fi
    cat "$out-sub.txt"
}

step_wire() {
    if [[ ! -f "$WS/install/setup.bash" ]]; then
        record wire пропущено "unitree_hg не собран (шаг 0.8)"; return
    fi
    local label venv iface
    for label in own-lib cyclonedds11; do
        [[ $label == own-lib ]] && venv="$VENV_OWN" || venv="$VENV_NEW"
        if ! have_sdk_venv "$venv"; then
            record "wire $label" пропущено "нет окружения $venv (шаг 0.5)"; continue
        fi
        for iface in lo auto; do
            wire_once "$label" "$venv" "$iface"
        done
    done
}

# ---------------------------------------------------------------------------

summary() {
    step "Итог"
    local row fact status note
    printf '\n| Факт | Результат | Подробности |\n|---|---|---|\n'
    for row in "${RESULTS[@]}"; do
        IFS='|' read -r fact status note <<< "$row"
        printf '| %s | %s | %s |\n' "$fact" "$status" "$note"
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
            0.8)     step_08 ;;
            0.5)     step_05 ;;
            coexist) step_coexist ;;
            wire)    step_wire ;;
        esac
    done
    summary
}

main "$@"
