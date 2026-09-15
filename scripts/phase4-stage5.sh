#!/usr/bin/env bash
#
# Phase 4, stage 5: G1 in Gazebo through the bridge of its package, driven by the same DDS
# adapter as in MuJoCo (docs/plans/phase4-dds-bridge.md, "Этап 5"; ADR-0033, ADR-0034).
#
# Built like scripts/phase4-stage4.sh and on its lessons: every check stands on a
# confirmation taken by a different tool than the one under test. Motion is read by a DDS
# participant of its own, not by the adapter; the controllers by `ros2 control`, not by the
# launch log; the middleware off /proc maps, not off the variable that asked for it; the
# environment of the scene off the generated launch file, not off the profile. Nothing stops
# at the first failure. The table at the end is a summary; the evidence is above it.
#
#   env    versions, commit; the execution venv synced and the G1 package installed; the
#          adapter's venv reinstalled; descriptions fetched; unitree_hg and unitree_g1_bridge
#          built with colcon into $STAGE_ROOT/ws -- the first compilation of the controller
#   reqs   validate; env check with the workspace sourced and without ROS 2; sim run
#          --dry-run: the launch sets the link's environment, the manager is told the bridge
#   scene  sim run; LowState seen by a witness; controllers active; CycloneDDS in every
#          process of the world; `run` from the adapter's venv without ROS 2; motion judged
#          by the witness; SIGTERM to sim run, and the world must be gone after
#
# Usage:
#   bash scripts/phase4-stage5.sh                # all steps, in order
#   bash scripts/phase4-stage5.sh env scene      # selected steps
#   GUI=1 bash scripts/phase4-stage5.sh scene    # with the simulator's window and RViz
#
# Prerequisites: scripts/phase0.sh (ROS 2 Jazzy, Gazebo, the execution venv);
# scripts/phase4-probe.sh 0.8 (unitree_ros2 fetched into $PROBE_ROOT/ws/src) and 0.5
# (venv-own-lib: cyclonedds 0.10.2 and the SDK); ros-jazzy-rmw-cyclonedds-cpp.
#
# Writes outside $STAGE_ROOT: `uv sync` and `uv pip install` in the execution venv;
# `pip install --no-deps -e` of the platform and the G1 package into venv-own-lib; the
# vendor descriptions under examples/robots/g1_23dof/vendor (gitignored).
#
# Roughly 10-15 minutes, most of it the first colcon build.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="examples/scenarios/humanoid.yaml"
readonly PROFILE="examples/robots/g1_23dof/profile.yaml"
readonly REGISTRY="examples/robots"
readonly PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly BRIDGE_SRC="$PACKAGE/ros2"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
readonly ROBOT_ID="g1"                      # humanoid.yaml, robots[0].id

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase4-stage5}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
GUI="${GUI:-0}"
readonly STAGE_ROOT PROBE_ROOT GUI
readonly PY_DIR="$STAGE_ROOT/py"
readonly WS="$STAGE_ROOT/ws"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly OWN_PY="$VENV_OWN/bin/python"
readonly OWN_CLI="$VENV_OWN/bin/$APP"
readonly UNITREE_HG_SRC="$PROBE_ROOT/ws/src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg"
readonly ALL_STEPS="env reqs scene"

mkdir -p "$STAGE_ROOT" "$PY_DIR"
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

# Nothing inherited but HOME and a plain PATH (stage 1): what the calling shell has sourced
# cannot decide an outcome, and which commands see ROS 2 is written in the command.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage5}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

readonly ROS_ENV="source '$ROS_SETUP'"
readonly WS_ENV="$ROS_ENV && source '$WS/install/setup.bash'"

# Processes of the scene are recognised by a variable only the scene's commands carry, not
# by names (stage 1, run 1: "spawner" matched the desktop's own daemons).
readonly SCENE_MARK="PHASE4_STAGE5_SCENE=$$"

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

running() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ $state != Z* ]]; }

scene_pids() {
    local p
    for p in $(pgrep -u "$(id -u)"); do
        grep -qzx "$SCENE_MARK" "/proc/$p/environ" 2>/dev/null && echo "$p"
    done
}

entry_names() {  # entry_names <python> <group suffix>
    "$1" -c "from importlib.metadata import entry_points; print(' '.join(sorted(e.name for e in entry_points(group='$APP.$2'))) or 'пусто')" 2>/dev/null ||
        echo "не читается"
}

# A process and everything under it. The world lives in a session of its own, not in the
# group of sim run (ADR-0037), so it is found by descent.
descendants() {  # descendants <pid>
    local p
    echo "$1"
    for p in $(pgrep -P "$1"); do descendants "$p"; done
}

# Processes under sim run and the ROS 2 middleware each has loaded, read off its maps.
rmw_in_tree() {  # rmw_in_tree <pid>
    local p libs
    for p in $(descendants "$1"); do
        libs="$(grep -oE 'librmw_[a-z_]+\.so' "/proc/$p/maps" 2>/dev/null |
            grep -vE '^librmw_(implementation|dds_common)' | sort -u | tr '\n' ' ')"
        [[ -n $libs ]] && printf '%s[%s] %s\n' "$(cat "/proc/$p/comm" 2>/dev/null)" "$p" "$libs"
    done
}

# ---------------------------------------------------------------------------
# Python probes. The witness runs in the adapter's venv as a participant of its own, on the
# configuration the package builds from the profile's params -- the one the world runs on.
# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/witness.py" <<'PY'
# LowState from a participant of its own: what the robot did, not what the adapter says.
# Waits up to <timeout> s for the first state, then reads for <window> s.
import json, sys, time
import yaml
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_g1.channel import STATE_TOPIC, LinkConfig

profile, window, timeout, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
params = yaml.safe_load(open(profile, encoding="utf-8"))["adapters"]["sim.gazebo"]["params"]
link = LinkConfig.from_params(params)
domain = Domain(link.domain, link.cyclonedds_xml())
participant = DomainParticipant(link.domain)
reader = DataReader(participant, Topic(participant, STATE_TOPIC, LowState_))
start, got = time.monotonic(), []
while True:
    now = time.monotonic()
    for sample in reader.take(N=256):
        if hasattr(sample, "tick"):
            got.append((now, sample.tick, sample.mode_machine,
                        [motor.q for motor in sample.motor_state],
                        [motor.tau_est for motor in sample.motor_state]))
    if got and now - got[0][0] >= window:
        break
    if not got and now - start >= timeout:
        break
    time.sleep(0.005)
if len(got) < 2:
    print(f"свидетель: LowState принято {len(got)} за {time.monotonic() - start:.0f} с (домен {link.domain})")
    sys.exit(1)
(t0, k0, machine, _, _), (t1, k1, _, q, tau) = got[0], got[-1]
wall = max(t1 - t0, 1e-9)
steps = sorted(b[1] - a[1] for a, b in zip(got, got[1:]) if b[1] != a[1])
median = steps[len(steps) // 2] if steps else 0
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"tick": k1, "mode_machine": machine, "q": q, "tau_est": tau}, handle)
print(f"свидетель: LowState {len(got)} за {wall:.2f} с, первое через {got[0][0] - start:.1f} с; "
      f"tick +{k1 - k0} мс мира, фактор {(k1 - k0) / 1000 / wall:.2f}, медианный шаг tick {median} мс, "
      f"mode_machine {machine}")
PY

    cat > "$PY_DIR/moved.py" <<'PY'
# Whether the contract's joints went toward the scenario's target, by slot (ADR-0027).
import json, sys
import yaml
from unitree_g1.slots import slots_of

before, after = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3])
target = yaml.safe_load(open(sys.argv[3], encoding="utf-8"))["task"]["target"]
order = yaml.safe_load(open(sys.argv[4], encoding="utf-8"))["contract"]["joint_order"]
slots = slots_of(order)
gap = lambda state: [abs(state["q"][slot] - goal) for slot, goal in zip(slots, target)]
was, now = gap(before), gap(after)
arms = [i for i, name in enumerate(order) if any(w in name for w in ("shoulder", "elbow", "wrist"))]
legs = [i for i in range(len(order)) if i not in arms]
worst = max(range(len(now)), key=now.__getitem__)
torque = max(abs(after["tau_est"][slot]) for slot in slots)
print(f"до цели было {max(was):.3f} рад, стало {now[worst]:.3f} ({order[worst]}); "
      f"руки {max(was[i] for i in arms):.3f} -> {max(now[i] for i in arms):.3f}, "
      f"ноги и пояс {max(was[i] for i in legs):.3f} -> {max(now[i] for i in legs):.3f}; "
      f"|tau_est| до {torque:.1f} Н·м; tick {before['tick']} -> {after['tick']}")
sys.exit(0 if now[worst] <= 0.5 * max(was) else 1)
PY

    cat > "$PY_DIR/scene_env.py" <<'PY'
# The environment the generated launch sets, as shell exports: read off the file, so what
# the checks run in is what the world runs in (ADR-0034).
import shlex, sys
from xml.etree import ElementTree as ET
body = open(sys.argv[1], encoding="utf-8").read().split("?>", 1)[1]
for item in ET.fromstring(body).iter("set_env"):
    print(f"export {item.get('name')}={shlex.quote(item.get('value') or '')}")
PY

    cat > "$PY_DIR/link_xml.py" <<'PY'
# The CycloneDDS configuration the adapter builds for sim.gazebo, for comparison with the launch.
import sys
import yaml
from unitree_g1.channel import LinkConfig
params = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["adapters"]["sim.gazebo"]["params"]
print(LinkConfig.from_params(params).cyclonedds_xml())
PY
}

# ---------------------------------------------------------------------------

step_env() {
    step "Окружение"
    info "лог: $LOG"
    info "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}"), ядро $(uname -r)"
    info "репозиторий: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l)"
    git -C "$REPO_ROOT" status --short 2>/dev/null | sed 's/^/      /'
    info "DISPLAY: ${DISPLAY:-не задан}; GUI=$GUI; lo: $(ip -o link show lo 2>/dev/null | sed -E 's/.*<([^>]*)>.*/\1/')"
    local ok=1

    if [[ ! -f $ROS_SETUP ]]; then
        record env отказ "нет $ROS_SETUP (scripts/phase0.sh)"; ok=0
    else
        info "Gazebo: $(clean "$ROS_ENV && gz sim --versions 2>/dev/null | head -1" || echo '?'); rmw_cyclonedds_cpp: $(clean "$ROS_ENV && ros2 pkg prefix rmw_cyclonedds_cpp 2>/dev/null" || echo нет)"
        local pkg
        for pkg in ros-jazzy-ros2-control ros-jazzy-gz-ros2-control ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-cyclonedds; do
            info "$pkg: $(dpkg -s "$pkg" 2>/dev/null | sed -n 's/^Version: //p' | grep . || echo 'не установлен')"
        done
    fi

    if [[ ! -x "$EXEC_PY" ]]; then
        record env отказ "нет окружения исполнения $EXEC_VENV (scripts/phase0.sh install exec)"; ok=0
    else
        # uv sync removes what pyproject.toml does not list -- the robot package among it.
        info "окружение исполнения: uv sync, затем пакет G1"
        if clean "cd '$REPO_ROOT' && uv sync --extra dev --extra ros2 && uv pip install --python '$EXEC_PY' -e '$PACKAGE'"; then
            info "бэкенды: $(entry_names "$EXEC_PY" sim_backends); двойники: $(entry_names "$EXEC_PY" twins)"
            [[ $(entry_names "$EXEC_PY" twins) == *unitree_g1_gazebo* ]] ||
                { record env отказ "в окружении исполнения нет двойника unitree_g1_gazebo"; ok=0; }
        else
            record env отказ "uv sync или установка пакета G1 не прошли — причина выше"; ok=0
        fi
    fi

    if [[ ! -x "$OWN_PY" ]] || ! "$OWN_PY" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; then
        record env отказ "нет $VENV_OWN с SDK: bash scripts/phase4-probe.sh 0.5"; ok=0
    else
        info "окружение адаптера $VENV_OWN: платформа и пакет G1 без зависимостей, заново"
        if clean "'$OWN_PY' -m pip install -q --no-cache-dir pydantic pyyaml typer packaging && '$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$PACKAGE'"; then
            info "cyclonedds: $("$OWN_PY" -m pip show cyclonedds 2>/dev/null | sed -n 's/^Version: //p'); адаптеры: $(entry_names "$OWN_PY" adapters); двойники: $(entry_names "$OWN_PY" twins)"
        else
            record env отказ "установка в $VENV_OWN не прошла — причина выше"; ok=0
        fi
    fi

    info "описания производителя: fetch_description.py"
    clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" ||
        { record env отказ "описания не загрузились — причина выше"; ok=0; }

    step "env — сборка unitree_hg и unitree_g1_bridge (первая компиляция контроллера)"
    if [[ ! -f "$UNITREE_HG_SRC/package.xml" ]]; then
        record "env colcon" отказ "нет $UNITREE_HG_SRC: bash scripts/phase4-probe.sh 0.8"; ok=0
    else
        info "unitree_ros2: $(git -C "$UNITREE_HG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
        rm -rf "$WS"; mkdir -p "$WS"
        if clean "$ROS_ENV && cd '$WS' && colcon build --base-paths '$UNITREE_HG_SRC' '$BRIDGE_SRC' --packages-select unitree_hg unitree_g1_bridge --event-handlers console_direct+" > "$STAGE_ROOT/colcon.txt" 2>&1; then
            local warnings
            warnings=$(grep -c 'warning:' "$STAGE_ROOT/colcon.txt")
            grep -E 'warning:|error:' "$STAGE_ROOT/colcon.txt" | sort | uniq -c | sort -rn | head -15 | sed 's/^/      /'
            record "env colcon" ok "собрано; предупреждений компилятора $warnings (первые выше)"
        else
            tail -60 "$STAGE_ROOT/colcon.txt" | sed 's/^/      /'
            record "env colcon" отказ "colcon build — хвост вывода выше, целиком $STAGE_ROOT/colcon.txt"; ok=0
        fi
        local plugin index
        plugin="$WS/install/unitree_g1_bridge/share/unitree_g1_bridge/unitree_g1_bridge.xml"
        index="$WS/install/unitree_g1_bridge/share/ament_index/resource_index/controller_interface__pluginlib__plugin/unitree_g1_bridge"
        if [[ -f $plugin && -f $index ]] && clean "$WS_ENV && ros2 pkg prefix unitree_g1_bridge" >/dev/null 2>&1; then
            record "env плагин" ok "описание плагина и запись индекса pluginlib на месте; $(ls "$WS/install/unitree_g1_bridge/lib/" 2>/dev/null | tr '\n' ' ')"
        else
            record "env плагин" отказ "описание $([[ -f $plugin ]] && echo есть || echo нет), индекс $([[ -f $index ]] && echo есть || echo нет)"; ok=0
        fi
    fi
    (( ok )) && record env ok "предусловия на месте"
}

step_reqs() {
    local out="$STAGE_ROOT/reqs" code
    step "reqs — validate профиля и сцены"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' validate '$PROFILE' '$SCENARIO' -R '$REGISTRY'" > "$out-validate.txt" 2>&1
    code=$?
    cat "$out-validate.txt"
    if (( code == 0 )); then
        record "reqs validate" ok "0 ошибок; $(grep -c '^внимание' "$out-validate.txt") предупреждений"
    else
        record "reqs validate" "отказ, код $code" "$(grep -m1 '^ошибка' "$out-validate.txt" || tail -1 "$out-validate.txt")"
    fi

    step "reqs — env check: адаптер в своём окружении, рабочее пространство и ROS 2 подключены"
    clean "$WS_ENV && cd '$REPO_ROOT' && '$OWN_CLI' env check '$SCENARIO' -R '$REGISTRY'" > "$out-env.txt" 2>&1
    code=$?
    cat "$out-env.txt"
    if (( code == 0 )); then
        record "reqs env check, всё есть" ok "$(grep -c '^проверено' "$out-env.txt") стороны проверены"
    else
        record "reqs env check, всё есть" "отказ, код $code" "$(grep -m3 '^ошибка' "$out-env.txt" | tr '\n' ' ')"
    fi
    clean "cd '$REPO_ROOT' && '$OWN_CLI' env check '$SCENARIO' -R '$REGISTRY'" > "$out-env-bare.txt" 2>&1
    code=$?
    if (( code == 1 )) && grep -qF "unitree_g1_bridge" "$out-env-bare.txt" && grep -qF "rmw_cyclonedds_cpp" "$out-env-bare.txt"; then
        record "reqs env check, без ROS 2" ok "называет unitree_g1_bridge и rmw_cyclonedds_cpp"
    else
        record "reqs env check, без ROS 2" отказ "код $code; $(grep -m3 '^ошибка' "$out-env-bare.txt" | tr '\n' ' ')"
    fi

    step "reqs — sim run --dry-run: окружение связи в запуске, мост в менеджере"
    rm -rf "$STAGE_ROOT/dry"
    clean "$WS_ENV && cd '$REPO_ROOT' && '$EXEC_CLI' sim run '$SCENARIO' --into '$STAGE_ROOT/dry' -R '$REGISTRY' --dry-run" > "$out-dry.txt" 2>&1
    code=$?
    cat "$out-dry.txt"
    local launch="$STAGE_ROOT/dry/bringup.launch.xml" controllers="$STAGE_ROOT/dry/$ROBOT_ID/controllers.yaml"
    if (( code == 0 )) && [[ -f $launch && -f $controllers ]]; then
        "$EXEC_PY" "$PY_DIR/scene_env.py" "$launch" > "$out-scene.env"
        sed -E 's/(CYCLONEDDS_URI=).{60}.*/\1…/; s/^/      /' "$out-scene.env"
        local uri want
        uri="$(clean "source '$out-scene.env' && printf '%s' \"\$CYCLONEDDS_URI\"")"
        want="$(clean "cd '$REPO_ROOT' && '$OWN_PY' '$PY_DIR/link_xml.py' '$PROFILE'")"
        if grep -qx "export ROS_DOMAIN_ID=1" "$out-scene.env" && grep -qx "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" "$out-scene.env" &&
            [[ -n $uri && $uri == "$want" ]] && grep -qF "unitree_g1_bridge/LowLevelBridge" "$controllers"; then
            record "reqs sim run --dry-run" ok "домен 1, rmw_cyclonedds_cpp, CYCLONEDDS_URI = конфигурация адаптера; менеджер знает LowLevelBridge"
        else
            record "reqs sim run --dry-run" отказ "окружение или контроллеры не те: $(tr '\n' ' ' < "$out-scene.env" | cut -c1-200)"
        fi
    else
        record "reqs sim run --dry-run" "отказ, код $code" "$(tail -1 "$out-dry.txt")"
    fi
}

step_scene() {
    local dir="$STAGE_ROOT/scene" pgid code waited=0 flags=""
    rm -rf "$dir"; mkdir -p "$dir"
    step "scene — sim run, мост, run из окружения адаптера, SIGTERM"
    if [[ -n "$(scene_pids)" ]] || pgrep -x gz >/dev/null || pgrep -f 'gz sim' >/dev/null; then
        record scene пропущено "Gazebo уже запущен — остановите его"; return
    fi
    if (( GUI )); then
        [[ -n ${DISPLAY:-} ]] || { record scene пропущено "GUI=1, а DISPLAY не задан"; return; }
    else
        flags="--headless --no-rviz"
    fi

    start_group "$dir/sim.log" "$WS_ENV && export $SCENE_MARK && cd '$REPO_ROOT' && exec '$EXEC_CLI' sim run '$SCENARIO' --into '$dir/world' -R '$REGISTRY' $flags"
    pgid=$LAST_GROUP
    info "sim run: процесс и группа $pgid, лог $dir/sim.log"

    # The world is written before it starts; the checks run in the environment it sets.
    while [[ ! -f "$dir/world/bringup.launch.xml" ]] && running "$pgid" && (( waited < 60 )); do
        sleep 1; waited=$(( waited + 1 ))
    done
    if [[ ! -f "$dir/world/bringup.launch.xml" ]]; then
        record "scene sim run" отказ "мир не записан: $(grep -m1 -E 'ошибка|не ' "$dir/sim.log" || tail -1 "$dir/sim.log")"
        tail -40 "$dir/sim.log" | sed 's/^/      /'
        stop_group "$pgid" 30; return
    fi
    "$EXEC_PY" "$PY_DIR/scene_env.py" "$dir/world/bringup.launch.xml" > "$dir/scene.env"
    local scene_env="$WS_ENV && source '$dir/scene.env'"

    # The witness waits for the first state, but not for a world that is gone (stage 1).
    start_group "$dir/before.txt" "cd '$REPO_ROOT' && exec timeout 220 '$OWN_PY' '$PY_DIR/witness.py' '$PROFILE' 2 180 '$dir/before.json'"
    local reader=$LAST_GROUP
    while running "$reader"; do
        running "$pgid" || { info "sim run завершился раньше, чем пришло состояние"; stop_group "$reader" 5; break; }
        sleep 1
    done
    cat "$dir/before.txt"
    if [[ ! -s "$dir/before.json" ]]; then
        record "scene состояние" отказ "$(tail -1 "$dir/before.txt")"
        info "хвост лога sim run:"; tail -60 "$dir/sim.log" | sed 's/^/      /'
        clean "$scene_env && timeout 30 ros2 control list_controllers -c /$ROBOT_ID/controller_manager" 2>&1 | sed 's/^/      /'
        clean "$scene_env && timeout 20 ros2 topic info /lowstate -v" 2>&1 | head -30 | sed 's/^/      /'
        stop_group "$pgid" 30; return
    fi
    record "scene состояние" ok "$(tail -1 "$dir/before.txt")"

    clean "$scene_env && ros2 daemon stop >/dev/null 2>&1; timeout 40 ros2 control list_controllers -c /$ROBOT_ID/controller_manager" > "$dir/controllers.txt" 2>&1
    sed 's/^/      /' "$dir/controllers.txt"
    if grep -E '^low_level_bridge\b.*\bactive\b' "$dir/controllers.txt" >/dev/null && grep -E '^joint_state_broadcaster\b.*\bactive\b' "$dir/controllers.txt" >/dev/null; then
        record "scene контроллеры" ok "low_level_bridge и joint_state_broadcaster active"
    else
        record "scene контроллеры" отказ "$(tr '\n' ' ' < "$dir/controllers.txt" | cut -c1-200)"
    fi

    rmw_in_tree "$pgid" > "$dir/rmw.txt"
    sed 's/^/      /' "$dir/rmw.txt"
    local total good
    total=$(wc -l < "$dir/rmw.txt"); good=$(grep -c librmw_cyclonedds_cpp "$dir/rmw.txt")
    if (( total > 0 && good == total )); then
        record "scene рантайм" ok "процессов мира с rmw $total, у всех librmw_cyclonedds_cpp"
    else
        record "scene рантайм" отказ "процессов с rmw $total, на CycloneDDS $good"
    fi

    local topic rtf
    topic="$(clean "$ROS_ENV && timeout 15 gz topic -l" 2>/dev/null | grep -m1 -E '^/world/[^/]+/stats$')"
    rtf="$( [[ -n $topic ]] && clean "$ROS_ENV && timeout 15 gz topic -e -t '$topic' -n 1" 2>/dev/null | grep -m1 real_time_factor)"
    record "scene RTF" число "${rtf:-не прочитан} (${topic:-темы stats нет})"

    clean "cd '$REPO_ROOT' && timeout 120 '$OWN_CLI' run '$SCENARIO' -R '$REGISTRY'" > "$dir/run.txt" 2>&1
    code=$?
    cat "$dir/run.txt"
    if (( code == 0 )); then
        record "scene навык" ok "$(grep -m1 '^выполнено' "$dir/run.txt")"
    else
        record "scene навык" "отказ, код $code" "$(grep -E '^(отказ|причина|g1:)' "$dir/run.txt" | tr '\n' ' ' | grep . || tail -1 "$dir/run.txt")"
    fi

    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/witness.py' '$PROFILE' 2 30 '$dir/after.json'" > "$dir/after.txt" 2>&1
    cat "$dir/after.txt"
    if clean "cd '$REPO_ROOT' && '$OWN_PY' '$PY_DIR/moved.py' '$dir/before.json' '$dir/after.json' '$SCENARIO' '$PROFILE'" > "$dir/moved.txt" 2>&1; then
        record "scene движение" ok "$(tail -1 "$dir/moved.txt")"
    else
        record "scene движение" отказ "$(tail -1 "$dir/moved.txt")"
    fi

    # Which processes of the world can hear SIGINT at all: bit 2 of SigIgn. A background
    # job of this script starts with SIGINT ignored, and ros2 launch listens only through
    # Python's handlers (ADR-0036).
    local p mask
    for p in $(descendants "$pgid"); do
        mask="$(sed -n 's/^SigIgn:[[:space:]]*//p' "/proc/$p/status" 2>/dev/null)"
        [[ -n $mask ]] || continue
        info "$(cat "/proc/$p/comm" 2>/dev/null)[$p] группа $(ps -o pgid= -p "$p" | tr -d ' ') SigIgn $mask — SIGINT $( (( 0x$mask & 2 )) && echo игнорируется || echo доходит)"
    done

    # SIGTERM to sim run alone, not to the group: the platform's stop is under test
    # (ADR-0032 fixed it; stage 4 could not check it with SIGINT from a background job).
    kill -TERM "$pgid" 2>/dev/null
    waited=0
    while running "$pgid" && (( waited < 40 )); do sleep 1; waited=$(( waited + 1 )); done
    local left
    left="$(scene_pids | tr '\n' ' ')"
    if ! running "$pgid" && [[ -z ${left// } ]]; then
        record "scene остановка" ok "sim run вышел за $waited с, процессов сцены не осталось"
    else
        record "scene остановка" отказ "через $waited с: sim run $(running "$pgid" && echo жив || echo вышел), осталось: ${left:-—}"
        ps -o pid=,args= -p "$(echo "$left" | tr ' ' ',' | sed 's/,$//')" 2>/dev/null | cut -c1-160 | sed 's/^/      /'
        tail -15 "$dir/sim.log" | sed 's/^/      /'
        # shellcheck disable=SC2086
        [[ -n ${left// } ]] && kill -KILL $left 2>/dev/null
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
