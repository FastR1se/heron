#!/usr/bin/env bash
#
# Phase 4, stage 6: experiment 0001 -- one skill, one goal, one set of gains and one
# duration on G1 in the vendor's MuJoCo twin and in Gazebo through the bridge, measured
# (docs/plans/phase4-dds-bridge.md, "Этап 6"; docs/experiments/0001-g1-dds-twins.md).
#
# Every trial starts a fresh world, so every trial starts from a robot that has just
# fallen (ADR-0018) and waited the same time. The numbers come from two sources that share
# no code: the run record the executor writes (ADR-0038), read with mcap by
# scripts/phase4-stage6-metrics.py, and a DDS witness of its own with a reader deep enough
# to keep every sample -- the state rate and the steady error after the run. Nothing stops
# at the first failure; the tables at the end are a summary, the evidence is above them.
#
#   env     conditions of the experiment (machine, versions, commits) into conditions.md;
#           the execution venv synced, the G1 package and mcap in both venvs
#   mujoco  TRIALS trials in the vendor's twin (with its window: it has no headless mode)
#   gazebo  TRIALS trials in Gazebo through the bridge, headless unless GUI=1
#   table   the metrics of all trials into table.md
#   media   one more trial per world with the window on and the screen recorded into
#           docs/media (ffmpeg, X11); its numbers stay out of the table
#
# Usage:
#   bash scripts/phase4-stage6.sh                        # env mujoco gazebo table
#   TRIALS=5 bash scripts/phase4-stage6.sh mujoco table
#   bash scripts/phase4-stage6.sh media
#
# Prerequisites: those of scripts/phase4-stage4.sh (the twin built by phase4-stage1.sh twin,
# venv-own-lib by phase4-probe.sh 0.5) and scripts/phase4-stage5.sh env (unitree_hg and the
# bridge built into $HOME/phase4-stage5/ws). A graphical session. ffmpeg for media.
#
# Writes outside $STAGE_ROOT: `uv sync` and `uv pip install` in the execution venv; `pip
# install` into venv-own-lib; the vendor descriptions (gitignored); media -> docs/media.
#
# Roughly 1-2 minutes per trial.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO_MUJOCO="examples/scenarios/humanoid_mujoco.yaml"
readonly SCENARIO_GAZEBO="examples/scenarios/humanoid.yaml"
readonly PROFILE="examples/robots/g1_23dof/profile.yaml"
readonly REGISTRY="examples/robots"
readonly PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly VENDOR="$REPO_ROOT/examples/robots/g1_23dof/vendor"
readonly METRICS="$REPO_ROOT/scripts/phase4-stage6-metrics.py"
readonly MEDIA="$REPO_ROOT/docs/media"
readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase4-stage6}"
STAGE1_ROOT="${STAGE1_ROOT:-$HOME/phase4-stage1}"
STAGE5_ROOT="${STAGE5_ROOT:-$HOME/phase4-stage5}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
TRIALS="${TRIALS:-3}"
GUI="${GUI:-0}"
SETTLE="${SETTLE:-8}"                      # seconds between the first state and the run
HOLD="${HOLD:-3}"                          # seconds of the witness's window after the run
readonly STAGE_ROOT STAGE1_ROOT STAGE5_ROOT PROBE_ROOT TRIALS GUI SETTLE HOLD
readonly PY_DIR="$STAGE_ROOT/py"
readonly WS="$STAGE5_ROOT/ws"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly OWN_PY="$VENV_OWN/bin/python"
readonly OWN_CLI="$VENV_OWN/bin/$APP"
readonly TWIN_BIN="$STAGE1_ROOT/unitree_mujoco/simulate/build/unitree_mujoco"
readonly TWIN_LIBS="$STAGE1_ROOT/unitree_sdk2/install/lib:$STAGE1_ROOT/mujoco-3.3.6/lib"
readonly ALL_STEPS="env mujoco gazebo table media"
readonly DEFAULT_STEPS="env mujoco gazebo table"

[[ $TRIALS =~ ^[1-9][0-9]*$ ]] || { printf 'TRIALS=%s: нужно целое больше нуля\n' "$TRIALS"; exit 2; }

mkdir -p "$STAGE_ROOT" "$PY_DIR"
LOG="$STAGE_ROOT/stage6-$(date +%Y%m%d-%H%M%S).log"
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
# cannot decide an outcome. `run` gets no ROS 2 and no scene environment from here: the
# DDS adapter must not load rclpy (ADR-0022), and the world's environment is run's to apply
# (ADR-0039).
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage6}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

readonly ROS_ENV="source '$ROS_SETUP'"
readonly WS_ENV="$ROS_ENV && source '$WS/install/setup.bash'"
readonly TWIN_ENV="export PATH='$(dirname "$TWIN_BIN")':\$PATH LD_LIBRARY_PATH='$TWIN_LIBS'"

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
leftovers() { pgrep -x unitree_mujoco; pgrep -f 'gz sim'; }

version_of() {  # version_of <python> <distribution>
    "$1" -c "import importlib.metadata as m; print(m.version('$2'))" 2>/dev/null || echo нет
}

commit_of() { git -C "$1" rev-parse --short HEAD 2>/dev/null || echo "нет ($1)"; }

# ---------------------------------------------------------------------------
# The witness: a DDS participant of its own, in the adapter's venv, on the link the
# package builds from the profile. Its reader keeps every sample between two takes, so
# what it counts is the twin's state rate, not its own polling (stage 5, observations).
# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/window.py" <<'PY'
# Waits up to <wait> s for the first LowState, then reads for <window> s. Writes the rate,
# the world's pace and every contract joint's mean and spread, in the contract's order.
import json, math, statistics, sys, time
import yaml
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.qos import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_g1.channel import STATE_TOPIC, LinkConfig
from unitree_g1.slots import slots_of

profile_path, target, wait, window, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
profile = yaml.safe_load(open(profile_path, encoding="utf-8"))
link = LinkConfig.from_params(profile["adapters"][target]["params"])
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
            got.append((now, sample.tick, sample.mode_machine, [sample.motor_state[s].q for s in slots]))
    if first is not None and now - first >= window:
        break
    if first is None and now - start >= wait:
        break
    time.sleep(0.01)
if len(got) < 2:
    print(f"свидетель: LowState принято {len(got)} за {time.monotonic() - start:.0f} с (домен {link.domain})")
    sys.exit(1)
ticks = sorted({item[1] for item in got})
steps = [b - a for a, b in zip(ticks, ticks[1:])]
wall = got[-1][0] - got[0][0]
series = list(zip(*(item[3] for item in got)))
result = {
    "samples": len(got), "distinct_ticks": len(ticks), "wall_s": wall, "tick_span_ms": ticks[-1] - ticks[0],
    "rate_hz": (len(ticks) - 1) / wall if wall > 0 else math.nan,
    "rtf": (ticks[-1] - ticks[0]) / 1000 / wall if wall > 0 else math.nan,
    "median_tick_step_ms": statistics.median(steps) if steps else math.nan,
    "first_after_s": first - start, "mode_machine": got[-1][2], "tick": ticks[-1],
    "q_mean": [statistics.fmean(s) for s in series], "q_std": [statistics.pstdev(s) for s in series],
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
print(f"свидетель: LowState {len(got)} за {wall:.2f} с, первое через {first - start:.1f} с; различных tick "
      f"{len(ticks)} ({result['rate_hz']:.0f} Гц), медианный шаг tick {result['median_tick_step_ms']} мс, "
      f"RTF {result['rtf']:.2f}, mode_machine {got[-1][2]}")
PY
}

# ---------------------------------------------------------------------------

REC_PID=""
screen_start() {  # screen_start <mp4> <seconds>
    local size
    REC_PID=""
    size="$(xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
    [[ -n $size ]] || size="$(xrandr 2>/dev/null | awk '/\*/ {print $1; exit}')"
    if ! command -v ffmpeg >/dev/null || [[ -z $size || -z ${DISPLAY:-} ]]; then
        record "media запись" невозможна "ffmpeg $(command -v ffmpeg >/dev/null && echo есть || echo нет), размер экрана ${size:-не прочитан}, DISPLAY ${DISPLAY:-нет}, сессия ${XDG_SESSION_TYPE:-?}"
        return
    fi
    [[ ${XDG_SESSION_TYPE:-} == wayland ]] && info "сессия Wayland: x11grab снимает только окна XWayland"
    start_group "$1.log" "exec ffmpeg -y -loglevel error -f x11grab -video_size '$size' -framerate 15 -i '$DISPLAY' -t $2 -pix_fmt yuv420p '$1'"
    REC_PID=$LAST_GROUP
}

screen_finish() {  # screen_finish <mp4> <gif>
    local waited=0
    [[ -n $REC_PID ]] || return 0
    while running "$REC_PID" && (( waited < 60 )); do sleep 1; waited=$(( waited + 1 )); done
    if [[ -s $1 ]] && ffmpeg -y -loglevel error -i "$1" \
        -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$2"; then
        record "media $(basename "$2")" ok "$2, $(du -h "$2" | cut -f1)"
    else
        record "media $(basename "$2")" отказ "$(tail -3 "$1.log" 2>/dev/null | tr '\n' ' ')"
    fi
}

gz_rtf() {
    local topic
    topic="$(clean "$ROS_ENV && timeout 15 gz topic -l" 2>/dev/null | grep -m1 -E '^/world/[^/]+/stats$')"
    [[ -n $topic ]] || { echo "темы stats нет"; return; }
    clean "$ROS_ENV && timeout 15 gz topic -e -t '$topic' -n 1" 2>/dev/null | grep -m1 real_time_factor | sed 's/^ *//' | grep . ||
        echo "не прочитан"
}

# The platform's own stop, judged by descent: the world lives in a session of its own (ADR-0037).
stop_world() {  # stop_world <pid of sim run>
    local pid="$1" waited=0 tree left="" p
    tree="$(descendants "$pid" | tr '\n' ' ')"
    kill -INT "$pid" 2>/dev/null
    while running "$pid" && (( waited < 40 )); do sleep 1; waited=$(( waited + 1 )); done
    for p in $tree; do running "$p" && left+="$p "; done
    if [[ -z $left ]]; then
        info "мир остановлен платформой за $waited с"
    else
        info "через $waited с остались процессы мира: $left — KILL"
        # shellcheck disable=SC2086
        kill -KILL $left 2>/dev/null
    fi
    stop_group "$pid" 5
}

# One trial: a fresh world, the first state, the fall, the run with a record, the window
# after it, the stop. Numbers into $dir/metrics.json.
trial() {  # trial <mujoco|gazebo> <label> <gui 0|1> <gif or "">
    local world="$1" label="$2" gui="$3" gif="$4"
    local dir="$STAGE_ROOT/trials/$label" scenario target env flags="" pgid reader code
    rm -rf "$dir"; mkdir -p "$dir"
    step "$label — мир, первое состояние, падение $SETTLE с, run --record, окно $HOLD с, остановка"
    case "$world" in
        mujoco) scenario="$SCENARIO_MUJOCO"; target="sim.mujoco"; env="$TWIN_ENV" ;;
        gazebo) scenario="$SCENARIO_GAZEBO"; target="sim.gazebo"; env="$WS_ENV"
                (( gui )) || flags="--headless --no-rviz" ;;
    esac
    if [[ -n "$(leftovers)" ]]; then
        record "$label" пропущено "остались процессы прошлого мира: $(leftovers | tr '\n' ' ')"; return
    fi

    start_group "$dir/sim.log" "$env && cd '$REPO_ROOT' && exec '$EXEC_CLI' sim run '$scenario' --into '$dir/world' -R '$REGISTRY' $flags"
    pgid=$LAST_GROUP
    start_group "$dir/first.txt" "cd '$REPO_ROOT' && exec timeout 220 '$OWN_PY' '$PY_DIR/window.py' '$PROFILE' $target 180 0.5 '$dir/first.json'"
    reader=$LAST_GROUP
    while running "$reader"; do
        running "$pgid" || { info "sim run завершился раньше, чем пришло состояние"; stop_group "$reader" 5; break; }
        sleep 1
    done
    cat "$dir/first.txt"
    if [[ ! -s "$dir/first.json" ]]; then
        record "$label" отказ "мир не ответил: $(tail -1 "$dir/first.txt")"
        tail -30 "$dir/sim.log" | sed 's/^/      /'
        stop_world "$pgid"; return
    fi
    sleep "$SETTLE"

    [[ -n $gif ]] && screen_start "$dir/screen.mp4" 15
    clean "cd '$REPO_ROOT' && timeout 120 '$OWN_CLI' run '$scenario' -R '$REGISTRY' --record '$dir/run.mcap'" > "$dir/run.txt" 2>&1
    code=$?
    cat "$dir/run.txt"
    sleep 2
    clean "cd '$REPO_ROOT' && timeout 60 '$OWN_PY' '$PY_DIR/window.py' '$PROFILE' $target 30 '$HOLD' '$dir/hold.json'" > "$dir/hold.txt" 2>&1
    cat "$dir/hold.txt"
    [[ $world == gazebo ]] && info "RTF по статистике Gazebo: $(gz_rtf)"
    [[ -n $gif ]] && screen_finish "$dir/screen.mp4" "$gif"
    stop_world "$pgid"

    if [[ ! -s "$dir/run.mcap" ]]; then
        record "$label" "отказ, код $code" "записи нет: $(grep -E '^(отказ|причина|g1:)' "$dir/run.txt" | tr '\n' ' ' | grep . || tail -1 "$dir/run.txt")"
        return
    fi
    local hold=()
    [[ -s "$dir/hold.json" ]] && hold=(--hold "$dir/hold.json")
    if "$EXEC_PY" "$METRICS" run "$dir/run.mcap" --label "$label" "${hold[@]}" --out "$dir/metrics.json" > "$dir/metrics.txt" 2>&1; then
        record "$label" "$( (( code == 0 )) && echo ok || echo "навык: код $code")" "$(tail -1 "$dir/metrics.txt")"
    else
        record "$label" отказ "метрики не посчитаны: $(tail -3 "$dir/metrics.txt" | tr '\n' ' ')"
    fi
}

# ---------------------------------------------------------------------------

row() { printf '| %s | %s |\n' "$1" "$2"; }

step_env() {
    step "Окружение"
    info "лог: $LOG"
    local ok=1 conditions="$STAGE_ROOT/conditions.md" pkg

    [[ -f $ROS_SETUP ]] || { record env отказ "нет $ROS_SETUP (scripts/phase0.sh)"; ok=0; }
    [[ -x $TWIN_BIN ]] || { record env отказ "нет $TWIN_BIN: bash scripts/phase4-stage1.sh twin"; ok=0; }
    [[ -f $WS/install/setup.bash ]] || { record env отказ "нет $WS: bash scripts/phase4-stage5.sh env"; ok=0; }
    [[ -n ${DISPLAY:-} ]] || { record env отказ "DISPLAY не задан: двойнику MuJoCo нужно окно"; ok=0; }

    if [[ ! -x $EXEC_PY ]]; then
        record env отказ "нет окружения исполнения $EXEC_VENV (scripts/phase0.sh install exec)"; ok=0
    else
        # uv sync removes what pyproject.toml does not list -- the robot package among it.
        info "окружение исполнения: uv sync --extra dev (с mcap), затем пакет G1"
        clean "cd '$REPO_ROOT' && uv sync --extra dev && uv pip install --python '$EXEC_PY' -e '$PACKAGE'" ||
            { record env отказ "uv sync или установка пакета G1 не прошли — причина выше"; ok=0; }
    fi
    if [[ ! -x $OWN_PY ]] || ! "$OWN_PY" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; then
        record env отказ "нет $VENV_OWN с SDK: bash scripts/phase4-probe.sh 0.5"; ok=0
    else
        info "окружение адаптера: зависимости платформы и mcap, затем платформа и пакет G1 без зависимостей"
        clean "'$OWN_PY' -m pip install -q --no-cache-dir pydantic pyyaml typer packaging mcap && '$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$PACKAGE'" ||
            { record env отказ "установка в $VENV_OWN не прошла — причина выше"; ok=0; }
    fi
    clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" > /dev/null ||
        { record env отказ "описания производителя не загрузились"; ok=0; }

    {
        printf '| Что | Значение |\n|---|---|\n'
        row "машина" "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1), потоков $(nproc), ОЗУ $(awk '/MemTotal/ {printf "%.1f ГБ", $2 / 1048576}' /proc/meminfo)"
        row "GPU" "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | grep . || echo 'не прочитан')"
        row "ОС" "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}"), ядро $(uname -r), сессия ${XDG_SESSION_TYPE:-?}"
        row "платформа" "$(commit_of "$REPO_ROOT"), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l)"
        row "контракт G1" "$("$EXEC_CLI" contract id "$REPO_ROOT/$PROFILE" 2>/dev/null || echo '?')"
        row "Gazebo" "$(clean "$ROS_ENV && gz sim --versions 2>/dev/null | head -1" || echo '?')"
        for pkg in ros-jazzy-ros-base ros-jazzy-ros2-control ros-jazzy-gz-ros2-control ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-cyclonedds; do
            row "$pkg" "$(dpkg -s "$pkg" 2>/dev/null | sed -n 's/^Version: //p' | grep . || echo 'не установлен')"
        done
        row "MuJoCo двойника" "$(basename "$(dirname "${TWIN_LIBS##*:}")")"
        row "unitree_mujoco" "$(commit_of "$STAGE1_ROOT/unitree_mujoco")"
        row "unitree_sdk2" "$(commit_of "$STAGE1_ROOT/unitree_sdk2/src")"
        row "unitree_sdk2_python" "$(commit_of "$PROBE_ROOT/unitree_sdk2_python")"
        row "unitree_ros2 (unitree_hg)" "$(commit_of "$PROBE_ROOT/ws/src/unitree_ros2")"
        row "описание URDF (unitree_ros)" "$(commit_of "$VENDOR/unitree_ros")"
        row "описание MJCF и сцена (unitree_mujoco)" "$(commit_of "$VENDOR/unitree_mujoco")"
        row "Python исполнения / адаптера" "$("$EXEC_PY" -V 2>&1) / $("$OWN_PY" -V 2>&1)"
        row "cyclonedds (адаптер, свидетель)" "$(version_of "$OWN_PY" cyclonedds)"
        row "mcap (адаптер / метрики)" "$(version_of "$OWN_PY" mcap) / $(version_of "$EXEC_PY" mcap)"
        row "ГСЧ" "платформа случайных чисел не использует; зерна нет"
        row "постановка" "попыток на мир $TRIALS; от первого состояния до run $SETTLE с; окно после run $HOLD с через 2 с; Gazebo $( (( GUI )) && echo 'с окном и RViz' || echo 'без окна и RViz'); MuJoCo с окном"
        row "записи" "$STAGE_ROOT/trials/<мир>-<n>/run.mcap"
    } > "$conditions"
    cat "$conditions"
    (( ok )) && record env ok "условия в $conditions"
}

step_trials() {  # step_trials <mujoco|gazebo>
    local n
    if [[ $1 == mujoco || $GUI == 1 ]] && [[ -z ${DISPLAY:-} ]]; then
        record "$1" пропущено "DISPLAY не задан"; return
    fi
    for (( n = 1; n <= TRIALS; n++ )); do trial "$1" "$1-$n" "$GUI" ""; done
}

step_table() {
    step "table — прогоны обоих миров"
    local files=()
    mapfile -t files < <(ls "$STAGE_ROOT"/trials/mujoco-[0-9]*/metrics.json "$STAGE_ROOT"/trials/gazebo-[0-9]*/metrics.json 2>/dev/null)
    if (( ${#files[@]} == 0 )); then record table пропущено "метрик нет"; return; fi
    "$EXEC_PY" "$METRICS" table "${files[@]}" | tee "$STAGE_ROOT/table.md"
    record table ok "$STAGE_ROOT/table.md, прогонов ${#files[@]}"
}

step_media() {
    [[ -n ${DISPLAY:-} ]] || { record media пропущено "DISPLAY не задан"; return; }
    mkdir -p "$MEDIA"
    trial mujoco mujoco-media 1 "$MEDIA/phase4-g1-mujoco.gif"
    trial gazebo gazebo-media 1 "$MEDIA/phase4-g1-gazebo-bridge.gif"
}

summary() {
    step "Итог"
    local row_ check status note
    printf '\n| Проверка | Результат | Подробности |\n|---|---|---|\n'
    for row_ in "${RESULTS[@]}"; do
        IFS='|' read -r check status note <<< "$row_"
        printf '| %s | %s | %s |\n' "$check" "$status" "$note"
    done
    printf '\nОтправьте целиком файл:\n    %s\n' "$LOG"
}

main() {
    local steps=("$@") s
    (( ${#steps[@]} )) || read -r -a steps <<< "$DEFAULT_STEPS"
    for s in "${steps[@]}"; do
        [[ " $ALL_STEPS " == *" $s "* ]] || { printf 'неизвестный шаг: %s. Доступны: %s\n' "$s" "$ALL_STEPS"; exit 2; }
    done
    write_probes
    for s in "${steps[@]}"; do
        case "$s" in
            env)    step_env ;;
            mujoco) step_trials mujoco ;;
            gazebo) step_trials gazebo ;;
            table)  step_table ;;
            media)  step_media ;;
        esac
    done
    summary
}

main "$@"
