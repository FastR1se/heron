#!/usr/bin/env bash
#
# Phase 4, stage 4: G1 in the vendor's MuJoCo twin, driven through the DDS adapter
# (docs/plans/phase4-dds-bridge.md, "Этап 4"; ADR-0030, ADR-0031).
#
# Built like scripts/phase4-stage1.sh and on its lesson: every check stands on a
# confirmation taken by a different tool than the one under test. Motion is read
# by a DDS participant of its own, not by the adapter; the command sim run started
# is read off /proc, not off sim run's output. Nothing stops at the first failure.
# The table at the end is a summary; the evidence is above it, all in one log.
#
# Order follows ADR-0020, "the adapter against a ready twin first":
#   env    versions, commit; the execution venv synced and the G1 package installed;
#          the platform and the package reinstalled into the adapter's venv (entry
#          points are written at install time -- stage 1, run 1); descriptions fetched;
#          the twin built by phase4-stage1.sh found
#   reqs   validate (MJCF rules on real files), env check with and without the twin
#          on PATH, sim run --dry-run and --headless
#   link   the twin started by hand, exactly as stage 1 did; `run` from the adapter's
#          venv -- the first run of CycloneChannel; a witness reads LowState before
#          and after
#   scene  the same, with the twin started by `sim run`; its argv compared with the
#          dry run; stopped by Ctrl+C to sim run, and the twin must be gone after
#
# Usage:
#   bash scripts/phase4-stage4.sh                # all steps, in order
#   bash scripts/phase4-stage4.sh env link       # selected steps
#
# Prerequisites: scripts/phase0.sh (the execution venv), scripts/phase4-probe.sh 0.5
# (venv-own-lib: cyclonedds 0.10.2 and the SDK), scripts/phase4-stage1.sh twin (the
# vendor's simulator built at the pinned commits). A graphical session: the twin
# has no headless mode (ADR-0023).
#
# Writes outside $STAGE_ROOT: `uv sync` and `uv pip install` in the execution venv;
# `pip install --no-deps -e` of the platform and the G1 package into venv-own-lib;
# the vendor descriptions under examples/robots/g1_23dof/vendor (gitignored).
#
# Roughly 5-10 minutes.

set -uo pipefail

readonly APP="heron"                       # the command name; section 17 rename touches this line

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly EXEC_PY="$EXEC_VENV/bin/python"
readonly EXEC_CLI="$EXEC_VENV/bin/$APP"
readonly SCENARIO="examples/scenarios/humanoid_mujoco.yaml"
readonly PROFILE="examples/robots/g1_23dof/profile.yaml"
readonly REGISTRY="examples/robots"
readonly SCENE="$REPO_ROOT/examples/robots/g1_23dof/vendor/unitree_mujoco/unitree_robots/g1/scene_23dof.xml"
readonly PACKAGE="$REPO_ROOT/examples/plugins/unitree_g1"
readonly DDS_DOMAIN=1                       # profile.yaml, adapters.sim.mujoco.params.domain

STAGE_ROOT="${STAGE_ROOT:-$HOME/phase4-stage4}"
STAGE1_ROOT="${STAGE1_ROOT:-$HOME/phase4-stage1}"
PROBE_ROOT="${PROBE_ROOT:-$HOME/phase4-probe}"
readonly STAGE_ROOT STAGE1_ROOT PROBE_ROOT
readonly PY_DIR="$STAGE_ROOT/py"
readonly VENV_OWN="$PROBE_ROOT/venv-own-lib"
readonly OWN_PY="$VENV_OWN/bin/python"
readonly OWN_CLI="$VENV_OWN/bin/$APP"
readonly TWIN_BIN="$STAGE1_ROOT/unitree_mujoco/simulate/build/unitree_mujoco"
readonly TWIN_LIBS="$STAGE1_ROOT/unitree_sdk2/install/lib:$STAGE1_ROOT/mujoco-3.3.6/lib"
readonly ALL_STEPS="env reqs link scene"

mkdir -p "$STAGE_ROOT" "$PY_DIR"
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

# Nothing inherited but HOME and a plain PATH (stage 1): what the calling shell has
# sourced cannot decide an outcome. No ROS 2 anywhere here -- the adapter's process
# must not load rclpy (ADR-0022), and sim run of a MuJoCo world does not need it.
readonly BASE_ENV=(env -i "HOME=$HOME" "USER=${USER:-stage4}" LANG=C.UTF-8 "TERM=${TERM:-dumb}"
    "PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
GUI_VARS=()
for var in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    [[ -n ${!var:-} ]] && GUI_VARS+=("$var=${!var}")
done
readonly GUI_VARS

clean()     { "${BASE_ENV[@]}" bash --noprofile --norc -c "$1"; }
clean_gui() { "${BASE_ENV[@]}" "${GUI_VARS[@]}" bash --noprofile --norc -c "$1"; }

# How the twin's side finds the vendor's program and its libraries. The same line
# stage 1 started the twin with; here it is what sim run inherits.
readonly TWIN_ENV="export PATH='$(dirname "$TWIN_BIN")':\$PATH LD_LIBRARY_PATH='$TWIN_LIBS'"

STARTED=()
LAST_GROUP=""
start_group() {  # start_group <log> <command>  -- always with the GUI variables
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

wait_for_line() {  # wait_for_line <file> <text> <seconds> <pid>
    local waited=0
    until grep -qF "$2" "$1" 2>/dev/null; do
        kill -0 "$4" 2>/dev/null || return 1
        (( waited >= $3 )) && return 1
        sleep 1; waited=$(( waited + 1 ))
    done
}

running() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ $state != Z* ]]; }
twin_pids() { pgrep -x unitree_mujoco; }
# The twin lives in a session of its own, not in the group of sim run (ADR-0037): found by descent.
descendants() { local p; echo "$1"; for p in $(pgrep -P "$1"); do descendants "$p"; done; }
group_of() {
    local p
    for p in $(descendants "$1"); do
        [[ "$(cat "/proc/$p/comm" 2>/dev/null)" == unitree_mujoco ]] && { echo "$p"; return; }
    done
    echo ""
}

entry_names() {  # entry_names <python> <group suffix>
    "$1" -c "from importlib.metadata import entry_points; print(' '.join(sorted(e.name for e in entry_points(group='$APP.$2'))) or 'пусто')" 2>/dev/null ||
        echo "не читается"
}

# ---------------------------------------------------------------------------
# Python probes: the witness runs in the adapter's venv, as its own participant.
# ---------------------------------------------------------------------------

write_probes() {
    cat > "$PY_DIR/witness.py" <<'PY'
# LowState from a participant of its own: what the robot did, not what the adapter says.
# Writes the newest state's tick, mode_machine and every slot's q to a JSON file.
import json, sys, time
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

domain_id, window, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
XML = ('<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="lo"/>'
       '</Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery>'
       '<ParticipantIndex>auto</ParticipantIndex><Peers><Peer address="127.0.0.1"/></Peers>'
       '</Discovery></Domain></CycloneDDS>')
domain = Domain(domain_id, XML)
participant = DomainParticipant(domain_id)
reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_))
start, got = time.monotonic(), []
while time.monotonic() - start < window + 10:
    for sample in reader.take(N=256):
        if hasattr(sample, "tick"):
            got.append((time.monotonic(), sample.tick, sample.mode_machine,
                        [motor.q for motor in sample.motor_state]))
    if got and time.monotonic() - got[0][0] >= window:
        break
    time.sleep(0.005)
if len(got) < 2:
    print(f"свидетель: LowState принято {len(got)} за {window + 10:g} с")
    sys.exit(1)
(t0, k0, machine, _), (t1, k1, _, q) = got[0], got[-1]
wall = max(t1 - t0, 1e-9)
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"tick": k1, "mode_machine": machine, "q": q}, handle)
print(f"свидетель: LowState {len(got)} за {wall:.2f} с ({(len(got) - 1) / wall:.0f} Гц приёма), "
      f"tick +{k1 - k0} мс мира, фактор {(k1 - k0) / 1000 / wall:.2f}, mode_machine {machine}")
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
print(f"до цели было {max(was):.3f} рад, стало {now[worst]:.3f} ({order[worst]}); "
      f"руки {max(was[i] for i in arms):.3f} -> {max(now[i] for i in arms):.3f}, "
      f"ноги и пояс {max(was[i] for i in legs):.3f} -> {max(now[i] for i in legs):.3f}; "
      f"tick {before['tick']} -> {after['tick']}")
sys.exit(0 if now[worst] <= 0.5 * max(was) else 1)
PY
}

# ---------------------------------------------------------------------------

step_env() {
    step "Окружение"
    info "лог: $LOG"
    info "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}"), ядро $(uname -r)"
    info "репозиторий: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), изменённых файлов $(git -C "$REPO_ROOT" status --short 2>/dev/null | wc -l)"
    info "DISPLAY: ${DISPLAY:-не задан}; lo: $(ip -o link show lo 2>/dev/null | sed -E 's/.*<([^>]*)>.*/\1/')"
    local ok=1

    if [[ ! -x "$EXEC_PY" ]]; then
        record env отказ "нет окружения исполнения $EXEC_VENV (scripts/phase0.sh install exec)"; ok=0
    else
        # uv sync removes what pyproject.toml does not list -- the robot package among it,
        # so the package goes in after every sync.
        info "окружение исполнения: uv sync, затем пакет G1"
        if clean "cd '$REPO_ROOT' && uv sync --extra dev && uv pip install --python '$EXEC_PY' -e '$PACKAGE'"; then
            info "бэкенды: $(entry_names "$EXEC_PY" sim_backends); двойники: $(entry_names "$EXEC_PY" twins)"
            [[ $(entry_names "$EXEC_PY" twins) == *unitree_g1_mujoco* && $(entry_names "$EXEC_PY" sim_backends) == *mujoco* ]] ||
                { record env отказ "в окружении исполнения нет бэкенда mujoco или двойника unitree_g1_mujoco"; ok=0; }
        else
            record env отказ "uv sync или установка пакета G1 не прошли — причина выше"; ok=0
        fi
    fi

    if [[ ! -x "$OWN_PY" ]] || ! "$OWN_PY" -c 'import unitree_sdk2py.core.channel' >/dev/null 2>&1; then
        record env отказ "нет $VENV_OWN с SDK: bash scripts/phase4-probe.sh 0.5"; ok=0
    else
        # Always reinstalled: an older install carries older entry points.
        info "окружение адаптера $VENV_OWN: платформа и пакет G1 без зависимостей, заново"
        if clean "'$OWN_PY' -m pip install -q --no-cache-dir pydantic pyyaml typer packaging && '$OWN_PY' -m pip install -q --no-cache-dir --no-deps -e '$REPO_ROOT' -e '$PACKAGE'"; then
            info "cyclonedds: $("$OWN_PY" -m pip show cyclonedds 2>/dev/null | sed -n 's/^Version: //p'); rclpy: $("$OWN_PY" -c 'import rclpy' >/dev/null 2>&1 && echo ЕСТЬ || echo нет)"
            info "адаптеры: $(entry_names "$OWN_PY" adapters); двойники: $(entry_names "$OWN_PY" twins)"
        else
            record env отказ "установка в $VENV_OWN не прошла — причина выше"; ok=0
        fi
    fi

    info "описания производителя: fetch_description.py"
    clean "cd '$REPO_ROOT' && '$EXEC_PY' examples/robots/g1_23dof/fetch_description.py" ||
        { record env отказ "описания не загрузились — причина выше"; ok=0; }
    [[ -f "$SCENE" ]] && info "сцена: $SCENE, sha256 $(sha256sum "$SCENE" | cut -c1-16)"

    if [[ -x "$TWIN_BIN" ]]; then
        info "двойник: $TWIN_BIN"
    else
        record env отказ "нет $TWIN_BIN: bash scripts/phase4-stage1.sh twin"; ok=0
    fi
    (( ok )) && record env ok "предусловия на месте"
}

step_reqs() {
    local out="$STAGE_ROOT/reqs" code
    step "reqs — validate профиля и сцены на настоящих файлах"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' validate '$PROFILE' '$SCENARIO' -R '$REGISTRY'" > "$out-validate.txt" 2>&1
    code=$?
    cat "$out-validate.txt"
    if (( code == 0 )) && grep -qF "[mjcf.differs_from_urdf]" "$out-validate.txt"; then
        record "reqs validate" ok "0 ошибок; $(grep -c '^внимание' "$out-validate.txt") предупреждений, среди них mjcf.differs_from_urdf"
    else
        record "reqs validate" "отказ, код $code" "$(grep -m1 '^ошибка' "$out-validate.txt" || tail -1 "$out-validate.txt")"
    fi

    step "reqs — env check: адаптер в своём окружении, двойник на PATH, DISPLAY"
    clean_gui "$TWIN_ENV && cd '$REPO_ROOT' && '$OWN_CLI' env check '$SCENARIO' -R '$REGISTRY'" > "$out-env.txt" 2>&1
    code=$?
    cat "$out-env.txt"
    local witness
    witness="$(clean_gui "$TWIN_ENV && command -v unitree_mujoco && test -n \"\$DISPLAY\" && '$OWN_PY' -c 'import cyclonedds, unitree_sdk2py; print(\"импорт есть\")'" 2>&1 | tr '\n' ' ')"
    info "command -v, DISPLAY, импорт: $witness"
    if (( code == 0 )); then
        record "reqs env check, всё есть" ok "$(grep -c '^проверено' "$out-env.txt") стороны проверены; $witness"
    else
        record "reqs env check, всё есть" "отказ, код $code" "$(grep -m2 '^ошибка' "$out-env.txt" | tr '\n' ' ')"
    fi

    clean "cd '$REPO_ROOT' && '$OWN_CLI' env check '$SCENARIO' -R '$REGISTRY'" > "$out-env-bare.txt" 2>&1
    code=$?
    if (( code == 1 )) && grep -qF "программа unitree_mujoco" "$out-env-bare.txt" && grep -qF "DISPLAY" "$out-env-bare.txt"; then
        record "reqs env check, без двойника" ok "называет программу и DISPLAY"
    else
        record "reqs env check, без двойника" отказ "код $code; $(grep -m2 '^ошибка' "$out-env-bare.txt" | tr '\n' ' ')"
    fi

    step "reqs — sim run --dry-run и --headless"
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' sim run '$SCENARIO' --into '$STAGE_ROOT/dry' -R '$REGISTRY' --dry-run" > "$out-dry.txt" 2>&1
    code=$?
    cat "$out-dry.txt"
    if (( code == 0 )) && grep -qE '^  unitree_mujoco -r g1 -s .*scene_23dof\.xml -i 1 -n lo$' "$out-dry.txt"; then
        record "reqs sim run --dry-run" ok "$(grep -m1 '^  unitree_mujoco' "$out-dry.txt" | sed 's/^ *//')"
    else
        record "reqs sim run --dry-run" "отказ, код $code" "$(tail -1 "$out-dry.txt")"
    fi
    clean "cd '$REPO_ROOT' && '$EXEC_CLI' sim run '$SCENARIO' --into '$STAGE_ROOT/dry' -R '$REGISTRY' --headless" > "$out-headless.txt" 2>&1
    code=$?
    if (( code == 1 )) && grep -qF "безголового режима" "$out-headless.txt" && ! twin_pids >/dev/null; then
        record "reqs sim run --headless" ok "отказ до запуска, двойник не поднят"
    else
        record "reqs sim run --headless" отказ "код $code: $(tail -1 "$out-headless.txt")"
    fi
}

# One run of the task against a twin that is up, judged by the witness.
drive() {  # drive <label> <dir>
    local label="$1" dir="$2" code
    clean "timeout 40 '$OWN_PY' '$PY_DIR/witness.py' $DDS_DOMAIN 2 '$dir/before.json'" > "$dir/before.txt" 2>&1
    cat "$dir/before.txt"
    if [[ ! -s "$dir/before.json" ]]; then
        record "$label состояние" отказ "$(tail -1 "$dir/before.txt")"
        return 1
    fi
    record "$label состояние" ok "$(tail -1 "$dir/before.txt")"

    clean "cd '$REPO_ROOT' && timeout 120 '$OWN_CLI' run '$SCENARIO' -R '$REGISTRY'" > "$dir/run.txt" 2>&1
    code=$?
    cat "$dir/run.txt"
    if (( code == 0 )); then
        record "$label навык" ok "$(grep -m1 '^выполнено' "$dir/run.txt")"
    else
        record "$label навык" "отказ, код $code" "$(grep -E '^(отказ|причина|g1:)' "$dir/run.txt" | tr '\n' ' ' | grep . || tail -1 "$dir/run.txt")"
    fi

    clean "timeout 40 '$OWN_PY' '$PY_DIR/witness.py' $DDS_DOMAIN 2 '$dir/after.json'" > "$dir/after.txt" 2>&1
    cat "$dir/after.txt"
    if clean "cd '$REPO_ROOT' && '$OWN_PY' '$PY_DIR/moved.py' '$dir/before.json' '$dir/after.json' '$SCENARIO' '$PROFILE'" > "$dir/moved.txt" 2>&1; then
        record "$label движение" ok "$(tail -1 "$dir/moved.txt")"
    else
        record "$label движение" отказ "$(tail -1 "$dir/moved.txt")"
    fi
}

step_link() {
    local dir="$STAGE_ROOT/link" pid
    rm -rf "$dir"; mkdir -p "$dir"
    step "link — двойник руками, run из окружения адаптера (ADR-0020: сначала адаптер против готового двойника)"
    if [[ -z ${DISPLAY:-} ]]; then record link пропущено "DISPLAY не задан"; return; fi
    if twin_pids >/dev/null; then record link пропущено "unitree_mujoco уже запущен"; return; fi
    start_group "$dir/twin.log" "$TWIN_ENV && cd '$(dirname "$TWIN_BIN")' && exec unitree_mujoco -r g1 -s '$SCENE' -i $DDS_DOMAIN -n lo"
    pid=$LAST_GROUP
    if ! wait_for_line "$dir/twin.log" "Mujoco data is prepared" 60 "$pid"; then
        record "link двойник" отказ "за 60 с нет «Mujoco data is prepared»: $(tail -1 "$dir/twin.log")"
        tail -30 "$dir/twin.log" | sed 's/^/      /'
        stop_group "$pid" 10; return
    fi
    sleep 2
    record "link двойник" ok "процесс $pid"
    drive link "$dir"
    stop_group "$pid" 10
}

step_scene() {
    local dir="$STAGE_ROOT/scene" pgid twin planned started waited=0
    rm -rf "$dir"; mkdir -p "$dir"
    step "scene — sim run поднимает двойника, run ведёт робота, Ctrl+C останавливает"
    if [[ -z ${DISPLAY:-} ]]; then record scene пропущено "DISPLAY не задан"; return; fi
    if twin_pids >/dev/null; then record scene пропущено "unitree_mujoco уже запущен"; return; fi

    start_group "$dir/sim.log" "$TWIN_ENV && cd '$REPO_ROOT' && exec '$EXEC_CLI' sim run '$SCENARIO' --into '$dir/world' -R '$REGISTRY'"
    pgid=$LAST_GROUP
    if ! wait_for_line "$dir/sim.log" "Mujoco data is prepared" 60 "$pgid"; then
        record "scene sim run" отказ "за 60 с двойник не готов: $(grep -m1 -E 'не |ошибка' "$dir/sim.log" || tail -1 "$dir/sim.log")"
        tail -40 "$dir/sim.log" | sed 's/^/      /'
        stop_group "$pgid" 10; return
    fi
    sleep 2
    twin="$(group_of "$pgid")"
    started="$(tr '\0' ' ' < "/proc/$twin/cmdline" 2>/dev/null | sed 's/ $//')"
    planned="$(grep -m1 '^  unitree_mujoco' "$STAGE_ROOT/reqs-dry.txt" 2>/dev/null | sed 's/^ *//')"
    info "запущено (/proc/$twin/cmdline): $started"
    info "сухой прогон:                 ${planned:-нет — шаг reqs не выполнялся}"
    if [[ -n $twin && -n $planned && $started == "$planned" ]]; then
        record "scene sim run" ok "двойник $twin запущен той командой, что напечатал сухой прогон"
    else
        record "scene sim run" "расходится" "запущено: ${started:-—}; план: ${planned:-—}"
    fi

    drive scene "$dir"

    # Ctrl+C to sim run alone, not TERM to the group: the backend's stop is under test.
    kill -INT "$pgid" 2>/dev/null
    while running "$pgid" && (( waited < 30 )); do sleep 1; waited=$(( waited + 1 )); done
    if ! running "$pgid" && [[ -n $twin ]] && ! running "$twin"; then
        record "scene остановка" ok "sim run вышел за $waited с, двойника $twin нет"
    else
        record "scene остановка" отказ "через $waited с: sim run $(running "$pgid" && echo жив || echo вышел), двойник $(running "${twin:-0}" && echo жив || echo нет)"
        tail -15 "$dir/sim.log" | sed 's/^/      /'
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
            link)  step_link ;;
            scene) step_scene ;;
        esac
    done
    summary
}

main "$@"
