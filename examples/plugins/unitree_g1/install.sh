#!/usr/bin/env bash
#
# The environment of the G1 package on a Linux machine (ADR-0071). Run by the platform's
# installer, never directly:
#
#   scripts/install.sh install unitree_g1      # everything below, in order
#   scripts/install.sh verify unitree_g1
#
# The package states what it needs (ADR-0025) and builds it here, because the platform's
# installer must not know it (ADR-0022, ADR-0034). Everything lands in $STACK_ROOT/unitree_g1:
#
#   dds     CycloneDDS 0.10.2, the C library, built from source into a prefix of its own: the
#           robot and the vendor's twin run 0.10.2, and 0.11+ crashes them on discovery (ADR-0022)
#   venv    the adapter's venv (python3.12): the cyclonedds binding built against that library,
#           the vendor's SDK for IDL types and CRC, the platform and this package. No rclpy,
#           ever: the adapter's process must not load ROS 2's libddsc (ADR-0022, ADR-0029)
#   ws      ROS 2 workspace: unitree_hg messages and this package's bridge controller for
#           Gazebo (ADR-0033); rmw_cyclonedds_cpp, which the world runs on (ADR-0034)
#   twin    the vendor's MuJoCo twin unitree_mujoco at the commits of ADR-0023, and a launcher
#           on PATH that gives it the libraries it was built against
#
# Environments it adds (scripts/activate.sh):
#   env/exec.sh   the twin's launcher on PATH -- a MuJoCo world starts from the execution venv
#   env/ros.sh    the workspace overlay -- a Gazebo world with the bridge
#   venv/         `source scripts/activate.sh unitree_g1` -- where `run` for a G1 is executed

set -euo pipefail

: "${COMMON_SH:?установщик пакета запускается платформой: scripts/install.sh install unitree_g1}"
# shellcheck source=../../../scripts/lib/common.sh
source "$COMMON_SH"

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_NAME="$(basename "$PKG_DIR")"
PKG_ROOT="$STACK_ROOT/$PKG_NAME"

# Pins. Each is a decision recorded in an ADR; changing one is a new decision.
CDDS_URL="https://github.com/eclipse-cyclonedds/cyclonedds.git"
CDDS_REF="0.10.2"                                            # ADR-0022
SDK_PY_URL="https://github.com/unitreerobotics/unitree_sdk2_python.git"
SDK_PY_COMMIT="65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"      # read in session 5
UNITREE_ROS2_URL="https://github.com/unitreerobotics/unitree_ros2.git"
UNITREE_ROS2_COMMIT="668d1ec5a05d1c38d3306bdca7d59f2ba3581a88" # unitree_hg messages
SDK2_URL="https://github.com/unitreerobotics/unitree_sdk2.git"
SDK2_COMMIT="9754cd153af3da471b0fe5f3aa535e426fb11db3"        # ADR-0023
TWIN_URL="https://github.com/unitreerobotics/unitree_mujoco.git"
TWIN_COMMIT="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"        # ADR-0023
MUJOCO_VERSION="3.3.6"                                       # the version the twin's README names
MUJOCO_URL="https://github.com/google-deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz"

PYTHON="python3.12"                   # the system interpreter; the binding compiles against it

CDDS_DIR="$PKG_ROOT/cyclonedds-$CDDS_REF"
CDDS_PREFIX="$CDDS_DIR/install"
VENV="$PKG_ROOT/venv"
PY="$VENV/bin/python"
SDK_PY_DIR="$PKG_ROOT/unitree_sdk2_python"
UNITREE_ROS2_DIR="$PKG_ROOT/unitree_ros2"
UNITREE_HG_SRC="$UNITREE_ROS2_DIR/cyclonedds_ws/src/unitree/unitree_hg"
WS="$PKG_ROOT/ws"
SDK2_DIR="$PKG_ROOT/unitree_sdk2"
MUJOCO_DIR="$PKG_ROOT/mujoco-$MUJOCO_VERSION"
TWIN_DIR="$PKG_ROOT/unitree_mujoco"
TWIN_BIN="$TWIN_DIR/simulate/build/unitree_mujoco"
BIN_DIR="$PKG_ROOT/bin"
ENV_DIR="$PKG_ROOT/env"

PARTS="dds venv ws twin"

# -- state -------------------------------------------------------------------------------------

has_dds()  { [[ -f "$CDDS_PREFIX/lib/libddsc.so" || -f "$CDDS_PREFIX/lib64/libddsc.so" ]]; }
has_venv() { [[ -x "$PY" ]] && [[ "$(dist_version "$PY" cyclonedds)" == "$CDDS_REF" ]] &&
             "$PY" -c "import unitree_sdk2py.core.channel, $APP" >/dev/null 2>&1; }
has_ws()   { [[ -f "$WS/install/unitree_g1_bridge/share/unitree_g1_bridge/unitree_g1_bridge.xml" &&
                -f "$WS/install/setup.bash" ]]; }
has_twin() { [[ -x "$TWIN_BIN" && -x "$BIN_DIR/unitree_mujoco" ]]; }

commit_of() { git -C "$1" rev-parse --short HEAD 2>/dev/null || echo '—'; }

# -- parts -------------------------------------------------------------------------------------

install_dds() {
    step "dds: CycloneDDS $CDDS_REF из исходника в $CDDS_PREFIX"
    # bison builds the IDL compiler of the C library.
    apt_install git cmake build-essential bison libssl-dev
    if has_dds; then
        ok "библиотека уже собрана"
        return 0
    fi
    fetch_pinned "$CDDS_URL" "$CDDS_REF" "$CDDS_DIR/src"
    run_clean "cmake -S '$CDDS_DIR/src' -B '$CDDS_DIR/build' -DCMAKE_INSTALL_PREFIX='$CDDS_PREFIX' \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SHM=OFF \
        && cmake --build '$CDDS_DIR/build' --target install -j\$(nproc)"
    ok "CycloneDDS $CDDS_REF собрана"
}

# pip, not uv: the binding is built from source against $CDDS_PREFIX (no wheels for Python
# 3.11+ at 0.10.2), and --no-cache-dir keeps a wheel built against another library out.
install_venv() {
    step "venv: окружение адаптера $VENV"
    apt_install python3.12-venv python3-dev
    if (( ! DRY_RUN )); then   # a dry run has built nothing it could find
        has_dds || die "нет CycloneDDS $CDDS_REF" "Сначала часть dds:  scripts/install.sh install $PKG_NAME dds"
        has_cmd "$PYTHON" || die "нет $PYTHON" "Ubuntu 24.04 поставляет его сама: sudo apt install python3.12"
    fi

    [[ -x "$PY" ]] || run_clean "'$PYTHON' -m venv '$VENV'"
    if (( ! DRY_RUN )) && [[ "$(dist_version "$PY" cyclonedds)" == "$CDDS_REF" ]]; then
        ok "обвязка cyclonedds $CDDS_REF на месте"
    else
        run_clean "CYCLONEDDS_HOME='$CDDS_PREFIX' '$PY' -m pip install --no-cache-dir 'cyclonedds==$CDDS_REF'"
    fi

    # The SDK's own requirements pull opencv and a cyclonedds pin it cannot build: only the
    # package, the rest is here (ADR-0029: IDL types and CRC are all the adapter takes from it).
    fetch_pinned "$SDK_PY_URL" "$SDK_PY_COMMIT" "$SDK_PY_DIR"
    run_clean "'$PY' -m pip install -q numpy && '$PY' -m pip install -q --no-deps -e '$SDK_PY_DIR'"
    sync_venv
}

# The platform with the groups `run` uses in this process -- records and learned skills --
# and this package. Reinstalled on every sync: entry points are written at install time.
sync_venv() {
    [[ -x "$PY" ]] || return 0
    info "платформа (группы record, onnx) и пакет $PKG_NAME в $VENV"
    run_clean "'$PY' -m pip install -q -e '$REPO_ROOT[record,onnx]' && '$PY' -m pip install -q --no-deps -e '$PKG_DIR'"
    (( DRY_RUN )) && return 0
    run_clean "'$PY' -c 'import sys, $APP.cli.main; sys.exit(\"rclpy\" in sys.modules)'" || die \
        "командная строка платформы загружает rclpy в окружении адаптера G1" \
        "Процесс адаптера не должен грузить libddsc ROS 2 (ADR-0022). Проверьте, не подключён ли" \
        "ROS 2 в ~/.bashrc, и что $VENV создан без --system-site-packages."
    ok "окружение адаптера: cyclonedds $(dist_version "$PY" cyclonedds), без rclpy"
}

install_ws() {
    step "ws: сообщения unitree_hg и мост unitree_g1_bridge в $WS"
    [[ -f "$ROS_SETUP" ]] || die "нет ROS 2 $ROS_DISTRO_NAME" "Сначала:  scripts/install.sh install ros2 gazebo"
    # rmw_cyclonedds_cpp: the world runs on CycloneDDS as the robot does (ADR-0034, p. 7 --
    # declared by the twin, installed by the package). The generator builds the messages' DDS IDL.
    apt_install "ros-$ROS_DISTRO_NAME-rmw-cyclonedds-cpp" "ros-$ROS_DISTRO_NAME-rosidl-generator-dds-idl"
    fetch_pinned "$UNITREE_ROS2_URL" "$UNITREE_ROS2_COMMIT" "$UNITREE_ROS2_DIR"
    run mkdir -p "$WS"
    # Rebuilt on every install: colcon skips what has not changed, and the bridge's sources
    # live in the repository and change with it.
    run_clean "source '$ROS_SETUP' && cd '$WS' && colcon build \
        --base-paths '$UNITREE_HG_SRC' '$PKG_DIR/ros2' \
        --packages-select unitree_hg unitree_g1_bridge"
    ok "рабочее пространство собрано"
}

install_twin() {
    step "twin: двойник производителя unitree_mujoco"
    # libboost-program-options-dev, not the README's libboost-all-dev: the simulator links only
    # program_options, and the SDK's headers include no boost, Eigen or spdlog (phase 4, stage 1).
    apt_install build-essential cmake git curl libyaml-cpp-dev \
                libboost-program-options-dev libglfw3-dev libfmt-dev

    if [[ -f "$SDK2_DIR/install/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake" ]]; then
        ok "unitree_sdk2 уже собран"
    else
        fetch_pinned "$SDK2_URL" "$SDK2_COMMIT" "$SDK2_DIR/src"
        # Into a prefix of its own, not the README's /opt/unitree_robotics: no sudo, and nothing
        # a later build could find by accident.
        run_clean "cmake -S '$SDK2_DIR/src' -B '$SDK2_DIR/build' -DCMAKE_INSTALL_PREFIX='$SDK2_DIR/install' \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF \
            && cmake --build '$SDK2_DIR/build' --target install -j\$(nproc)"
    fi

    if [[ -f "$MUJOCO_DIR/lib/libmujoco.so" ]]; then
        ok "MuJoCo $MUJOCO_VERSION на месте"
    else
        run_clean "curl -fsSL --retry 3 -o '$PKG_ROOT/mujoco.tar.gz' '$MUJOCO_URL' \
            && tar -xzf '$PKG_ROOT/mujoco.tar.gz' -C '$PKG_ROOT'"
        (( DRY_RUN )) || info "sha256 архива: $(sha256sum "$PKG_ROOT/mujoco.tar.gz" | cut -d' ' -f1)"
    fi

    if [[ -x "$TWIN_BIN" ]]; then
        ok "unitree_mujoco уже собран"
    else
        fetch_pinned "$TWIN_URL" "$TWIN_COMMIT" "$TWIN_DIR"
        # The symlink is the vendor README's own step, not an edit of the twin (ADR-0023).
        run ln -sfn "$MUJOCO_DIR" "$TWIN_DIR/simulate/mujoco"
        run_clean "cmake -S '$TWIN_DIR/simulate' -B '$TWIN_DIR/simulate/build' -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH='$SDK2_DIR/install' \
            && cmake --build '$TWIN_DIR/simulate/build' --target unitree_mujoco -j\$(nproc)"
    fi

    # The platform starts the twin by its name on PATH (ADR-0025). The binary needs the SDK's
    # CycloneDDS and MuJoCo from its own build, and a shell that has sourced ROS 2 carries
    # Jazzy's libddsc on LD_LIBRARY_PATH: the launcher puts the twin's first, for this process
    # only, and replaces itself with the binary so the world's signals reach it (ADR-0037).
    run mkdir -p "$BIN_DIR"
    write_file "$BIN_DIR/unitree_mujoco" "#!/bin/sh
# Written by $PKG_NAME/install.sh: the vendor's twin with the libraries it was built against.
LD_LIBRARY_PATH='$SDK2_DIR/install/lib:$MUJOCO_DIR/lib'\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH
exec '$TWIN_BIN' \"\$@\""
    run chmod +x "$BIN_DIR/unitree_mujoco"
    ok "двойник: $BIN_DIR/unitree_mujoco"
}

write_file() {  # write_file <path> <content>
    if (( DRY_RUN )); then
        info "[dry-run] запись $1"
        return 0
    fi
    printf '%s\n' "$2" > "$1"
}

# The pieces scripts/activate.sh adds to the platform's environments. Each checks at source
# time what exists, so a partial installation gives a partial, not a broken, environment.
write_env() {
    run mkdir -p "$ENV_DIR"
    write_file "$ENV_DIR/exec.sh" "# Written by $PKG_NAME/install.sh; sourced by scripts/activate.sh exec|ros.
if [ -x '$BIN_DIR/unitree_mujoco' ]; then
    PATH='$BIN_DIR':\$PATH
fi"
    write_file "$ENV_DIR/ros.sh" "# Written by $PKG_NAME/install.sh; sourced by scripts/activate.sh ros.
if [ -f '$WS/install/setup.bash' ]; then
    . '$WS/install/setup.bash'
fi"
}

# -- commands ----------------------------------------------------------------------------------

cmd_install() {
    local parts=("$@") part
    (( ${#parts[@]} )) || read -r -a parts <<< "$PARTS"
    run mkdir -p "$PKG_ROOT"
    for part in "${parts[@]}"; do
        case "$part" in
            dds)  install_dds ;;
            venv) install_venv ;;
            ws)   install_ws ;;
            twin) install_twin ;;
            *)    die "неизвестная часть пакета $PKG_NAME: $part" "Доступны: $PARTS" ;;
        esac
    done
    write_env
}

cmd_verify() {
    local failed=0 out
    if has_venv; then
        out="$(run_clean "'$PY' -c 'import sys, cyclonedds, unitree_sdk2py.core.channel, $APP.cli.main; print(\"rclpy\" in sys.modules)'" 2>&1)"
        if [[ "$out" == "False" ]]; then
            ok "окружение адаптера: cyclonedds $(dist_version "$PY" cyclonedds), SDK, платформа; rclpy не загружен"
        else
            warn "окружение адаптера: $out"; failed=1
        fi
        info "адаптеры: $("$PY" -c "from importlib.metadata import entry_points; print(' '.join(sorted(e.name for e in entry_points(group='$APP.adapters'))))")"
    else
        warn "нет окружения адаптера (часть venv)"; failed=1
    fi

    if has_ws && run_clean "source '$WS/install/setup.bash' && ros2 pkg prefix unitree_g1_bridge && ros2 interface show unitree_hg/msg/LowCmd >/dev/null" >/dev/null 2>&1; then
        ok "мост и сообщения видны ROS 2: $WS"
    else
        warn "мост unitree_g1_bridge или unitree_hg не видны ROS 2 (часть ws)"; failed=1
    fi

    if has_twin; then
        if out="$(LD_LIBRARY_PATH="$SDK2_DIR/install/lib:$MUJOCO_DIR/lib" ldd "$TWIN_BIN" | grep 'not found')"; then
            warn "двойнику не хватает библиотек: $out"; failed=1
        else
            ok "двойник собран, библиотеки находятся: $BIN_DIR/unitree_mujoco"
        fi
        info "окна у двойника нет без графической сессии: sim run MuJoCo — из сессии с DISPLAY (ADR-0023)"
    else
        warn "нет двойника unitree_mujoco (часть twin)"; failed=1
    fi
    return "$failed"
}

# Rows "part|state|where" for scripts/install.sh status and report; exit 1 if a part is missing.
cmd_status() {
    local missing=0
    if has_dds; then echo "dds|есть|$CDDS_PREFIX ($CDDS_REF)"; else echo "dds|нет|$CDDS_PREFIX"; missing=1; fi
    if has_venv; then
        echo "venv|есть|$VENV, cyclonedds $(dist_version "$PY" cyclonedds), unitree_sdk2_python $(commit_of "$SDK_PY_DIR")"
    else
        echo "venv|нет|$VENV"; missing=1
    fi
    if has_ws; then echo "ws|есть|$WS, unitree_ros2 $(commit_of "$UNITREE_ROS2_DIR")"; else echo "ws|нет|$WS"; missing=1; fi
    if has_twin; then
        echo "twin|есть|unitree_mujoco $(commit_of "$TWIN_DIR"), unitree_sdk2 $(commit_of "$SDK2_DIR/src"), MuJoCo $MUJOCO_VERSION"
    else
        echo "twin|нет|$TWIN_BIN"; missing=1
    fi
    return "$missing"
}

case "${1:-}" in
    install) shift; cmd_install "$@" ;;
    sync)    sync_venv ;;
    verify)  cmd_verify ;;
    status)  cmd_status ;;
    *)       die "команда установщика пакета: install [часть...] | sync | verify | status" "Части: $PARTS" ;;
esac
