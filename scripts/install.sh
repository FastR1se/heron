#!/usr/bin/env bash
#
# The one installer of the Linux test machine (ADR-0071; replaces scripts/phase0.sh).
#
# Platform components, installed here:
#   base         apt basics, uv, ffmpeg
#   ros2         ROS 2 Jazzy desktop and the control stack the generators name (ADR-0013)
#   gazebo       Gazebo Harmonic and its bridge to ROS 2
#   exec         .venv of the repository: the platform with every optional group it runs with,
#                and every package of examples/plugins; Python of ROS 2's own version
#   descriptions vendor descriptions of the example robots (examples/robots/*/fetch_description.py)
#   train        training stack in $TRAIN_ROOT: Isaac Sim, Isaac Lab, rsl_rl, the platform and the
#                packages (ADR-0005, ADR-0044); a venv of its own -- ROS 2 never enters it
#   docker       Docker and the NVIDIA container toolkit
#
# Robot packages: every examples/plugins/<package>/install.sh. A package states and builds its
# own environment -- vendor SDKs, a ROS 2 workspace, a twin -- in $STACK_ROOT/<package>
# (ADR-0022, ADR-0034): this script knows no robot, it finds the installers and runs them.
#
# Two virtual environments of the platform, not one: the training stack never shares an
# interpreter with the execution stack (invariant 2, ADR-0003). The script never repairs the
# system silently: a violated precondition stops it and names the command for a human.
#
# Usage:  scripts/install.sh help

set -euo pipefail

# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

# ---------------------------------------------------------------------------------------------
# Constants. Versions live here and nowhere else in the file.
# ---------------------------------------------------------------------------------------------

REQUIRED_UBUNTU="24.04"                   # ADR-0005; ROS 2 Jazzy is tied to noble
GZ_DISTRO="harmonic"

TRAIN_PYTHON="3.11"                       # Isaac Sim wheels; Ubuntu 24.04 ships 3.12
ISAACSIM_SPEC="isaacsim[all,extscache]==5.1.0"
ISAACLAB_REF="v2.3.0"
TORCH_SPEC="torch==2.7.0 torchvision==0.22.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
FLATDICT_SPEC="flatdict==4.0.1"           # the version Isaac Lab pins; see ensure_flatdict

# Optional groups of pyproject.toml the execution venv carries: MuJoCo, tests, what ROS 2
# messages import, records, learned skills.
EXEC_EXTRAS="sim,dev,ros2,record,onnx"

PLATFORM_STEPS="base ros2 gazebo exec descriptions train docker"
DEFAULT_PLATFORM_STEPS="base ros2 gazebo exec descriptions"

# Python packages the ros2_control adapter imports when it connects. Mirrors ADAPTER_PACKAGES
# in src/<name>/ros2/names.py (a test holds the two together), and verify_exec checks that the
# execution environment can actually import them (ADR-0014).
ADAPTER_IMPORTS="rclpy sensor_msgs std_msgs trajectory_msgs builtin_interfaces action_msgs control_msgs"

# Thresholds from ADR-0005. Changing a number here is a decision, not a tweak.
MIN_SWAP_GIB=16                           # 15.3 GiB RAM: swap is what keeps Isaac alive
MIN_VRAM_MIB=7000                         # peak VRAM budget is 7 GiB out of 8

ACCEPT_NVIDIA_EULA=0
PACKAGES="$(package_installers | tr '\n' ' ')"
PACKAGES="${PACKAGES% }"

# ---------------------------------------------------------------------------------------------
# Components: names are validated before any work starts. A typo found after an hour of
# downloads is the same defect as a contract mismatch found on the robot.
# ---------------------------------------------------------------------------------------------

is_package() { [[ " $PACKAGES " == *" $1 "* ]]; }

# expand_components <arg>...: the named components in a fixed order, "all" and nothing expanded.
expand_components() {
    local wanted=" $* " out=() name
    if [[ $# -eq 0 ]]; then
        wanted=" $DEFAULT_PLATFORM_STEPS $PACKAGES "
    elif [[ "$wanted" == *" all "* ]]; then
        wanted=" $PLATFORM_STEPS $PACKAGES "
    fi
    for name in $wanted; do
        [[ " $PLATFORM_STEPS $PACKAGES all " == *" $name "* ]] && continue
        if [[ "$name" == *=* ]]; then
            die "неизвестный компонент: $name" \
                "Похоже на переменную окружения. Она ставится перед командой, а не после:" \
                "    ${name} scripts/install.sh ..." \
                "Доступны: $PLATFORM_STEPS $PACKAGES all"
        fi
        die "неизвестный компонент: $name" "Доступны: $PLATFORM_STEPS $PACKAGES all"
    done
    for name in $PLATFORM_STEPS $PACKAGES; do
        [[ "$wanted" == *" $name "* ]] && out+=("$name")
    done
    printf '%s\n' "${out[@]}"
}

# The installer of a package gets the shared library and the settings of this run; everything
# else it derives itself (ADR-0071). Exit 3 means "skipped: an input only a human can provide
# is missing", which stops nothing else.
package_cmd() {  # package_cmd <package> <command> [args...]
    local pkg="$1"
    shift
    COMMON_SH="$REPO_ROOT/scripts/lib/common.sh" DRY_RUN="$DRY_RUN" STACK_ROOT="$STACK_ROOT" \
        TRAIN_ROOT="$TRAIN_ROOT" bash "$PLUGINS_DIR/$pkg/install.sh" "$@"
}

gib() { awk -v kb="$1" 'BEGIN { printf "%.1f", kb / 1048576 }'; }

# ---------------------------------------------------------------------------------------------
# Preconditions. Everything here is measured, nothing is repaired.
# ---------------------------------------------------------------------------------------------

check_os() {
    [[ -r /etc/os-release ]] || die "не читается /etc/os-release — система не определяется."
    # shellcheck disable=SC1091
    . /etc/os-release
    [[ "${ID:-}" == "ubuntu" ]] || die \
        "нужна Ubuntu, обнаружено: ${PRETTY_NAME:-неизвестно}" \
        "ADR-0005 фиксирует родную Ubuntu $REQUIRED_UBUNTU; другой дистрибутив — новое решение, а не правка скрипта."
    [[ "${VERSION_ID:-}" == "$REQUIRED_UBUNTU" ]] || die \
        "нужна Ubuntu $REQUIRED_UBUNTU, обнаружено: ${VERSION_ID:-неизвестно}" \
        "ROS 2 $ROS_DISTRO_NAME собирается только под noble; смена версии ОС тянет смену дистрибутива ROS."
    ok "ОС: ${PRETTY_NAME:-ubuntu $REQUIRED_UBUNTU}"
}

check_arch() {
    local arch
    arch="$(uname -m)"
    [[ "$arch" == "x86_64" ]] || die \
        "архитектура $arch не поддерживается" \
        "Isaac Sim собран под x86_64; сборки под ARM отложены (раздел 15)."
    ok "архитектура: $arch, ядро $(uname -r)"
}

check_user() {  # check_user <needs sudo: 0|1>
    [[ "$(id -u)" -ne 0 ]] || die \
        "скрипт запущен от root" \
        "venv, кэши и группа docker должны принадлежать рабочему пользователю." \
        "Запускайте от своего пользователя: sudo вызывается точечно там, где он нужен."
    if (( $1 )) && ! sudo -n true 2>/dev/null; then
        info "нужен sudo, пароль спрашивается один раз"
        sudo -v || die "sudo недоступен — установка системных пакетов невозможна."
    fi
    ok "пользователь: $(id -un)$( (( $1 )) && echo ', sudo доступен')"
}

check_memory() {
    local ram_kb swap_kb swap_gib
    ram_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    swap_kb="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_gib="$(gib "$swap_kb")"
    info "RAM: $(gib "$ram_kb") ГиБ, swap: ${swap_gib} ГиБ"

    # Compared in KiB: the printed value goes through awk and carries the locale decimal
    # separator, which turns a float comparison into a lie.
    if (( swap_kb < MIN_SWAP_GIB * 1048576 )); then
        die "swap ${swap_gib} ГиБ, по ADR-0005 для обучения нужно не меньше ${MIN_SWAP_GIB} ГиБ" \
            "При 15.3 ГиБ RAM Isaac Sim уходит в OOM без подкачки — ровно та нехватка памяти," \
            "из-за которой отвергнут WSL2. Достаточно файла подкачки на корне:" \
            "    sudo fallocate -l ${MIN_SWAP_GIB}G /swapfile && sudo chmod 600 /swapfile" \
            "    sudo mkswap /swapfile && sudo swapon /swapfile" \
            "    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab" \
            "Скрипт этого не делает сам: разметка памяти — решение человека."
    fi
    ok "памяти достаточно по порогу ADR-0005"
}

# Rough disk cost of a component in GiB, from the ADR-0005 estimate.
component_cost_gib() {
    case "$1" in
        base)         echo 1  ;;
        ros2)         echo 4  ;;
        gazebo)       echo 2  ;;
        exec)         echo 2  ;;
        descriptions) echo 1  ;;
        train)        echo 40 ;;   # Isaac Sim ~20 plus asset cache 10-20
        docker)       echo 10 ;;
        *)            echo 2  ;;   # a robot package: vendor SDKs, a workspace, a twin
    esac
}

component_installed() {
    case "$1" in
        base)         has_cmd uv ;;
        ros2)         [[ -d "/opt/ros/$ROS_DISTRO_NAME" ]] ;;
        gazebo)       has_cmd gz ;;
        exec)         [[ -x "$EXEC_CLI" ]] && "$EXEC_PY" -c 'import mujoco' >/dev/null 2>&1 ;;
        descriptions) descriptions_fetched ;;
        train)        [[ -x "$TRAIN_PY" ]] &&
                      "$TRAIN_PY" -m pip show isaacsim-rl >/dev/null 2>&1 ;;
        docker)       has_cmd docker ;;
        *)            package_cmd "$1" status >/dev/null 2>&1 ;;
    esac
}

# The question is "is there room for what we are about to install", not "is this a fresh
# machine": an installed component occupies its space and asks for no more.
check_disk() {
    local c need=0 pending=() free_kb
    for c in "$@"; do
        component_installed "$c" && continue
        pending+=("$c")
        need=$(( need + $(component_cost_gib "$c") ))
    done
    free_kb="$(df -Pk "$HOME" | awk 'NR == 2 { print $4 }')"   # KiB: no suffix, no locale

    if (( ${#pending[@]} == 0 )); then
        ok "свободно в $HOME: $(gib "$free_kb") ГиБ; ставить нечего, всё уже на месте"
        return 0
    fi
    info "свободно в $HOME: $(gib "$free_kb") ГиБ; предстоит: ${pending[*]}, оценка ${need} ГиБ"
    if (( free_kb < need * 1048576 )); then
        die "свободно $(gib "$free_kb") ГиБ, на ${pending[*]} нужно около ${need} ГиБ" \
            "Оценка ADR-0005: Isaac Sim с кэшем ассетов 30-40, ROS 2 desktop 4, Gazebo 2, Docker 10." \
            "Ставьте компоненты по одному или освободите место."
    fi
    ok "места достаточно"
}

check_gpu() {
    has_cmd nvidia-smi || die \
        "nvidia-smi не найден — драйвер NVIDIA не установлен" \
        "Установите проприетарный драйвер и перезагрузитесь:" \
        "    sudo ubuntu-drivers install" \
        "Драйвер ставится отдельно: он требует перезагрузки, а его версия входит в условия" \
        "каждого эксперимента (раздел 13)."
    local name driver vram
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    info "GPU: ${name}, драйвер ${driver}, VRAM ${vram} МиБ"
    if (( vram < MIN_VRAM_MIB )); then
        warn "VRAM ${vram} МиБ ниже бюджета ADR-0005 (${MIN_VRAM_MIB} МиБ пикового потребления)"
        warn "порог go/no-go проверяется измерительным прогоном, а не этим скриптом"
    fi
    ok "GPU виден драйверу"
}

check_vulkan() {
    if ! has_cmd vulkaninfo; then
        warn "vulkaninfo не найден, проверка отложена до компонента base (пакет vulkan-tools)"
        return 0
    fi
    if vulkaninfo --summary 2>/dev/null | grep -qi 'devicename.*nvidia'; then
        ok "Vulkan видит устройство NVIDIA"
    else
        die "Vulkan не видит NVIDIA" \
            "Isaac Sim отказывает на этом шаге чаще, чем из-за нехватки VRAM." \
            "Смотрите вывод:  vulkaninfo --summary" \
            "На гибридной графике помогает явный выбор устройства:" \
            "    export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia"
    fi
}

check_network() {
    if curl -fsS --max-time 10 -o /dev/null https://packages.ros.org; then
        ok "сеть доступна"
    else
        die "нет доступа к packages.ros.org — установка невозможна."
    fi
}

# What is measured depends on what is about to be installed: the swap of ADR-0005 guards
# Isaac, not a ROS 2 package, so it stops only a run that installs the training stack.
cmd_check() {
    local comps=("$@") c sudo_needed=0 train=0
    for c in "${comps[@]}"; do
        case "$c" in
            exec|descriptions) ;;
            train) train=1 ;;
            *)     sudo_needed=1 ;;
        esac
    done
    step "Предусловия: ${comps[*]}"
    check_os
    check_arch
    check_user "$sudo_needed"
    check_disk "${comps[@]}"
    if (( train )); then
        check_memory
        check_gpu
        check_vulkan
    fi
    check_network
    printf '\n    %sпредусловия выполнены%s\n' "$C_GREEN$C_BOLD" "$C_OFF"
}

# ---------------------------------------------------------------------------------------------
# Platform components. Every one is idempotent: a second run changes nothing.
# ---------------------------------------------------------------------------------------------

install_base() {
    step "base: системные пакеты и uv"
    apt_install curl wget git build-essential cmake pkg-config \
                python3-venv python3-pip python3-dev \
                software-properties-common ca-certificates gnupg lsb-release \
                vulkan-tools libvulkan1 mesa-utils ffmpeg
    if has_cmd uv; then
        ok "uv уже установлен: $(uv --version)"
    else
        info "ставится uv"
        run_sh "curl -LsSf https://astral.sh/uv/install.sh | sh"
        export PATH="$HOME/.local/bin:$PATH"
        info "uv положен в ~/.local/bin — добавьте каталог в PATH, если его там нет"
    fi
}

install_ros2() {
    step "ros2: ROS 2 ${ROS_DISTRO_NAME}"
    if [[ -d "/opt/ros/$ROS_DISTRO_NAME" ]]; then
        ok "уже установлен: /opt/ros/$ROS_DISTRO_NAME"
    else
        run sudo add-apt-repository -y universe
        run_sh "sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(. /etc/os-release && echo \$UBUNTU_CODENAME) main\" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null"
        unset APT_UPDATED
        apt_install "ros-${ROS_DISTRO_NAME}-desktop" ros-dev-tools
    fi
    # The set equals LAUNCH_PACKAGES of the launch generator and TWIN_PACKAGES of the
    # ros2_control twin; tests hold it from both sides (ADR-0013). The MCAP storage of rosbag2
    # is named because ADR-0003 wants MCAP on execution.
    apt_install "ros-${ROS_DISTRO_NAME}-ros2-control" \
                "ros-${ROS_DISTRO_NAME}-ros2-controllers" \
                "ros-${ROS_DISTRO_NAME}-control-msgs" \
                "ros-${ROS_DISTRO_NAME}-xacro" \
                "ros-${ROS_DISTRO_NAME}-robot-state-publisher" \
                "ros-${ROS_DISTRO_NAME}-joint-state-publisher" \
                "ros-${ROS_DISTRO_NAME}-rosbag2-storage-mcap"
}

install_gazebo() {
    step "gazebo: Gazebo ${GZ_DISTRO}"
    if has_cmd gz; then
        ok "уже установлен: $(gz sim --version 2>/dev/null | head -1)"
    else
        run_sh "sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg"
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null"
        unset APT_UPDATED
        apt_install "gz-${GZ_DISTRO}"
    fi
    # The bridge is what makes Gazebo an execution world rather than a viewer.
    apt_install "ros-${ROS_DISTRO_NAME}-ros-gz" \
                "ros-${ROS_DISTRO_NAME}-gz-ros2-control"
}

# Python version of the ROS 2 distribution, read off the installation instead of written
# down. The execution venv has to match it exactly: rclpy is a compiled extension that
# arrives through PYTHONPATH when setup.bash is sourced, and an interpreter of another version
# finds the package and fails on its binary -- silently, as an import that does not resolve.
#
# Deliberately NOT $TRAIN_PYTHON: the training stack is pinned to 3.11 by the Isaac Sim wheels
# and must not share an interpreter with ROS 2 at all (invariant 2).
ros2_python_version() {
    local lib
    for lib in "/opt/ros/$ROS_DISTRO_NAME/lib/"python3.*; do
        [[ -d "$lib" ]] || continue
        basename "$lib" | sed 's/^python//'
        return 0
    done
    return 1
}

need_uv() {
    has_cmd uv || die "uv не найден" "Сначала компонент base:  scripts/install.sh install base"
}

# The platform and every package of examples/plugins, editable. `uv pip install` and not
# `uv sync`: a sync removes whatever pyproject.toml does not list, the robot packages among
# it -- which is how every run script of phase 4 came to reinstall them (ADR-0071).
install_platform_into() {  # install_platform_into <python> <extras or empty>
    local python="$1" extras="${2:+[$2]}" editables
    mapfile -t editables < <(plugin_editables)
    run uv pip install --python "$python" -e "$REPO_ROOT$extras" "${editables[@]}"
}

install_exec() {
    step "exec: окружение исполнения $EXEC_VENV"
    need_uv
    local exec_python have interpreter
    exec_python="$(ros2_python_version)" || die \
        "не видно Python дистрибутива ROS 2 в /opt/ros/$ROS_DISTRO_NAME/lib" \
        "Окружение исполнения обязано совпадать с ним по версии: rclpy — скомпилированное" \
        "расширение, и интерпретатор другой версии его не загрузит." \
        "Сначала компонент ros2:  scripts/install.sh install ros2"
    info "Python дистрибутива ROS 2: $exec_python"

    if [[ -d "$EXEC_VENV" ]]; then
        have="$(venv_python_version "$EXEC_VENV")"
        [[ "$have" == "$exec_python" ]] || die \
            "venv исполнения на Python ${have:-?}, а ROS 2 собран под $exec_python" \
            "Адаптер ros2_control не сможет импортировать rclpy. Пересоздайте окружение:" \
            "    rm -rf '$EXEC_VENV' && scripts/install.sh install exec"
        ok "venv уже есть (Python $have)"
    else
        # The system interpreter ROS 2 was built against, not a managed build of the same
        # version: the one rclpy's binary is known to load.
        interpreter="/usr/bin/python$exec_python"
        [[ -x "$interpreter" ]] || interpreter="$exec_python"
        run uv venv --python "$interpreter" "$EXEC_VENV"
    fi
    info "платформа с группами $EXEC_EXTRAS и пакеты examples/plugins"
    install_platform_into "$EXEC_PY" "$EXEC_EXTRAS"
}

descriptions_fetched() {
    local script
    for script in "$ROBOTS_DIR"/*/fetch_description.py; do
        [[ -f "$script" ]] || continue
        [[ -d "$(dirname "$script")/vendor" ]] || return 1
    done
}

# Vendor descriptions are inputs that are not ours: the repository keeps their source and
# commit, the files are fetched next to each profile (ADR-0017, .gitignore).
install_descriptions() {
    step "descriptions: описания производителей для примеров роботов"
    [[ -x "$EXEC_PY" ]] || die "нет окружения исполнения" "Сначала:  scripts/install.sh install exec"
    local script
    for script in "$ROBOTS_DIR"/*/fetch_description.py; do
        [[ -f "$script" ]] || continue
        info "${script#"$REPO_ROOT"/}"
        run_sh "cd '$REPO_ROOT' && '$EXEC_PY' '$script'"
    done
}

# isaaclab.sh installs its extensions with `python -m pip` and locates Isaac Sim through pip
# metadata; `uv venv` creates an environment without pip, so it is put there explicitly -- a
# requirement of isaaclab.sh, not a workaround for it.
ensure_pip() {
    if (( ! DRY_RUN )) && "$TRAIN_PY" -m pip --version >/dev/null 2>&1; then
        ok "pip в окружении обучения есть"
        return 0
    fi
    run uv pip install --python "$TRAIN_PY" pip setuptools wheel
}

# isaacsim-rl comes from the [all] extra and is exactly what isaaclab.sh looks for. Its
# absence stops the run here, with the reason named, rather than later as "Python executable
# not found at _isaac_sim/python.sh".
check_isaacsim_installed() {
    (( DRY_RUN )) && return 0
    "$TRAIN_PY" -m pip show isaacsim-rl >/dev/null 2>&1 && { ok "пакеты Isaac Sim на месте"; return 0; }
    die "в окружении обучения нет пакета isaacsim-rl" \
        "Isaac Lab ищет Isaac Sim именно по нему. Вероятная причина: набор '[all]' поставлен" \
        "не полностью. Что стоит на самом деле:" \
        "    $TRAIN_PY -m pip list | grep isaacsim" \
        "Повторная установка:" \
        "    uv pip install --python '$TRAIN_PY' --reinstall '$ISAACSIM_SPEC' --extra-index-url https://pypi.nvidia.com"
}

# isaaclab.sh installs its modules one by one and exits 0 even when one of them fails to
# build, so a missing package would surface much later as an import error.
check_isaaclab_installed() {
    (( DRY_RUN )) && return 0
    local pkg missing=()
    for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
        "$TRAIN_PY" -m pip show "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    (( ${#missing[@]} == 0 )) && { ok "пакеты Isaac Lab установлены"; return 0; }
    die "после установки не хватает пакетов Isaac Lab: ${missing[*]}" \
        "isaaclab.sh завершается успешно, даже если модуль не собрался. Соберите вручную:" \
        "    cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && ./isaaclab.sh --install rsl_rl"
}

# flatdict 4.0.1 has a setup.py that imports pkg_resources, which setuptools 81 and later no
# longer ship; in pip's isolated build the newest setuptools lands and the build fails -- and
# with it isaaclab, which pins that exact version. Building against the venv's own setuptools
# is the only way through; the upstream pin is not ours to move.
ensure_flatdict() {
    if (( ! DRY_RUN )) && "$TRAIN_PY" -m pip show flatdict >/dev/null 2>&1; then
        ok "flatdict на месте"
        return 0
    fi
    info "flatdict ставится без изоляции сборки: его setup.py требует pkg_resources"
    run uv pip install --python "$TRAIN_PY" --no-build-isolation "$FLATDICT_SPEC"
}

# Isaac Lab lifts shared packages above the pins of isaacsim-kernel. Only a real launch shows
# whether that matters, so this reports and does not stop.
report_pip_conflicts() {
    (( DRY_RUN )) && return 0
    local out
    if out="$("$TRAIN_PY" -m pip check 2>&1)"; then
        ok "конфликтов версий в окружении обучения нет"
        return 0
    fi
    warn "конфликты версий в окружении обучения:"
    printf '%s\n' "$out" | sed 's/^/        /' >&2
    warn "существенно ли это, показывает запуск:  scripts/install.sh verify train"
}

install_train() {
    step "train: окружение обучения $TRAIN_ROOT (Isaac Sim, Isaac Lab, rsl_rl)"
    need_uv
    (( ACCEPT_NVIDIA_EULA )) || die "не принята лицензия NVIDIA" \
        "Пакеты Isaac Sim ставятся только после согласия с лицензией NVIDIA, и скрипт не" \
        "принимает её за человека. Прочитайте условия и повторите с флагом:" \
        "    scripts/install.sh install train --accept-nvidia-eula"

    run mkdir -p "$TRAIN_ROOT"
    if [[ -d "$TRAIN_VENV" ]]; then
        ok "venv обучения уже есть: $TRAIN_VENV"
    else
        info "отдельный venv на Python ${TRAIN_PYTHON}: миры не делят интерпретатор (инвариант 2)"
        run uv venv --seed --python "$TRAIN_PYTHON" "$TRAIN_VENV"
    fi
    ensure_pip

    if (( ! DRY_RUN )) && "$TRAIN_PY" -m pip show isaacsim-rl >/dev/null 2>&1; then
        ok "Isaac Sim уже установлен: $(dist_version "$TRAIN_PY" isaacsim)"
    else
        info "torch под CUDA 12.8, несколько ГиБ"
        run_sh "uv pip install --python '$TRAIN_PY' $TORCH_SPEC --index-url $TORCH_INDEX"
        info "Isaac Sim: загрузка десятков ГиБ, шаг долгий"
        run uv pip install --python "$TRAIN_PY" "$ISAACSIM_SPEC" --extra-index-url https://pypi.nvidia.com
    fi
    check_isaacsim_installed

    fetch_pinned https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_REF" "$ISAACLAB_DIR"
    ensure_flatdict
    if (( ! DRY_RUN )) && "$TRAIN_PY" -m pip show isaaclab_rl >/dev/null 2>&1; then
        ok "Isaac Lab уже установлен: $(dist_version "$TRAIN_PY" isaaclab)"
    else
        info "Isaac Lab с rsl_rl: других RL-фреймворков в стеке нет (раздел 11)"
        run_sh "cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && ./isaaclab.sh --install rsl_rl"
    fi
    check_isaaclab_installed

    # The same code computes contract_id on both sides of the boundary, which is the whole
    # reason the platform is here (ADR-0044); tasks are found by their entry point. Fact 0.14
    # of phase 6: the core installs into this venv replacing nothing.
    info "платформа и пакеты examples/plugins"
    install_platform_into "$TRAIN_PY" ""
    report_pip_conflicts
    info "Первый запуск Isaac Sim компилирует шейдеры: десятки минут, выглядит как зависание."
}

install_docker() {
    step "docker: Docker и NVIDIA Container Toolkit"
    if has_cmd docker; then
        ok "уже установлен: $(docker --version)"
    else
        run sudo install -m 0755 -d /etc/apt/keyrings
        run_sh "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg"
        run sudo chmod a+r /etc/apt/keyrings/docker.gpg
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable\" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null"
        unset APT_UPDATED
        apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi
    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        ok "nvidia-container-toolkit уже установлен"
    else
        run_sh "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run_sh "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
        unset APT_UPDATED
        apt_install nvidia-container-toolkit
        run sudo nvidia-ctk runtime configure --runtime=docker
        run sudo systemctl restart docker
    fi
    if id -nG "$(id -un)" | grep -qw docker; then
        ok "пользователь уже в группе docker"
    else
        run sudo usermod -aG docker "$(id -un)"
        warn "группа docker добавлена — нужен перелогин, иначе docker потребует sudo"
    fi
}

install_package() {
    local pkg="$1" code=0
    step "$pkg: окружение пакета робота в $STACK_ROOT/$pkg"
    package_cmd "$pkg" install || code=$?
    case $code in
        0) ;;
        3) SKIPPED+=("$pkg") ;;
        *) die "установщик пакета $pkg завершился с кодом $code" \
               "Причина — в выводе выше; установщик: examples/plugins/$pkg/install.sh" ;;
    esac
}

cmd_install() {
    local c
    SKIPPED=()
    for c in "$@"; do
        case "$c" in
            base)         install_base ;;
            ros2)         install_ros2 ;;
            gazebo)       install_gazebo ;;
            exec)         install_exec ;;
            descriptions) install_descriptions ;;
            train)        install_train ;;
            docker)       install_docker ;;
            *)            install_package "$c" ;;
        esac
    done
    if (( ${#SKIPPED[@]} )); then
        warn "пропущены пакеты без внешнего входа: ${SKIPPED[*]} (причина — в их выводе выше)"
    fi
    printf '\n    %sустановлено: %s%s\n' "$C_GREEN$C_BOLD" "$*" "$C_OFF"
    info "Проверка:   scripts/install.sh verify $*"
    info "Окружения:  source scripts/activate.sh <exec|ros|train|пакет>"
}

# Editable installs are links, but entry points are written at install time: a package that
# gained an entry point is invisible until it is installed again. This refreshes every
# environment that exists and installs nothing new -- no sudo, no builds.
cmd_sync() {  # cmd_sync [exec|train|<package>...]; without arguments every one that exists
    local targets=("$@") c
    (( ${#targets[@]} )) || read -r -a targets <<< "exec train $PACKAGES"
    step "sync: платформа и пакеты заново в окружениях: ${targets[*]}"
    for c in "${targets[@]}"; do
        case "$c" in
            exec)
                if [[ -x "$EXEC_PY" ]]; then
                    need_uv
                    install_platform_into "$EXEC_PY" "$EXEC_EXTRAS"
                    ok "окружение исполнения"
                else
                    warn "нет окружения исполнения: scripts/install.sh install exec"
                fi ;;
            train)
                if [[ -x "$TRAIN_PY" ]]; then
                    need_uv
                    install_platform_into "$TRAIN_PY" ""
                    ok "окружение обучения"
                elif (( $# )); then
                    warn "нет окружения обучения: scripts/install.sh install train --accept-nvidia-eula"
                fi ;;
            *)
                is_package "$c" || die "неизвестное окружение: $c" "Доступны: exec train $PACKAGES"
                package_cmd "$c" sync ;;
        esac
    done
}

# ---------------------------------------------------------------------------------------------
# Verification: every environment actually starts.
# ---------------------------------------------------------------------------------------------

verify_ros2() {
    step "Проверка ROS 2"
    [[ -f "$ROS_SETUP" ]] || die "ROS 2 $ROS_DISTRO_NAME не установлен:  scripts/install.sh install ros2"
    run_clean "source '$ROS_SETUP' && echo \"    пакетов доступно: \$(ros2 pkg list | wc -l)\""
    ok "ROS 2 $ROS_DISTRO_NAME отвечает"
}

verify_gazebo() {
    step "Проверка Gazebo, без окна, 200 шагов"
    has_cmd gz || die "gz не найден:  scripts/install.sh install gazebo"
    run_clean "gz sim -s -r --iterations 200 empty.sdf >/dev/null 2>&1"
    ok "сервер Gazebo отработал 200 шагов и завершился"
}

# The other side of ADAPTER_PACKAGES: a test in the repository checks that every declared
# package is installed by this script; only the machine can check that they actually import.
verify_adapter_imports() {
    local missing=() name
    for name in $ADAPTER_IMPORTS; do
        run_clean "source '$ROS_SETUP' && '$EXEC_PY' -c 'import $name'" >/dev/null 2>&1 ||
            missing+=("$name")
    done
    (( ${#missing[@]} == 0 )) || die "адаптер ros2_control не подключится: не импортируется ${missing[*]}" \
        "Окружение исполнения и ROS 2 должны совпадать по версии Python:" \
        "    $(venv_python_version "$EXEC_VENV")  — venv исполнения" \
        "    $(ros2_python_version)  — ROS 2 $ROS_DISTRO_NAME" \
        "Если версии разные:  rm -rf '$EXEC_VENV' && scripts/install.sh install exec"
    ok "адаптер ros2_control видит ROS 2: $ADAPTER_IMPORTS"
}

verify_exec() {
    step "Проверка окружения исполнения"
    [[ -x "$EXEC_CLI" ]] || die "окружение исполнения не создано:  scripts/install.sh install exec"
    (( DRY_RUN )) && { info "[dry-run] импорты адаптера, шаг MuJoCo, точки входа"; return 0; }
    verify_adapter_imports
    "$EXEC_PY" - <<'PY'
import mujoco

XML = """<mujoco><worldbody><body pos="0 0 1"><joint type="free"/>
<geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"    MuJoCo {mujoco.__version__}: 100 шагов, высота {data.qpos[2]:.4f} м")
PY
    ok "MuJoCo считает физику"
    info "$("$EXEC_CLI" version)"
    local group
    for group in adapters twins train_tasks; do
        info "точки входа $group: $("$EXEC_PY" -c "from importlib.metadata import entry_points; print(' '.join(sorted(e.name for e in entry_points(group='$APP.$group'))) or 'нет')")"
    done
    ok "платформа и пакеты роботов в окружении исполнения"
}

verify_train() {
    step "Проверка окружения обучения"
    [[ -x "$TRAIN_PY" ]] || die "окружение обучения не создано" \
        "Выполните:  scripts/install.sh install train --accept-nvidia-eula"
    if run_clean "'$TRAIN_PY' -c 'import rclpy'" >/dev/null 2>&1; then
        die "rclpy импортируется в окружении обучения" \
            "ROS 2 в окружение обучения не попадает (инвариант 2, ADR-0003); запрет объявлен бэкендом" \
            "isaaclab (ADR-0044). Проверьте, не подключён ли ROS 2 в профиле оболочки (~/.bashrc)."
    fi
    ok "rclpy не импортируется (инвариант 2)"
    info "задачи обучения: $(run_clean "'$TRAIN_CLI' task list" 2>/dev/null | tr '\n' ' ')"
    info "Isaac Sim, пустая сцена без окна; первый запуск компилирует шейдеры десятки минут"
    run_clean "cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless"
    ok "Isaac Sim поднял пустую сцену и завершился"
}

verify_docker() {
    step "Проверка Docker"
    has_cmd docker || die "docker не найден:  scripts/install.sh install docker"
    run docker info --format '    сервер {{.ServerVersion}}, среды исполнения: {{range $k, $v := .Runtimes}}{{$k}} {{end}}'
    ok "Docker отвечает"
}

cmd_verify() {
    local c code
    for c in "$@"; do
        case "$c" in
            base)         has_cmd uv || die "uv не найден:  scripts/install.sh install base"
                          ok "uv: $(uv --version)" ;;
            ros2)         verify_ros2 ;;
            gazebo)       verify_gazebo ;;
            exec)         verify_exec ;;
            descriptions) descriptions_fetched || die "описания не загружены:  scripts/install.sh install descriptions"
                          ok "описания производителей на месте" ;;
            train)        verify_train ;;
            docker)       verify_docker ;;
            *)            step "Проверка пакета $c"
                          code=0
                          package_cmd "$c" verify || code=$?
                          (( code == 0 )) || die "пакет $c не прошёл проверку (код $code)" ;;
        esac
    done
    printf '\n    %sпроверено: %s%s\n' "$C_GREEN$C_BOLD" "$*" "$C_OFF"
}

# ---------------------------------------------------------------------------------------------
# Status and report
# ---------------------------------------------------------------------------------------------

status_row() { printf '| %s | %s | %s |\n' "$1" "$2" "$3"; }

cmd_status() {
    local c state where pkg
    printf '\n| Компонент | Состояние | Где, версия |\n|---|---|---|\n'
    for c in $PLATFORM_STEPS; do
        component_installed "$c" && state="есть" || state="нет"
        case "$c" in
            base)         where="$(uv --version 2>/dev/null || echo '—')" ;;
            ros2)         where="/opt/ros/$ROS_DISTRO_NAME" ;;
            gazebo)       where="$(gz sim --version 2>/dev/null | head -1 || echo '—')" ;;
            exec)         where="$EXEC_VENV, Python $(venv_python_version "$EXEC_VENV" || echo '—'), платформа $(dist_version "$EXEC_PY" "$APP" || echo '—')" ;;
            descriptions) where="$ROBOTS_DIR/*/vendor" ;;
            train)        where="$TRAIN_ROOT, isaacsim $(dist_version "$TRAIN_PY" isaacsim || echo '—')" ;;
            docker)       where="$(docker --version 2>/dev/null || echo '—')" ;;
        esac
        status_row "$c" "$state" "$where"
    done
    for pkg in $PACKAGES; do
        # status exits 1 when a part is missing; the rows are what is wanted here.
        { package_cmd "$pkg" status || true; } | while IFS='|' read -r part state where; do
            status_row "$pkg: $part" "$state" "$where"
        done
    done
    printf '\nОкружения:  source scripts/activate.sh <exec|ros|train|%s>\n' "${PACKAGES// /|}"
}

# Versions of every environment, in the shape section 13 demands from an experiment card.
cmd_report() {
    local os_name gpu driver
    os_name="$(. /etc/os-release && echo "${PRETTY_NAME:-—}")"
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    cat <<REPORT

## Условия прогона

| Что | Значение |
|---|---|
| ОС | ${os_name} |
| Ядро | $(uname -r) |
| CPU | $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo) |
| RAM | $(gib "$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)") ГиБ |
| Swap | $(gib "$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)") ГиБ |
| GPU | ${gpu:-—} |
| Драйвер NVIDIA | ${driver:-—} |
| ROS 2 | $([[ -d "/opt/ros/$ROS_DISTRO_NAME" ]] && echo "$ROS_DISTRO_NAME" || echo '—') |
| Gazebo | $(gz sim --version 2>/dev/null | head -1 || echo '—') |
| MuJoCo (исполнение) | $(dist_version "$EXEC_PY" mujoco || echo '—') |
| Isaac Sim | $(dist_version "$TRAIN_PY" isaacsim || echo '—') |
| Isaac Lab | $(dist_version "$TRAIN_PY" isaaclab || echo '—') ($ISAACLAB_REF) |
| Коммит репозитория | $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '—') |
| Дата | $(date -Is) |
REPORT
    local pkg
    for pkg in $PACKAGES; do
        { package_cmd "$pkg" status || true; } | while IFS='|' read -r part _ where; do
            printf '| %s: %s | %s |\n' "$pkg" "$part" "$where"
        done
    done
    echo
}

# ---------------------------------------------------------------------------------------------

cmd_help() {
    cat <<HELP
Установка и проверка Linux-машины: платформа и окружения пакетов роботов.

  scripts/install.sh <команда> [компонент...] [флаги]

Команды:
  install [компонент...]   поставить; без аргументов — $DEFAULT_PLATFORM_STEPS и все пакеты
  check [компонент...]     проверить предусловия, ничего не меняя
  verify [компонент...]    запустить поставленное и убедиться, что оно работает
  sync [окружение...]      заново поставить платформу и пакеты в окружения exec, train, пакетов
                           (после git pull: точки входа пишутся при установке); без sudo и сборок
  status                   что поставлено и где
  report                   версии всех сред таблицей для карточки эксперимента
  help                     эта справка

Компоненты платформы:  $PLATFORM_STEPS
Пакеты роботов:        ${PACKAGES:-нет}   (examples/plugins/<пакет>/install.sh)
all                    всё перечисленное; train требует --accept-nvidia-eula

Флаги:
  --dry-run                печатать команды, ничего не выполняя
  --accept-nvidia-eula     согласие с лицензией NVIDIA; без него train не ставится

Переменные окружения:
  STACK_ROOT   окружения пакетов роботов, по умолчанию ~/robot-stack
  TRAIN_ROOT   стек обучения, по умолчанию ~/train-stack
  Что нужно пакету сверх этого, говорит он сам — README и шапка examples/plugins/<пакет>/install.sh.

После установки:  source scripts/activate.sh <exec|ros|train|пакет>
В .venv не выполняйте «uv sync»: он удаляет пакеты роботов. Обновить — scripts/install.sh sync.
HELP
}

main() {
    local positional=()
    while (( $# )); do
        case "$1" in
            --dry-run)            DRY_RUN=1 ;;
            --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=1 ;;
            -h|--help)            cmd_help; exit 0 ;;
            -*)                   die "неизвестный флаг: $1" "Справка:  scripts/install.sh help" ;;
            *)                    positional+=("$1") ;;
        esac
        shift
    done
    export DRY_RUN

    local cmd="${positional[0]:-help}" comps=()
    case "$cmd" in
        install|check|verify)
            mapfile -t comps < <(expand_components "${positional[@]:1}")
            (( ${#comps[@]} )) || exit 1   # expand_components has already said why
            ;;
    esac
    # Refused before the first download, not after an hour of the other components.
    if [[ "$cmd" == install && " ${comps[*]} " == *" train "* ]] && (( ! ACCEPT_NVIDIA_EULA )); then
        die "не принята лицензия NVIDIA, а среди компонентов есть train" \
            "Прочитайте условия Isaac Sim и повторите с флагом --accept-nvidia-eula" \
            "или поставьте остальное без train."
    fi

    case "$cmd" in
        check)   cmd_check "${comps[@]}" ;;
        install) cmd_check "${comps[@]}"; cmd_install "${comps[@]}" ;;
        verify)  cmd_verify "${comps[@]}" ;;
        sync)    cmd_sync "${positional[@]:1}" ;;
        status)  cmd_status ;;
        report)  cmd_report ;;
        help)    cmd_help ;;
        *)       die "неизвестная команда: $cmd" "Справка:  scripts/install.sh help" ;;
    esac
}

main "$@"
