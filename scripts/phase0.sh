#!/usr/bin/env bash
#
# Phase 0 bring-up on the Linux test machine: "environment is up, all three
# worlds start" (ARCHITECTURE.md, section 14).
#
# Scope:
#   - ROS 2 Jazzy + Gazebo Harmonic  -> execution world
#   - MuJoCo                         -> second physics world (ADR-0002)
#   - Isaac Sim / Isaac Lab + rsl_rl -> training world, isolated venv (ADR-0005)
#
# Two virtual environments, not one. Invariant 2 ("worlds touch through files")
# is enforced here already: the training stack never shares an interpreter with
# the execution stack, so ROS 2 cannot leak into training (ADR-0003).
#
# The script never repairs the system silently. A violated precondition stops
# the run and prints the command a human is expected to inspect or execute.
#
# Usage:  scripts/phase0.sh help

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants. Versions live here and nowhere else in the file.
# ---------------------------------------------------------------------------

readonly REQUIRED_UBUNTU="24.04"          # ADR-0005; ROS 2 Jazzy is tied to noble
readonly ROS_DISTRO_NAME="jazzy"
readonly GZ_DISTRO="harmonic"

readonly TRAIN_PYTHON="3.11"              # Isaac Sim wheels; Ubuntu 24.04 ships 3.12
readonly ISAACSIM_SPEC="isaacsim[all,extscache]==5.1.0"
readonly ISAACLAB_REF="v2.3.0"
readonly TORCH_SPEC="torch==2.7.0 torchvision==0.22.0"
readonly TORCH_INDEX="https://download.pytorch.org/whl/cu128"
readonly FLATDICT_SPEC="flatdict==4.0.1"       # version Isaac Lab pins; see ensure_flatdict
readonly EXEC_EXTRAS="sim dev"                 # optional groups in pyproject.toml: MuJoCo, tests

readonly INSTALL_STEPS="base ros2 gazebo exec isaac docker"
readonly VERIFY_ENVS="ros2 gazebo mujoco isaac"

# Thresholds from ADR-0005. Changing a number here is a decision, not a tweak.
readonly MIN_SWAP_GIB=16                  # 15.3 GiB RAM: swap is what keeps Isaac alive
readonly MIN_VRAM_MIB=7000                # peak VRAM budget is 7 GiB out of 8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/train-stack}"   # Isaac lives outside the repo: tens of GiB
readonly TRAIN_ROOT
readonly EXEC_VENV="$REPO_ROOT/.venv"
readonly TRAIN_VENV="$TRAIN_ROOT/venv"
readonly ISAACLAB_DIR="$TRAIN_ROOT/IsaacLab"

DRY_RUN=0
ACCEPT_NVIDIA_EULA=0

# ---------------------------------------------------------------------------
# Output. Messages address a human, so they are in Russian; identifiers and
# comments stay English (CLAUDE.md).
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_BOLD=''; C_OFF=''
fi

step() { printf '\n%s==> %s%s\n' "$C_BOLD$C_BLUE" "$*" "$C_OFF"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %sok%s  %s\n' "$C_GREEN" "$C_OFF" "$*"; }
warn() { printf '    %sвнимание%s  %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }

die() {
    printf '\n%sостановлено:%s %s\n' "$C_RED$C_BOLD" "$C_OFF" "$1" >&2
    shift
    local line
    for line in "$@"; do printf '    %s\n' "$line" >&2; done
    printf '\n' >&2
    exit 1
}

run() {
    if (( DRY_RUN )); then
        printf '    %s[dry-run]%s %s\n' "$C_YELLOW" "$C_OFF" "$*"
        return 0
    fi
    "$@"
}

# Same as run(), for pipelines and redirections that need a shell.
run_sh() {
    if (( DRY_RUN )); then
        printf '    %s[dry-run]%s %s\n' "$C_YELLOW" "$C_OFF" "$1"
        return 0
    fi
    bash -c "$1"
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

# Names are validated before any work starts. Catching a typo after an hour of
# downloads is the same defect as catching a contract mismatch on the robot.
validate_names() {
    local kind="$1" allowed="$2"
    shift 2
    local name
    for name in "$@"; do
        if [[ " $allowed " != *" $name "* ]]; then
            if [[ "$name" == *=* ]]; then
                die "неизвестный $kind: $name" \
                    "Похоже на переменную окружения. Она ставится перед командой, а не после:" \
                    "    ${name} ./phase0.sh ..." \
                    "Доступны: $allowed"
            fi
            die "неизвестный $kind: $name" "Доступны: $allowed"
        fi
    done
}

apt_install() {
    local pkg missing=()
    for pkg in "$@"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if (( ${#missing[@]} == 0 )); then
        ok "уже установлено: $*"
        return 0
    fi
    info "ставится: ${missing[*]}"
    run sudo apt-get install -y "${missing[@]}"
}

apt_update_once() {
    if [[ -z "${APT_UPDATED:-}" ]]; then
        run sudo apt-get update
        APT_UPDATED=1
    fi
}

gib() { awk -v kb="$1" 'BEGIN { printf "%.1f", kb / 1048576 }'; }

# ---------------------------------------------------------------------------
# Preconditions. Everything here is measured, nothing is repaired.
# ---------------------------------------------------------------------------

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

check_user() {
    [[ "$(id -u)" -ne 0 ]] || die \
        "скрипт запущен от root" \
        "venv, кэши и группа docker должны принадлежать рабочему пользователю." \
        "Запускайте от своего пользователя: sudo вызывается точечно там, где он нужен."
    if ! sudo -n true 2>/dev/null; then
        info "нужен sudo, пароль спрашивается один раз"
        sudo -v || die "sudo недоступен — установка системных пакетов невозможна."
    fi
    ok "пользователь: $(id -un), sudo доступен"
}

check_memory() {
    local ram_kb swap_kb swap_gib
    ram_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    swap_kb="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_gib="$(gib "$swap_kb")"
    info "RAM: $(gib "$ram_kb") ГиБ, swap: ${swap_gib} ГиБ"

    # Compared in KiB: the printed value goes through awk and carries the
    # locale decimal separator, which turns a float comparison into a lie.
    if (( swap_kb < MIN_SWAP_GIB * 1048576 )); then
        die "swap ${swap_gib} ГиБ, по ADR-0005 нужно не меньше ${MIN_SWAP_GIB} ГиБ" \
            "При 15.3 ГиБ RAM Isaac Sim уходит в OOM без подкачки — ровно та нехватка памяти," \
            "из-за которой отвергнут WSL2." \
            "Переразбивать диск не требуется, достаточно файла подкачки на корне:" \
            "    sudo fallocate -l ${MIN_SWAP_GIB}G /swapfile && sudo chmod 600 /swapfile" \
            "    sudo mkswap /swapfile && sudo swapon /swapfile" \
            "    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab" \
            "Скрипт этого не делает сам: разметка памяти — решение человека."
    fi
    ok "памяти достаточно по порогу ADR-0005"
}

# Rough disk cost of a step in GiB, from the ADR-0005 estimate.
step_cost_gib() {
    case "$1" in
        base)   echo 1  ;;
        ros2)   echo 4  ;;
        gazebo) echo 2  ;;
        exec)   echo 1  ;;
        isaac)  echo 40 ;;   # Isaac Sim ~20 plus asset cache 10-20
        docker) echo 10 ;;
        *)      echo 0  ;;
    esac
}

step_installed() {
    case "$1" in
        base)   has_cmd uv ;;
        ros2)   [[ -d "/opt/ros/$ROS_DISTRO_NAME" ]] ;;
        gazebo) has_cmd gz ;;
        exec)   [[ -x "$EXEC_VENV/bin/python" ]] &&
                "$EXEC_VENV/bin/python" -c 'import mujoco' >/dev/null 2>&1 ;;
        isaac)  [[ -x "$TRAIN_VENV/bin/python" ]] &&
                "$TRAIN_VENV/bin/python" -m pip show isaacsim-rl >/dev/null 2>&1 ;;
        docker) has_cmd docker ;;
        *)      return 1 ;;
    esac
}

# The question is "is there room for what we are about to install", not "is this
# a fresh machine". A step already installed occupies its space and asks for no
# more, so a fixed global threshold would refuse every second run.
check_disk() {
    local steps=("$@")
    (( ${#steps[@]} )) || steps=(base ros2 gazebo exec isaac docker)

    local s need=0 pending=()
    for s in "${steps[@]}"; do
        step_installed "$s" && continue
        pending+=("$s")
        need=$(( need + $(step_cost_gib "$s") ))
    done

    local free_kb
    free_kb="$(df -Pk / | awk 'NR == 2 { print $4 }')"   # KiB: no suffix, no locale

    if (( ${#pending[@]} == 0 )); then
        ok "свободно на /: $(gib "$free_kb") ГиБ; ставить нечего, всё уже на месте"
        return 0
    fi
    info "свободно на /: $(gib "$free_kb") ГиБ; предстоит: ${pending[*]}, оценка ${need} ГиБ"

    if (( free_kb < need * 1048576 )); then
        die "на / свободно $(gib "$free_kb") ГиБ, на шаги ${pending[*]} нужно около ${need} ГиБ" \
            "Оценка ADR-0005: Isaac Sim с кэшем ассетов 30-40, ROS 2 desktop 4, Gazebo 2, образы Docker 10." \
            "Ставьте шаги по одному или освободите место."
    fi
    ok "места достаточно"
}

check_gpu() {
    has_cmd nvidia-smi || die \
        "nvidia-smi не найден — драйвер NVIDIA не установлен" \
        "Это главный риск фазы 0: гибридная графика Intel iGPU плюс RTX 4060." \
        "Установите проприетарный драйвер и перезагрузитесь:" \
        "    sudo ubuntu-drivers install" \
        "Драйвер ставится отдельно от этого скрипта: он требует перезагрузки, а его версия" \
        "входит в условия каждого эксперимента (раздел 13)."

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
        warn "vulkaninfo не найден, проверка отложена до шага base (пакет vulkan-tools)"
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
    if run_sh "curl -fsS --max-time 10 -o /dev/null https://packages.ros.org"; then
        ok "сеть доступна"
    else
        die "нет доступа к packages.ros.org — установка невозможна."
    fi
}

# Takes the steps that are about to run, so the disk check measures the right
# thing. Without arguments it assumes the full set.
cmd_check() {
    step "Предусловия"
    check_os
    check_arch
    check_user
    check_memory
    check_disk "$@"
    check_gpu
    check_vulkan
    check_network
    printf '\n    %sпредусловия выполнены%s\n' "$C_GREEN$C_BOLD" "$C_OFF"
}

# ---------------------------------------------------------------------------
# Installation. Every step is idempotent: a second run changes nothing.
# ---------------------------------------------------------------------------

install_base() {
    step "Базовые пакеты и uv"
    apt_update_once
    apt_install curl wget git build-essential cmake pkg-config \
                python3-venv python3-pip python3-dev \
                software-properties-common ca-certificates gnupg lsb-release \
                vulkan-tools libvulkan1 mesa-utils
    if has_cmd uv; then
        ok "uv уже установлен: $(uv --version)"
    else
        info "ставится uv"
        run_sh "curl -LsSf https://astral.sh/uv/install.sh | sh"
        info "uv положен в ~/.local/bin — добавьте каталог в PATH, если его там нет"
    fi
}

install_ros2() {
    step "ROS 2 ${ROS_DISTRO_NAME}"
    if [[ -d "/opt/ros/$ROS_DISTRO_NAME" ]]; then
        ok "уже установлен: /opt/ros/$ROS_DISTRO_NAME"
    else
        run sudo add-apt-repository -y universe
        run_sh "sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(. /etc/os-release && echo \$UBUNTU_CODENAME) main\" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null"
        unset APT_UPDATED
        apt_update_once
        apt_install "ros-${ROS_DISTRO_NAME}-desktop" ros-dev-tools
    fi
    # Control stack and URDF tooling: needed from phase 2 onward, cheap to add now.
    apt_install "ros-${ROS_DISTRO_NAME}-ros2-control" \
                "ros-${ROS_DISTRO_NAME}-ros2-controllers" \
                "ros-${ROS_DISTRO_NAME}-xacro" \
                "ros-${ROS_DISTRO_NAME}-robot-state-publisher" \
                "ros-${ROS_DISTRO_NAME}-rosbag2-storage-mcap"
    info "MCAP-хранилище rosbag2 поставлено явно: ADR-0003 требует MCAP на исполнении"
}

install_gazebo() {
    step "Gazebo ${GZ_DISTRO}"
    if has_cmd gz; then
        ok "уже установлен: $(gz sim --version 2>/dev/null | head -1)"
    else
        run_sh "sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg"
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null"
        unset APT_UPDATED
        apt_update_once
        apt_install "gz-${GZ_DISTRO}"
    fi
    # The bridge is what makes Gazebo an execution world rather than a viewer.
    apt_install "ros-${ROS_DISTRO_NAME}-ros-gz" \
                "ros-${ROS_DISTRO_NAME}-gz-ros2-control"
}

install_exec_env() {
    step "Окружение исполнения (MuJoCo)"
    has_cmd uv || die "uv не найден" "Сначала выполните шаг base:  scripts/phase0.sh install base"

    if [[ -d "$EXEC_VENV" ]]; then
        ok "venv уже есть: $EXEC_VENV"
    else
        run uv venv --python "$TRAIN_PYTHON" "$EXEC_VENV"
    fi

    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        # The core installs without vendor dependencies, so MuJoCo sits in an
        # optional group and a plain `uv sync` would leave the second world out.
        info "зависимости ставятся из pyproject.toml, группы: $EXEC_EXTRAS"
        local extras="" name
        for name in $EXEC_EXTRAS; do extras="$extras --extra $name"; done
        run_sh "cd '$REPO_ROOT' && uv sync$extras"
    else
        info "пакета ядра ещё нет (фаза 1), ставится только MuJoCo"
        run_sh "VIRTUAL_ENV='$EXEC_VENV' uv pip install mujoco numpy"
    fi
}

# isaaclab.sh installs its extensions with `python -m pip` and locates Isaac Sim
# through pip metadata. `uv venv` creates an environment without pip, so the
# tool needs it put there explicitly — this is a requirement of isaaclab.sh,
# not a workaround for it.
ensure_pip() {
    local venv="$1"
    if (( DRY_RUN )); then
        info "[dry-run] проверка pip в $venv"
        return 0
    fi
    if "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
        ok "pip в окружении есть"
        return 0
    fi
    info "в окружении нет pip, а isaaclab.sh ставит расширения через него"
    run_sh "VIRTUAL_ENV='$venv' uv pip install pip setuptools wheel"
}

# isaacsim-rl comes from the [all] extra and is exactly what isaaclab.sh looks
# for. Its absence must stop the run here, with the reason named, rather than
# surface later as "Python executable not found at _isaac_sim/python.sh".
check_isaacsim_installed() {
    if (( DRY_RUN )); then
        info "[dry-run] проверка пакета isaacsim-rl"
        return 0
    fi
    if "$TRAIN_VENV/bin/python" -m pip show isaacsim-rl >/dev/null 2>&1; then
        ok "пакеты Isaac Sim на месте"
        return 0
    fi
    die "в окружении обучения нет пакета isaacsim-rl" \
        "Isaac Lab ищет Isaac Sim именно по нему, поэтому дальше идти бессмысленно." \
        "Вероятная причина: набор extras '[all]' поставлен не полностью." \
        "Проверьте, что стоит на самом деле:" \
        "    $TRAIN_VENV/bin/python -m pip list | grep isaacsim" \
        "Повторная установка:" \
        "    VIRTUAL_ENV='$TRAIN_VENV' uv pip install --reinstall '$ISAACSIM_SPEC' --extra-index-url https://pypi.nvidia.com"
}

# isaaclab.sh installs its modules one by one and exits 0 even when one of them
# fails to build, so a missing package surfaces much later as an import error in
# a tutorial. The installed set is checked here instead.
check_isaaclab_installed() {
    if (( DRY_RUN )); then
        info "[dry-run] проверка пакетов Isaac Lab"
        return 0
    fi
    local pkg missing=()
    for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
        "$TRAIN_VENV/bin/python" -m pip show "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if (( ${#missing[@]} == 0 )); then
        ok "пакеты Isaac Lab установлены"
        return 0
    fi
    die "после установки не хватает пакетов Isaac Lab: ${missing[*]}" \
        "isaaclab.sh ставит модули по очереди и завершается успешно, даже если один из них" \
        "не собрался, — поэтому ошибка всплыла бы позже, при импорте в тьюториале." \
        "Соберите недостающий модуль вручную и посмотрите на причину:" \
        "    cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && ./isaaclab.sh --install rsl_rl"
}

# flatdict 4.0.1 has a setup.py that imports pkg_resources, which setuptools 81
# and later no longer ship. pip builds it in an isolated environment where the
# newest setuptools lands, so the build fails there — and with it the whole
# isaaclab package, which pins that exact version. Building against the venv's
# own setuptools is the only way through; the upstream pin is not ours to move.
ensure_flatdict() {
    if (( DRY_RUN )); then
        info "[dry-run] проверка flatdict"
        return 0
    fi
    if "$TRAIN_VENV/bin/python" -m pip show flatdict >/dev/null 2>&1; then
        ok "flatdict на месте"
        return 0
    fi
    info "flatdict ставится без изоляции сборки: его setup.py требует pkg_resources"
    run_sh "VIRTUAL_ENV='$TRAIN_VENV' uv pip install --no-build-isolation '$FLATDICT_SPEC'"
}

# Isaac Lab lifts shared packages above the versions isaacsim-kernel pins. Only
# a real launch shows whether that matters, so this reports and does not stop:
# the answer belongs to `verify isaac`, not to a guess made here.
report_pip_conflicts() {
    if (( DRY_RUN )); then
        return 0
    fi
    local out
    if out="$("$TRAIN_VENV/bin/python" -m pip check 2>&1)"; then
        ok "конфликтов версий в окружении обучения нет"
        return 0
    fi
    warn "конфликты версий в окружении обучения:"
    printf '%s\n' "$out" | sed 's/^/        /' >&2
    warn "Isaac Lab поднял пакеты выше пинов isaacsim-kernel."
    warn "Существенно это или нет, показывает запуск:  ./phase0.sh verify isaac"
}

install_isaac() {
    step "Окружение обучения (Isaac Sim, Isaac Lab, rsl_rl)"
    has_cmd uv || die "uv не найден" "Сначала выполните шаг base:  scripts/phase0.sh install base"

    if (( ! ACCEPT_NVIDIA_EULA )); then
        die "не принята лицензия NVIDIA" \
            "Пакеты Isaac Sim ставятся только после согласия с лицензией NVIDIA." \
            "Скрипт не принимает её за человека. Прочитайте условия и повторите с флагом:" \
            "    scripts/phase0.sh install isaac --accept-nvidia-eula"
    fi

    run mkdir -p "$TRAIN_ROOT"
    if [[ -d "$TRAIN_VENV" ]]; then
        ok "venv обучения уже есть: $TRAIN_VENV"
    else
        info "отдельный venv на Python ${TRAIN_PYTHON}: миры не делят интерпретатор (инвариант 2)"
        run uv venv --seed --python "$TRAIN_PYTHON" "$TRAIN_VENV"
    fi
    ensure_pip "$TRAIN_VENV"

    info "torch под CUDA 12.8, несколько ГиБ"
    run_sh "VIRTUAL_ENV='$TRAIN_VENV' uv pip install $TORCH_SPEC --index-url $TORCH_INDEX"

    info "Isaac Sim: загрузка десятков ГиБ, шаг долгий"
    run_sh "VIRTUAL_ENV='$TRAIN_VENV' uv pip install '$ISAACSIM_SPEC' --extra-index-url https://pypi.nvidia.com"
    check_isaacsim_installed

    if [[ -d "$ISAACLAB_DIR/.git" ]]; then
        ok "Isaac Lab уже склонирован: $ISAACLAB_DIR"
    else
        run git clone --depth 1 --branch "$ISAACLAB_REF" \
            https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    fi

    ensure_flatdict

    info "Isaac Lab ставится с rsl_rl: других RL-фреймворков в стеке нет (раздел 11)"
    run_sh "cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && ./isaaclab.sh --install rsl_rl"

    check_isaaclab_installed
    report_pip_conflicts

    info "Первый запуск Isaac Sim компилирует шейдеры: это десятки минут и выглядит как зависание."
    info "Дальше:  ./phase0.sh verify isaac"
}

install_docker() {
    step "Docker и NVIDIA Container Toolkit"
    if has_cmd docker; then
        ok "уже установлен: $(docker --version)"
    else
        run sudo install -m 0755 -d /etc/apt/keyrings
        run_sh "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg"
        run sudo chmod a+r /etc/apt/keyrings/docker.gpg
        run_sh "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable\" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null"
        unset APT_UPDATED
        apt_update_once
        apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi

    if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        ok "nvidia-container-toolkit уже установлен"
    else
        run_sh "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run_sh "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
        unset APT_UPDATED
        apt_update_once
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

cmd_install() {
    local steps=("$@")
    if (( ${#steps[@]} == 0 )); then
        steps=(base ros2 gazebo exec isaac docker)
    fi
    local s
    for s in "${steps[@]}"; do
        case "$s" in
            base)   install_base ;;
            ros2)   install_ros2 ;;
            gazebo) install_gazebo ;;
            exec)   install_exec_env ;;
            isaac)  install_isaac ;;
            docker) install_docker ;;
            *)      die "неизвестный шаг установки: $s" "Доступны: base ros2 gazebo exec isaac docker" ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Verification: phase 0 closes when all three worlds actually start.
# ---------------------------------------------------------------------------

verify_ros2() {
    step "Проверка ROS 2"
    [[ -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]] || die "ROS 2 $ROS_DISTRO_NAME не установлен."
    run_sh "source /opt/ros/$ROS_DISTRO_NAME/setup.bash && echo \"    пакетов доступно: \$(ros2 pkg list | wc -l)\""
    ok "ROS 2 $ROS_DISTRO_NAME отвечает"
}

verify_gazebo() {
    step "Проверка Gazebo, headless, 200 шагов"
    has_cmd gz || die "gz не найден — Gazebo не установлен."
    run_sh "gz sim -s -r --iterations 200 empty.sdf >/dev/null 2>&1"
    ok "сервер Gazebo отработал 200 шагов и завершился"
}

verify_mujoco() {
    step "Проверка MuJoCo"
    [[ -x "$EXEC_VENV/bin/python" ]] || die "venv исполнения не создан" \
        "Выполните:  scripts/phase0.sh install exec"
    if (( DRY_RUN )); then
        info "[dry-run] шаг модели MuJoCo"
        return 0
    fi
    "$EXEC_VENV/bin/python" -c 'import mujoco' 2>/dev/null || die \
        "в окружении исполнения нет пакета mujoco" \
        "Ядро ставится без вендорских зависимостей (раздел 10), поэтому MuJoCo живёт" \
        "в опциональной группе 'sim' и приходит вместе с шагом установки:" \
        "    ./phase0.sh install exec"
    "$EXEC_VENV/bin/python" - <<'PY'
import mujoco

XML = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"    MuJoCo {mujoco.__version__}: 100 шагов, высота {data.qpos[2]:.4f} м")
PY
    ok "MuJoCo считает физику"
}

verify_isaac() {
    step "Проверка Isaac Sim, headless"
    [[ -x "$TRAIN_VENV/bin/python" ]] || die "venv обучения не создан" \
        "Выполните:  scripts/phase0.sh install isaac --accept-nvidia-eula"
    [[ -d "$ISAACLAB_DIR" ]] || die "Isaac Lab не склонирован в $ISAACLAB_DIR"
    info "первый запуск компилирует шейдеры и занимает десятки минут"
    run_sh "cd '$ISAACLAB_DIR' && source '$TRAIN_VENV/bin/activate' && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless"
    ok "Isaac Sim поднял пустую сцену и завершился"
}

cmd_verify() {
    local envs=("$@")
    if (( ${#envs[@]} == 0 )); then
        envs=(ros2 gazebo mujoco isaac)
    fi
    local e
    for e in "${envs[@]}"; do
        case "$e" in
            ros2)   verify_ros2 ;;
            gazebo) verify_gazebo ;;
            mujoco) verify_mujoco ;;
            isaac)  verify_isaac ;;
            *)      die "неизвестная среда: $e" "Доступны: ros2 gazebo mujoco isaac" ;;
        esac
    done
    printf '\n    %sфаза 0: среды запускаются%s\n' "$C_GREEN$C_BOLD" "$C_OFF"
}

# ---------------------------------------------------------------------------
# Report: versions of every environment, in the shape section 13 demands from
# an experiment card. The output goes into docs/experiments/NNNN-*.md.
# ---------------------------------------------------------------------------

cmd_report() {
    local os_name gpu driver ros gz mj isaac commit
    os_name="$(. /etc/os-release && echo "${PRETTY_NAME:-—}")"
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    ros="$([[ -d "/opt/ros/$ROS_DISTRO_NAME" ]] && echo "$ROS_DISTRO_NAME" || true)"
    gz="$(gz sim --version 2>/dev/null | head -1 || true)"
    mj="$("$EXEC_VENV/bin/python" -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null || true)"
    isaac="$("$TRAIN_VENV/bin/python" -c 'import isaacsim; print(isaacsim.__version__)' 2>/dev/null || true)"
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"

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
| ROS 2 | ${ros:-—} |
| Gazebo | ${gz:-—} |
| MuJoCo | ${mj:-—} |
| Isaac Sim | ${isaac:-—} |
| Isaac Lab | ${ISAACLAB_REF} |
| Коммит репозитория | ${commit:-—} |
| Дата | $(date -Is) |

REPORT
}

# ---------------------------------------------------------------------------

cmd_help() {
    cat <<'HELP'
Разворот фазы 0 на Linux: окружение поднято, все три среды запускаются.

  scripts/phase0.sh <команда> [аргументы] [флаги]

Команды:
  check [шаг...]            проверить предусловия и ничего не менять
  install [шаг...]          поставить среды; без аргументов — все шаги подряд
  verify [среда...]         запустить среды и убедиться, что они работают
  report                    версии всех сред таблицей для карточки эксперимента
  all                       check, install, verify
  help                      эта справка

Шаги установки:  base ros2 gazebo exec isaac docker
Среды проверки:  ros2 gazebo mujoco isaac

Флаги:
  --dry-run                 печатать команды, ничего не выполняя
  --accept-nvidia-eula      согласие с лицензией NVIDIA; без него шаг isaac не идёт

Переменные окружения:
  TRAIN_ROOT                где живёт стек обучения, по умолчанию ~/train-stack

Скрипт ничего не чинит молча: нарушенное предусловие останавливает работу
и печатает команду, которую человек выполняет сам.
HELP
}

main() {
    local positional=()
    while (( $# )); do
        case "$1" in
            --dry-run)            DRY_RUN=1 ;;
            --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=1 ;;
            -h|--help)            cmd_help; exit 0 ;;
            -*)                   die "неизвестный флаг: $1" "Справка:  scripts/phase0.sh help" ;;
            *)                    positional+=("$1") ;;
        esac
        shift
    done

    local cmd="${positional[0]:-help}"
    local args=("${positional[@]:1}")

    case "$cmd" in
        check)   validate_names "шаг установки" "$INSTALL_STEPS" "${args[@]}"
                 cmd_check "${args[@]}" ;;
        install) validate_names "шаг установки" "$INSTALL_STEPS" "${args[@]}"
                 cmd_check "${args[@]}"; cmd_install "${args[@]}" ;;
        verify)  validate_names "среда" "$VERIFY_ENVS" "${args[@]}"
                 cmd_verify "${args[@]}" ;;
        report)  cmd_report ;;
        all)     cmd_check; cmd_install; cmd_verify ;;
        help)    cmd_help ;;
        *)       die "неизвестная команда: $cmd" "Справка:  scripts/phase0.sh help" ;;
    esac
}

main "$@"
