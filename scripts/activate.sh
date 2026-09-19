# shellcheck shell=bash
#
# Enter one of the machine's environments in the current bash shell (ADR-0071):
#
#   source scripts/activate.sh exec        .venv; twins of robot packages on PATH.
#                                          validate, gen, sim run of a MuJoCo world, mission run
#   source scripts/activate.sh ros         ROS 2, the packages' workspaces, then .venv.
#                                          sim run of a Gazebo world, run of a ros2_control robot
#   source scripts/activate.sh train       the training stack: train run, task list
#   source scripts/activate.sh <package>   a robot package's own venv, e.g. unitree_g1:
#                                          run / mission robot for that robot
#
# Which process needs which environment is the platform's rule, not this file's: the adapter
# of a DDS robot must not load ROS 2's libddsc (ADR-0022), training never sees rclpy
# (invariant 2). A shell that has sourced ROS 2 cannot unsource it, so train and package
# environments refuse it and ask for a new terminal instead of pretending.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "activate.sh подключается, а не запускается:  source scripts/activate.sh <окружение>" >&2
    exit 2
fi

_activate() {
    local name="${1:-}" file pkg
    # shellcheck source=lib/common.sh
    source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

    local packages=()
    for file in "$STACK_ROOT"/*/venv/bin/activate; do
        [[ -f "$file" ]] && packages+=("$(basename "$(dirname "$(dirname "$(dirname "$file")")")")")
    done

    if [[ -z "$name" ]]; then
        echo "окружения: exec ros train ${packages[*]}" >&2
        return 2
    fi

    # Another venv first out: two activated venvs are one PATH with two answers.
    if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -F deactivate >/dev/null; then
        deactivate
    fi

    case "$name" in
        exec|ros)
            [[ -f "$EXEC_VENV/bin/activate" ]] ||
                { echo "нет $EXEC_VENV:  scripts/install.sh install exec" >&2; return 1; }
            if [[ "$name" == ros ]]; then
                [[ -f "$ROS_SETUP" ]] ||
                    { echo "нет $ROS_SETUP:  scripts/install.sh install ros2" >&2; return 1; }
                # shellcheck disable=SC1090
                source "$ROS_SETUP"
                for file in "$STACK_ROOT"/*/env/ros.sh; do
                    # shellcheck disable=SC1090
                    [[ -f "$file" ]] && source "$file"
                done
            fi
            for file in "$STACK_ROOT"/*/env/exec.sh; do
                # shellcheck disable=SC1090
                [[ -f "$file" ]] && source "$file"
            done
            # The venv last: its python is the first on PATH, ROS 2's modules come by PYTHONPATH.
            # shellcheck disable=SC1091
            source "$EXEC_VENV/bin/activate"
            ;;
        train)
            if [[ -n "${ROS_DISTRO:-}" ]]; then
                echo "в этой оболочке подключён ROS 2 ($ROS_DISTRO), а в обучение он не попадает (инвариант 2)." >&2
                echo "Откройте новый терминал без ROS 2 (проверьте ~/.bashrc)." >&2
                return 1
            fi
            [[ -f "$TRAIN_VENV/bin/activate" ]] ||
                { echo "нет $TRAIN_VENV:  scripts/install.sh install train --accept-nvidia-eula" >&2; return 1; }
            # shellcheck disable=SC1091
            source "$TRAIN_VENV/bin/activate"
            ;;
        *)
            if [[ ! -f "$STACK_ROOT/$name/venv/bin/activate" ]]; then
                echo "нет окружения $name в $STACK_ROOT:  scripts/install.sh install $name" >&2
                echo "окружения: exec ros train ${packages[*]}" >&2
                return 1
            fi
            if [[ -n "${ROS_DISTRO:-}" ]]; then
                echo "в этой оболочке подключён ROS 2 ($ROS_DISTRO); процесс робота $name его не грузит." >&2
                echo "Откройте новый терминал без ROS 2 (проверьте ~/.bashrc)." >&2
                return 1
            fi
            # shellcheck disable=SC1090
            source "$STACK_ROOT/$name/venv/bin/activate"
            ;;
    esac
    echo "окружение $name: $(command -v "$APP")"
}

_activate "$@"
set -- $?
unset -f _activate
# common.sh defines helpers for scripts; the interactive shell keeps only the environment.
unset -f step info ok warn die run run_sh run_clean has_cmd apt_update_once apt_install \
    fetch_pinned venv_python_version dist_version plugin_editables package_installers
return "$1"
