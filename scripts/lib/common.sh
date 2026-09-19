# shellcheck shell=bash
#
# Shared by scripts/install.sh, the installers of robot packages (examples/plugins/*/install.sh)
# and the acceptance runs in scripts/runs/. Sourced, never executed.
#
# Only definitions live here: no shell options are set and nothing exits on source, because
# scripts/activate.sh sources this file into the user's own interactive shell.
#
# Where things live on the Linux machine (ADR-0071):
#   $REPO_ROOT/.venv          execution environment: the platform, ROS 2's Python, MuJoCo
#   $TRAIN_ROOT               training stack: Isaac Sim, Isaac Lab, rsl_rl (tens of GiB)
#   $STACK_ROOT/<package>/    everything a robot package builds for itself: its venv, its
#                             vendor SDKs, its ROS 2 workspace, its twin; world.sh adds what the
#                             world process needs from it (scripts/activate.sh ros)

COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON_SH_DIR/../.." && pwd)"

# The platform's name is written in src/<name>/core/branding.py and, for packaging, in
# pyproject.toml. Scripts read it from there instead of spelling it (CLAUDE.md).
APP="$(sed -n 's/^name = "\([^"]*\)"$/\1/p' "$REPO_ROOT/pyproject.toml" | head -n 1)"
ENV_PREFIX="${APP^^}"

ROS_DISTRO_NAME="jazzy"
ROS_SETUP="/opt/ros/$ROS_DISTRO_NAME/setup.bash"

EXEC_VENV="$REPO_ROOT/.venv"
EXEC_PY="$EXEC_VENV/bin/python"
EXEC_CLI="$EXEC_VENV/bin/$APP"

TRAIN_ROOT="${TRAIN_ROOT:-$HOME/train-stack}"
TRAIN_VENV="$TRAIN_ROOT/venv"
TRAIN_PY="$TRAIN_VENV/bin/python"
TRAIN_CLI="$TRAIN_VENV/bin/$APP"
ISAACLAB_DIR="$TRAIN_ROOT/IsaacLab"

STACK_ROOT="${STACK_ROOT:-$HOME/robot-stack}"

PLUGINS_DIR="$REPO_ROOT/examples/plugins"
ROBOTS_DIR="$REPO_ROOT/examples/robots"

DRY_RUN="${DRY_RUN:-0}"

# -- output: messages address a human, so they are in Russian --------------------------------

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

# die <message> [line...]: stop, saying what a human is expected to inspect or execute.
die() {
    printf '\n%sостановлено:%s %s\n' "$C_RED$C_BOLD" "$C_OFF" "$1" >&2
    shift
    local line
    for line in "$@"; do printf '    %s\n' "$line" >&2; done
    printf '\n' >&2
    exit 1
}

# -- execution ---------------------------------------------------------------------------------

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

# A command in an environment that inherits nothing but HOME and a plain PATH: whatever the
# calling shell has sourced -- ROS 2, an activated venv -- cannot decide the outcome. Builds
# against ROS 2 source it explicitly inside the command.
run_clean() {
    if (( DRY_RUN )); then
        printf '    %s[dry-run, чистое окружение]%s %s\n' "$C_YELLOW" "$C_OFF" "$1"
        return 0
    fi
    env -i HOME="$HOME" USER="${USER:-$(id -un)}" LANG=C.UTF-8 TERM="${TERM:-dumb}" \
        PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        bash --noprofile --norc -c "$1"
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

# -- system packages ---------------------------------------------------------------------------

apt_update_once() {
    if [[ -z "${APT_UPDATED:-}" ]]; then
        run sudo apt-get update
        APT_UPDATED=1
    fi
}

# apt_install <package>...: only what is missing, and named before it is installed.
apt_install() {
    local pkg missing=()
    for pkg in "$@"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if (( ${#missing[@]} == 0 )); then
        ok "системные пакеты на месте: $*"
        return 0
    fi
    info "ставятся системные пакеты: ${missing[*]}"
    apt_update_once
    run sudo apt-get install -y "${missing[@]}"
}

# -- sources at pinned commits -----------------------------------------------------------------

# fetch_pinned <url> <commit-or-tag> <dir>: one shallow checkout, reused when it is already the
# pinned one. A checkout of another commit stops the run: replacing it is a human's decision.
fetch_pinned() {
    local url="$1" ref="$2" dir="$3" have pinned
    if [[ -d "$dir/.git" ]]; then
        # A tag leaves no local ref in a shallow fetch, so the ref asked for is kept beside it.
        have="$(git -C "$dir" rev-parse HEAD)"
        pinned="$(cat "$dir/.git/pinned-ref" 2>/dev/null || true)"
        if [[ "$pinned" == "$ref" || "$have" == "$ref"* ]]; then
            ok "исходник на месте: $dir @ ${have:0:7}"
            return 0
        fi
        die "в $dir другой коммит: ${have:0:7}, закреплён $ref" \
            "Закреплённые коммиты — решение (ADR), а не умолчание. Если замена намеренна:" \
            "    rm -rf '$dir'  и повторите установку"
    fi
    info "загружается $url @ $ref"
    run mkdir -p "$dir"
    run git -C "$dir" init -q
    run git -C "$dir" remote add origin "$url"
    run git -C "$dir" fetch -q --depth 1 origin "$ref"
    run git -C "$dir" checkout -q FETCH_HEAD
    run_sh "echo '$ref' > '$dir/.git/pinned-ref'"
}

# -- Python environments -----------------------------------------------------------------------

venv_python_version() {
    "$1/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

# dist_version <python> <distribution>: the installed version, or nothing.
dist_version() {
    "$1" -c "import importlib.metadata as m; print(m.version('$2'))" 2>/dev/null
}

# Every Python package under examples/plugins, as editable-install arguments. The platform's
# environments carry all of them: a plugin is found by its entry point, so an environment
# without it simply does not see the robot (ADR-0021).
plugin_editables() {
    local dir args=()
    for dir in "$PLUGINS_DIR"/*/; do
        [[ -f "$dir/pyproject.toml" ]] && args+=(-e "${dir%/}")
    done
    if (( ${#args[@]} )); then printf '%s\n' "${args[@]}"; fi
}

# The package directories that ship an installer of their own (ADR-0071).
package_installers() {
    local file
    for file in "$PLUGINS_DIR"/*/install.sh; do
        [[ -f "$file" ]] && basename "$(dirname "$file")"
    done
}
