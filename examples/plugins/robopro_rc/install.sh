#!/usr/bin/env bash
#
# The environment of the RC package on a Linux machine (ADR-0071). Run by the platform's
# installer, never directly:
#
#   API_DIR=~/robopro_api scripts/install.sh install robopro_rc
#   scripts/install.sh verify robopro_rc
#
# venv  $STACK_ROOT/robopro_rc/venv (python3.12; the vendor's API 1.4.1 supports 3.10-3.13):
#       the platform with records, this package and the vendor's API archive. The archive is
#       not on an index and carries no licence to be vendored (README): a human downloads it
#       and names the directory in API_DIR. Without it the part is skipped, not failed.
#
# `source scripts/activate.sh robopro_rc` -- where `run --target real` for an RC arm runs.

set -euo pipefail

: "${COMMON_SH:?установщик пакета запускается платформой: scripts/install.sh install robopro_rc}"
# shellcheck source=../../../scripts/lib/common.sh
source "$COMMON_SH"

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_NAME="$(basename "$PKG_DIR")"
PKG_ROOT="$STACK_ROOT/$PKG_NAME"

API_VERSION="1.4.1"                  # ADR-0062: the adapter reads this version's RTD receiver
API_DIR="${API_DIR:-$HOME/robopro_api}"
PYTHON="python3.12"

VENV="$PKG_ROOT/venv"
PY="$VENV/bin/python"

has_venv() { [[ -x "$PY" ]] && [[ "$(dist_version "$PY" API)" == "$API_VERSION" ]] &&
             "$PY" -c "import $APP, robopro_rc" >/dev/null 2>&1; }

install_venv() {
    step "venv: окружение стенда $VENV"
    if [[ ! -d "$API_DIR/API" ]]; then
        warn "нет архива API производителя: $API_DIR/API"
        warn "распакуйте архив API «Робопро» $API_VERSION и укажите каталог:  API_DIR=<каталог> scripts/install.sh install $PKG_NAME"
        warn "без него пакет проверяется тестами (записывающий канал), но к роботу не подключится"
        exit 3
    fi
    apt_install python3.12-venv
    [[ -x "$PY" ]] || run_clean "'$PYTHON' -m venv '$VENV'"
    run_clean "'$PY' -m pip install -q '$API_DIR'"
    sync_venv
}

sync_venv() {
    [[ -x "$PY" ]] || return 0
    info "платформа (группа record) и пакет $PKG_NAME в $VENV"
    run_clean "'$PY' -m pip install -q -e '$REPO_ROOT[record]' && '$PY' -m pip install -q --no-deps -e '$PKG_DIR'"
    (( DRY_RUN )) && return 0
    [[ "$(dist_version "$PY" API)" == "$API_VERSION" ]] || die \
        "в $VENV API $(dist_version "$PY" API || echo 'не установлен'), нужен $API_VERSION" \
        "Адаптер написан по исходнику этой версии (ADR-0062). Архив нужной версии:" \
        "    API_DIR=<каталог> scripts/install.sh install $PKG_NAME"
    ok "окружение стенда: API $API_VERSION, платформа, пакет"
}

cmd_verify() {
    has_venv || { warn "нет окружения стенда (часть venv) или API не $API_VERSION"; return 1; }
    run_clean "'$PY' -c 'import API.rc_api, numpy, scipy, robopro_rc.adapter'" ||
        { warn "API производителя не импортируется"; return 1; }
    ok "API $API_VERSION и пакет импортируются; связь с роботом — scripts/runs/rc5-stand.sh env probe"
}

cmd_status() {
    if has_venv; then
        echo "venv|есть|$VENV, API $(dist_version "$PY" API)"
    else
        echo "venv|нет|$VENV (архив API: API_DIR=$API_DIR)"
        return 1
    fi
}

case "${1:-}" in
    install) shift
             [[ $# -eq 0 || "$*" == venv ]] || die "часть пакета $PKG_NAME одна: venv"
             run mkdir -p "$PKG_ROOT"
             install_venv ;;
    sync)    sync_venv ;;
    verify)  cmd_verify ;;
    status)  cmd_status ;;
    *)       die "команда установщика пакета: install | sync | verify | status" ;;
esac
