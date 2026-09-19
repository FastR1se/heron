# Windows: the development environment -- the same set CI installs (.github/workflows/ci.yml):
# the platform with its test groups and every package of examples/plugins, editable (ADR-0071).
# No simulator, ROS 2 or robot SDK here: those live on the Linux machine (scripts/install.sh).
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev-setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\dev-setup.ps1 -Descriptions
#
# Caches and the interpreter stay on D (CLAUDE.md): the repository's .uv-cache and .uv-python,
# both ignored by git, unless UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR are already set.

param(
    [switch]$Descriptions   # also fetch the vendor descriptions the tests read (large downloads)
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $repo '.uv-cache' }
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $repo '.uv-python' }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv не найден: https://docs.astral.sh/uv/getting-started/installation/'
}

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    # 3.11: the lowest the platform supports, the one CI and mypy check against.
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv не создал окружение' }
}

$editables = @('-e', '.[dev,record,onnx,testing]')
Get-ChildItem (Join-Path $repo 'examples\plugins') -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName 'pyproject.toml')
} | ForEach-Object { $editables += @('-e', $_.FullName) }

# `uv pip install`, not `uv sync`: a sync removes whatever pyproject.toml does not list, the
# robot packages among it.
uv pip install --python $python @editables
if ($LASTEXITCODE -ne 0) { throw 'uv pip install не прошёл' }

if ($Descriptions) {
    Get-ChildItem (Join-Path $repo 'examples\robots') -Directory | ForEach-Object {
        $fetch = Join-Path $_.FullName 'fetch_description.py'
        if (Test-Path $fetch) {
            & $python $fetch
            if ($LASTEXITCODE -ne 0) { throw "описание не загрузилось: $fetch" }
        }
    }
}

# The platform's name is read from pyproject.toml, not spelled here (CLAUDE.md).
$app = (Select-String -Path (Join-Path $repo 'pyproject.toml') -Pattern '^name = "(.+)"$' |
    Select-Object -First 1).Matches[0].Groups[1].Value
& (Join-Path $repo ".venv\Scripts\$app.exe") version
if ($LASTEXITCODE -ne 0) { throw "$app не запускается из .venv" }
Write-Host 'Окружение разработки готово: .venv\Scripts\python.exe -m pytest'
