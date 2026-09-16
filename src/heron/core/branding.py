"""The only place where the platform name is spelled out.

ARCHITECTURE.md, section 17: the name is provisional, so renaming must stay a
mechanical operation instead of a refactoring. Nothing else in the source tree
may write the name as a literal -- not CLI output, not filesystem paths, not
environment variables, not plugin discovery. Everything reads it from here.

Two consequences worth stating, because they are easy to violate by accident:

* Data schemas carry no name at all, only ``schema_version``. A field named
  after the platform would put the name inside user data, where a rename cannot
  reach it.
* Inside the package, imports are relative. An absolute ``from heron.x import y``
  spells the name once per import; a rename would then be a sweep over every
  module instead of a directory rename plus this file.

A test in ``tests/test_branding.py`` enforces both rules against the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

APP_NAME: Final = "heron"
"""Package, CLI command, entry-point prefix, Docker namespace. Lowercase."""

APP_TITLE: Final = "Heron"
"""Human-facing form, used in CLI banners and generated file headers."""

ENV_PREFIX: Final = APP_NAME.upper()
"""Prefix of every environment variable the platform reads."""

ENTRY_POINT_GROUPS: Final = {
    "adapters": f"{APP_NAME}.adapters",
    "sim": f"{APP_NAME}.sim_backends",
    "train": f"{APP_NAME}.train_backends",
    "tasks": f"{APP_NAME}.train_tasks",
    "skills": f"{APP_NAME}.skills",
    "twins": f"{APP_NAME}.twins",
}
"""Plugin groups. A third-party package names one of these in its pyproject."""

DOCKER_NAMESPACE: Final = APP_NAME
"""Image names in profiles are stored bare; the namespace is prepended here."""


def env_var(suffix: str) -> str:
    """Name of an environment variable: ``env_var("HOME")`` -> ``<PREFIX>_HOME``."""
    return f"{ENV_PREFIX}_{suffix.upper()}"


def data_home() -> Path:
    """Root of on-disk platform data.

    Read from ``<PREFIX>_HOME`` when set, otherwise the XDG default. Section 17
    fixes the fallback as ``~/.local/share/<APP_NAME>``.
    """
    override = os.environ.get(env_var("HOME"))
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / APP_NAME


def store_root() -> Path:
    """Root of the artifact store (``<PREFIX>_STORE``, else under ``data_home``)."""
    override = os.environ.get(env_var("STORE"))
    if override:
        return Path(override).expanduser()
    return data_home() / "store"


def registry_root() -> Path:
    """Root of the robot profile registry (``<PREFIX>_REGISTRY``)."""
    override = os.environ.get(env_var("REGISTRY"))
    if override:
        return Path(override).expanduser()
    return data_home() / "registry"


def image_name(bare: str) -> str:
    """Qualify a profile's bare image name with the platform's Docker namespace.

    Profiles store ``robot-arm:latest``; the namespace is not part of the data,
    because a rename must not have to rewrite user profiles.
    """
    return f"{DOCKER_NAMESPACE}/{bare}"
