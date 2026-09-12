"""Generators: from a profile and a scenario to the artifacts a world needs.

ADR-0009 draws the line these modules live on. Kinematics is an input -- the
platform reads a description, it never writes one. What is derivable from the
profile is derived: control interfaces from the contract, controller class from
the control mode, physics step from the state rates, namespaces and frames from
the scenario ids.

ADR-0010 fixes the form: description rather than code, no template engine, one
output directory, byte-identical output for identical input. That last property
is what turns "are these artifacts current?" into a question with an answer.
"""

from .bundle import build, build_profile, scenario_for_profile, scenario_instances
from .common import (
    Bundle,
    Difference,
    GeneratedFile,
    GeneratorError,
    Instance,
    resolve_description,
)

__all__ = [
    "Bundle",
    "Difference",
    "GeneratedFile",
    "GeneratorError",
    "Instance",
    "build",
    "build_profile",
    "resolve_description",
    "scenario_for_profile",
    "scenario_instances",
]
