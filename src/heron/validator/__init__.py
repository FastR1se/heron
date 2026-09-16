"""The validator: the component the platform's novelty sits in.

Invariant 6 -- "a mismatch between environments is a validator error, not
strange behaviour of a robot" -- is implemented here and nowhere else.
"""

from .policy import check_contracts, check_policy_against_profile
from .report import Finding, Report, error, warning
from .rules import check_profile, check_scenario, check_skill_against_profile

__all__ = [
    "Finding",
    "Report",
    "check_contracts",
    "check_policy_against_profile",
    "check_profile",
    "check_scenario",
    "check_skill_against_profile",
    "error",
    "warning",
]
