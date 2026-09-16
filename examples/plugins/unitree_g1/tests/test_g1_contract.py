"""The DDS adapter held to the platform's contract suite, from this package's tests (ADR-0056).

The platform does not list this package anywhere: a package holds its own adapter
to the suite the way a user's package does, by subclassing it with a harness.
"""

from __future__ import annotations

from unitree_g1 import testing

from heron.testing.adapter_contract import AdapterContract, Harness


class TestG1DdsAdapter(AdapterContract):
    harness = Harness(
        plugin=testing.PLUGIN,
        make=testing.recording_adapter,
        profile=testing.profile_23dof,
        config=dict(testing.LINK_PARAMS),
        prime=testing.prime,
    )
