"""The RC adapter held to the platform's contract suite, from this package's tests (ADR-0056).

The package began as the stub ``profile new robopro_rc --control-mode command`` writes,
and this file is that stub's test with the harness filled in: the path a user takes.
"""

from __future__ import annotations

from robopro_rc import testing

from heron.testing.adapter_contract import AdapterContract, Harness


class TestRoboproRcAdapter(AdapterContract):
    harness = Harness(
        plugin=testing.PLUGIN,
        make=testing.recording_adapter,
        profile=testing.profile,
        config=dict(testing.LINK_PARAMS),
        prime=testing.prime,
        finish=testing.finish,
    )
