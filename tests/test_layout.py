"""Guards the two-distribution layout that everything else assumes.

ceynex-core and ceynex-contracts both contribute to one `ceynex` namespace
package. That works until someone adds a `ceynex/__init__.py` back — at which
point one distribution shadows the other and every teammate's imports break with
a message that points nowhere near the cause. These tests make it point here.
"""

from pathlib import Path

import pytest


def test_contracts_are_installed():
    """`make install` pulls ceynex-contracts from the sibling checkout first."""
    try:
        from ceynex.contracts import ALL_AGENTS, AgentState  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only on a broken install
        pytest.fail(
            "ceynex.contracts is not importable. Run `make install`, which installs "
            f"ceynex-contracts from ../ceynex-contracts before this package. ({exc})"
        )


def test_ceynex_is_a_namespace_package():
    import ceynex

    offenders = [Path(p) / "__init__.py" for p in ceynex.__path__]
    existing = [p for p in offenders if p.exists()]
    assert not existing, (
        f"delete {existing} — `ceynex` must stay an implicit namespace package so "
        "ceynex-core and ceynex-contracts can both contribute subpackages"
    )


def test_both_distributions_contribute_to_the_namespace():
    """A single __path__ entry means one distribution is shadowing the other."""
    import ceynex
    import ceynex.contracts
    import ceynex.orchestrator

    assert Path(ceynex.contracts.__file__).parent.name == "contracts"
    assert Path(ceynex.orchestrator.__file__).parent.name == "orchestrator"
    assert ceynex.__path__, "namespace package resolved to no search path at all"


def test_core_does_not_ship_a_copy_of_the_contracts():
    """The frozen package lives in one repo. A local copy drifts and nobody notices."""
    stray = Path(__file__).resolve().parent.parent / "ceynex" / "contracts"
    assert not stray.exists(), (
        f"{stray} is a second copy of the frozen contracts — delete it and rely on "
        "the ceynex-contracts distribution"
    )
