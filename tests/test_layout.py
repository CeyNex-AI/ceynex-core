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


def test_config_is_found_from_an_arbitrary_working_directory(tmp_path, monkeypatch):
    """Regression: the container installs the package to site-packages and the
    config to /app, so a repo-relative constant resolved to a path that does not
    exist and the API crash-looped on startup."""
    from ceynex.settings import config_dir, load_config

    monkeypatch.chdir(tmp_path)
    load_config.cache_clear()
    assert (config_dir() / "llm.yaml").is_file()
    assert load_config("llm")["models"]["merge"]["model"]


def test_an_explicit_config_dir_wins(tmp_path, monkeypatch):
    import yaml

    from ceynex.settings import config_dir, load_config

    custom = tmp_path / "custom-config"
    custom.mkdir()
    (custom / "llm.yaml").write_text(yaml.safe_dump({"provider": "test"}), encoding="utf-8")

    monkeypatch.setenv("CEYNEX_CONFIG_DIR", str(custom))
    load_config.cache_clear()
    try:
        assert config_dir() == custom
        assert load_config("llm")["provider"] == "test"
    finally:
        load_config.cache_clear()


# --- packaged data files -------------------------------------------------
#
# Three separate deployment failures came from the same mistake: a file the code
# reads at runtime resolved through a source-relative path, which exists in a
# checkout and does not exist once the package is installed. Everything passed
# locally; the container died on startup each time. These assert the files are
# reachable the way the installed package reaches them.

PACKAGED_DATA = [
    ("ceynex.data", "reference/countries.csv"),
    ("ceynex.data", "reference/hs_codes.csv"),
    ("ceynex.data", "reference/partner_aggregates.csv"),
    ("ceynex.data", "reference/partner_aliases.csv"),
    ("ceynex.data", "reference/trade_agreements.csv"),
    ("ceynex.data", "reference/trade_agreement_coverage.csv"),
    ("ceynex.contracts", "schema/schema.sql"),
    ("ceynex.contracts", "schema/schema.cypher"),
]


@pytest.mark.parametrize(("package", "resource"), PACKAGED_DATA, ids=[r for _, r in PACKAGED_DATA])
def test_runtime_data_files_are_reachable_as_package_resources(package, resource):
    from importlib.resources import files

    target = files(package)
    for part in resource.split("/"):
        target = target / part
    assert target.is_file(), (
        f"{package}:{resource} is not reachable through importlib.resources. "
        "Add it to [tool.setuptools.package-data] — without that it works in a "
        "checkout and disappears in the container."
    )


def test_no_runtime_data_is_read_through_a_source_relative_path():
    """`Path(__file__).parent / "reference"` is the shape that keeps breaking."""
    import re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent / "ceynex"
    offenders = []
    pattern = re.compile(r'Path\(__file__\)\.parent[\w.\s]*/\s*"(reference|schema|config)"')
    for source in root.rglob("*.py"):
        if pattern.search(source.read_text(encoding="utf-8")):
            offenders.append(str(source.relative_to(root.parent)))
    assert not offenders, f"use importlib.resources instead in: {offenders}"
