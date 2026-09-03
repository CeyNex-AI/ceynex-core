"""The news sidecar — see docs/ARCHITECTURE_DELTA.md D11.

Current-events context alongside an answer, from GDELT DOC 2.0. Deliberately a
sidecar and not a data source: nothing in here may become `Evidence`, be cited by
an agent, or appear in a printed report. `schema.py` explains why that line is
drawn where it is.

This package needs `__init__.py`. The namespace-package rule in CLAUDE.md bans
only the top-level `ceynex/__init__.py`; every subpackage has one.
"""
