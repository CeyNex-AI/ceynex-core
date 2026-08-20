"""Registers the real apparel data sources (SRS 3.1.7) — Apparel & Textiles only.

This is the one place that knows which EDB/JAAF files exist and how each is
configured (path, edition/layout, latest year). `ceynex/data/pipeline.py`
imports `APPAREL_SOURCES` rather than knowing about individual connectors —
when a member other than M3 builds the real multi-sector pipeline harness,
this module is what plugs into it; nothing else here should need to change.

Paths are relative to the repo root (matching `ceynex/data/pipeline.py`'s
`STAGING_DIR` convention) and point under `data/raw/<source>/manual/` — a
directory that exists but is gitignored (`data/raw/*/*` in `.gitignore`), so
the *paths* are portable and committed, but the files themselves are not.
Each teammate drops their own copy in place before running the pipeline:

- EDB EPI PDFs are freely downloadable — see the EDB e-service books portal
  linked from `data.md`. Save each edition under its own filename below.
- JAAF pages must be saved manually (Ctrl+S in a browser) — srilankaapparel.com
  disallows automated fetching (robots.txt). See `ceynex/data/connectors/jaaf.py`.

Running without the files in place raises a clear `FileNotFoundError`
(JAAF) or `ValueError` (EDB, no local path and no url) rather than silently
producing nothing.
"""

from pathlib import Path

from ceynex.data.connectors.edb import EDBConnector, EDBReportSource
from ceynex.data.connectors.jaaf import JAAFConnector

_EDB_DIR = Path("data/raw/edb/manual")
_JAAF_DIR = Path("data/raw/jaaf/manual")

EDB_SOURCE = EDBConnector(
    sources=[
        EDBReportSource(
            edition_year=2023,
            latest_year=2023,
            path=str(_EDB_DIR / "export-performance-indicators-of-sri-lanka-2023.pdf"),
            layout="annual",
        ),
        EDBReportSource(
            edition_year=2024,
            latest_year=2024,
            path=str(_EDB_DIR / "export-performance-indicators-of-sri-lanka-2024.pdf"),
            layout="annual",
        ),
        EDBReportSource(
            edition_year=2018,
            latest_year=2018,
            path=str(_EDB_DIR / "export-performance-indicators-2009-2018.pdf"),
            layout="archive",
        ),
    ],
)

JAAF_SOURCE = JAAFConnector(
    annual_exports_path=str(_JAAF_DIR / "Annual Exports – Sri Lanka Apparel.html"),
    market_wise_path=str(_JAAF_DIR / "Market Wise Exports – Sri Lanka Apparel.html"),
)

APPAREL_SOURCES = [EDB_SOURCE, JAAF_SOURCE]
