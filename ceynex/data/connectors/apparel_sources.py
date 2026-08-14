"""Registers the real apparel data sources (SRS 3.1.7) — Apparel & Textiles only.

This is the one place that knows which EDB/JAAF files exist and how each is
configured (path, edition/layout, latest year). `ceynex/data/pipeline.py`
imports `APPAREL_SOURCES` rather than knowing about individual connectors —
when a member other than M3 builds the real multi-sector pipeline harness,
this module is what plugs into it; nothing else here should need to change.

Paths point at files under the local user's Downloads folder for now (not
committed, not portable to another machine) — replace with `data/raw/...`
committed copies or a shared drop location once the team has one.
"""

from ceynex.data.connectors.edb import EDBConnector, EDBReportSource
from ceynex.data.connectors.jaaf import JAAFConnector

EDB_SOURCE = EDBConnector(
    sources=[
        EDBReportSource(
            edition_year=2023,
            latest_year=2023,
            path=r"C:\Users\kndhi\Downloads\export-performance-indicators-of-sri-lanka-2023.pdf",
            layout="annual",
        ),
        EDBReportSource(
            edition_year=2024,
            latest_year=2024,
            path=r"C:\Users\kndhi\Downloads\export-performance-indicators-of-sri-lanka-2024.pdf",
            layout="annual",
        ),
        EDBReportSource(
            edition_year=2018,
            latest_year=2018,
            path=r"C:\Users\kndhi\Downloads\export-performance-indicators-2009-2018.pdf",
            layout="archive",
        ),
    ],
)

JAAF_SOURCE = JAAFConnector(
    annual_exports_path=r"C:\Users\kndhi\Downloads\Annual Exports – Sri Lanka Apparel.html",
    market_wise_path=r"C:\Users\kndhi\Downloads\Market Wise Exports – Sri Lanka Apparel.html",
)

APPAREL_SOURCES = [EDB_SOURCE, JAAF_SOURCE]
