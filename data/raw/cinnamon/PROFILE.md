# Cinnamon annual fallback: curated source workbook

The active workbook is `CeyNex_project/data/raw/cinnamon/2026-08-15/
cinnamon_annual_fallback_2011_2025.xlsx`, outside `ceynex-core` and excluded
from Git. Run `python tools/audit_agriculture_source_workbooks.py` to check
the selected workbook, its row-level attribution, and the connector mapping.

The workbook stages 85 FAOSTAT annual observations and 15 DEA/EAC observations.
Only the five DEA/EAC annual total export-volume rows for 2013-2017 map to
`fact_trade` under source ID `CINNAMON`. FAOSTAT values remain separate and
are not overwritten. The workbook's source and limits sheets explain that
these data are not the unavailable grade/location/date purchasing-price panel.
