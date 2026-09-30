# Tea annual production and exports: curated source workbook

The active workbook is `CeyNex_project/data/raw/tea_board/2026-08-15/
tea_annual_production_exports_2011_2025.xlsx`, outside `ceynex-core` and
excluded from Git. Run `python tools/audit_agriculture_source_workbooks.py` to
check the selected workbook, its attribution, and the connector mapping.

The workbook covers 2011-2025. Its 2011-2016 annual export totals cite the
Central Bank of Sri Lanka, while 2017-2025 cite Tea Exporters Association
tables. Its `Sources and Notes` sheet records publisher URLs and limitations.
The `TEA_BOARD` connector ID describes the curated tea data pathway; it does
not mean every row was published by the Tea Board. Tea Board production PDFs
are retained as supporting material. The connector stages production and
exports but maps only total export volume to `fact_trade`.
