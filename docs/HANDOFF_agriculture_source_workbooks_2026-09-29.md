# Agriculture source workbook verification — 2026-09-29

## Selected files and provenance

`ceynex.data.pipeline._agriculture_raw_dir()` selects the workspace-level
`CeyNex_project/data/raw/` directory before the repository's own `data/raw/`.
The active files are its `tea_board/2026-08-15/tea_annual_production_exports_2011_2025.xlsx`
and `cinnamon/2026-08-15/cinnamon_annual_fallback_2011_2025.xlsx`.
Both are kept outside Git. Run the read-only audit with:

```powershell
.\.venv\Scripts\python.exe tools\audit_agriculture_source_workbooks.py
```

| Connector ID | Actual workbook coverage | Publisher attribution | Staged / mapped rows |
| --- | --- | --- | ---: |
| `TEA_BOARD` | Annual tea production and exports, 2011–2025 | Central Bank of Sri Lanka totals for 2011–2016; Tea Exporters Association tables for 2017–2025 | 102 / 15 |
| `CINNAMON` | FAOSTAT annual series, 2011–2025 where published; DEA/EAC observations, 2013–2017 | Per-row `source`, `source_file`, and `source_url` | 100 / 5 |

`TEA_BOARD` is a pipeline source ID, not the publisher of every row. The
workbook's `Sources and Notes` sheet records the original URLs. The Tea Board
production PDFs are supporting material; production is staged but not mapped
to the frozen `fact_trade` schema. Cinnamon's FAOSTAT values remain separate;
only DEA/EAC annual export volumes map through `CINNAMON`.

## Local checks

The user ran ingestion twice. Both runs ended with 15 `TEA_BOARD` rows, five
`CINNAMON` rows, and 5,565 total `fact_trade` rows. The read-only workbook
audit checked annual-panel versus connector-sheet totals, publisher metadata,
per-row cinnamon URLs, and mapped row counts.

The 2026-09-29 cross-source rerun found 12 comparable annual pairs: 11 minor,
one material, zero severe. The material 2020 tea finding (265,569,000 kg in
the curated tea workbook versus 279,710,426.42 kg in UN Comtrade, 5.32%) was
already present and unresolved as one `dq_flag`; zero new flags were written.
The five-question direct-agent evaluation passed 5/5 in deterministic degraded
mode, with a mean of 2.2 evidence records per question. These are local tests,
not production deployment evidence.

## Deployment and presentation boundary

The ignored active workbooks must be transferred by the team's approved VM
raw-data procedure. After ingestion on the VM, repeat the workbook audit,
source row-count check, cross-source validation, and five-question agent
evaluation. Record the VM snapshot hashes and results before claiming live
coverage. The tea trend's 2011 and 2025 values are traceable in the active
workbook, but they describe the full 14-year span, not a five-year change.
