# Agriculture & Commodity contribution: final report draft

The Agriculture & Commodity workstream assembled Sri Lankan tea and cinnamon
series, mapped compatible annual export volumes into `fact_trade`, retained
production and other incompatible measurements in staging, and integrated
source-aware price and volume evidence into the agriculture agent. The curated
tea workbook covers 2011–2025. Its early annual totals cite Central Bank
tables; the later totals cite Tea Exporters Association tables. The cinnamon
fallback stages FAOSTAT and DEA/EAC observations separately and maps only
five DEA/EAC annual export totals (2013–2017). Publisher and file provenance
remain in the ignored workbooks and their source profiles.

The selected annual-naive models were evaluated with three expanding-window
held-out years. Tea export-volume MAPE was 3.18%, with 3/3 interval coverage;
cinnamon FAOSTAT producer-price MAPE was 12.05%, with only 1/3 coverage for a
nominal 80% interval. The agent penalizes the latter's confidence by 0.14 and
states the limitation. This cinnamon price series is a fallback, not a
reproduction of the unavailable published purchasing-price benchmark.

Local cross-source validation compared 12 overlapping tea/cinnamon annual
export-volume pairs with UN Comtrade. Eleven differences were minor and one
was material: the 2020 curated tea total of 265,569,000 kg differed from the
partner-aggregated UN Comtrade total of 279,710,426.42 kg by 5.32%. Both
values were preserved; one unresolved material `dq_flag` records the conflict.
The five direct agriculture-agent questions passed in deterministic degraded
mode, including honest refusals where district shares and tea-to-rubber
substitution data were absent. These results describe the local system.

WITS tariff ingestion was deliberately deferred. Real exporter or sample-user
feedback has not been documented, so no usability outcome is claimed. Live
deployment of the two curated workbooks remains subject to the team's VM
handoff and post-deployment verification.
