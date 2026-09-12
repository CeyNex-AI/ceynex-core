# Agriculture & Commodity: 90-second demo runbook

This is a reproducible demonstration of the **Agriculture & Commodity Agent**.
It uses the direct-agent evaluation path, local PostgreSQL and Neo4j, and the
registered local models. It does not make a live LLM request, so every answer
is explicitly marked `degraded=True`; the figures and evidence remain grounded.

## Pre-flight (before an audience arrives)

From the repository root in Windows PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
docker compose ps
.\.venv\Scripts\python.exe -m eval.agriculture_agent_e2e
```

Proceed only if PostgreSQL is healthy, Neo4j is healthy, and the evaluator
reports `passed: 5`, `failed: 0`. The model registry must contain the frozen
versions created by:

```powershell
.\.venv\Scripts\python.exe -m ceynex.models.agriculture.evaluation --register
```

This command creates local, git-ignored model artifacts; it does not modify the
remote repository.

## 90-second script

### 0--10 seconds: scope

> CeyNex provides evidence-backed support for Sri Lankan agriculture exporters.
> This demonstration uses annual Tea Board and FAOSTAT source series, a local
> knowledge graph, and registered forecast models. Every answer shows its
> assumptions and evidence boundary.

### 10--40 seconds: cinnamon forecast

Ask:

> Will cinnamon prices rise or fall over the next two quarters?

Show the `A02` result from `eval/agriculture_agent_e2e`. State the verified
answer exactly:

> The annual cinnamon producer-price model forecasts **10.05 USD/kg for 2025**,
> with an **80% prediction interval of 8.96--11.15 USD/kg**.

Then explain the reliability information, rather than presenting the interval
as a guarantee:

- The selected annual-naive model has **12.05% rolling-origin MAPE**.
- It has **33% coverage (1 of 3 held-out folds)** for nominal 80% intervals.
- Because coverage is below 80%, the agent reduces forecast confidence by
  **0.14**; the verified direct-agent confidence was **0.4395**.
- The question requests quarters, but the model is annual. The response says
  this explicitly instead of inventing quarterly values.

### 40--65 seconds: tea trend and data-quality flag

Ask:

> How have tea export volumes changed over the last five years?

Show the `A04` result. State:

> Tea Board annual export volume fell from **323,012,000 kg in 2011** to
> **257,440,000 kg in 2025**, a **20.3% decline**.

Then show the data-quality treatment, not a reconciled replacement value:

```powershell
.\.venv\Scripts\python.exe -m eval.agriculture_validation
```

The verified comparison reports one material finding: for 2020, Tea Board has
**265,569,000 kg** and partner-aggregated UN Comtrade has **279,710,426.42 kg**
(**5.32%** difference). Say:

> CeyNex preserves both source values and exposes the material difference as a
> `dq_flag`; it does not silently choose one number.

### 65--85 seconds: honest limitations

State both limitations clearly:

> WITS tariff data is deliberately deferred, so it is not represented as
> validated data. The published cinnamon purchasing-price benchmark panel was
> unavailable; the reported cinnamon result uses FAOSTAT's annual producer-price
> fallback and is not a reproduction of that published benchmark.

### 85--90 seconds: close

> The point is not to claim certainty. It is to give an exporter a forecast,
> the evidence behind it, and a clear statement of where the data is weaker.

## Presenter safeguards

- Do not call the district-share question in the demo. The graph contains
  district membership but no sourced numerical share; the direct agent correctly
  refuses to name a largest district.
- Do not quote a model metric without its target and source. The M1 cinnamon
  model is a FAOSTAT producer-price model, not an export-value model.
- Do not describe the local evaluation as deployed-system performance. Local
  raw snapshots and model artifacts are intentionally separate from deployment.
- Do not use the full-orchestrator degraded console output for these two demo
  prompts until it is re-tested. A local rehearsal on 2026-09-11 appended an
  irrelevant `0.0%` district claim during cross-agent composition. The direct
  Agriculture Agent evaluation above is the verified demonstration path.

## Evidence reviewed

- [Agriculture source-series evaluation](EVALUATION.md#3a-m1-agriculture-source-series-evaluation)
- [Cross-source validation](EVALUATION.md#agriculture-cross-source-validation)
- [Agriculture pipeline](DATA_PIPELINE.md)
- [Agriculture data-quality handoff](AGRICULTURE_PIPELINE_HANDOFF.md)
