# M1 agriculture-model deployment handoff

This is the operational handoff for M1's two registered annual forecasts:

- Sri Lanka tea export volume — `agriculture/tea/export_volume`, unit `kg`.
- Sri Lanka cinnamon producer price — `agriculture/cinnamon/producer_price`,
  unit `USD/kg`.

It complements the data-ingestion handoff in
[AGRICULTURE_PIPELINE_HANDOFF.md](AGRICULTURE_PIPELINE_HANDOFF.md).  It does
not change the frozen database schema or require an ingestion run.

## Why registration is a deployment step

The repository deliberately excludes both input data and executable model
artifacts:

- `/models/` is git-ignored.  It contains `model.pkl` files, which are Python
  pickles and must only be loaded when produced by this trusted project.
- `data/raw/<source>/<YYYY-MM-DD>/` contents are git-ignored.  Only connector
  `PROFILE.md` files are versioned, not source data.

Therefore a Git checkout alone cannot serve the M1 models.  The raw snapshots
are required **at registration or retraining time**; the persistent model
directory is required **at forecast-serving time**.  A serving instance does
not need to read the raw snapshots after the artifacts have been created, but
they must be retained in controlled storage so registration is reproducible.

## Preconditions

1. Deploy a clean checkout that includes model-registration commit
   `a5a0434` and the later target-specific forecast-agent commit.  Do not run
   registration from an uncommitted working tree: the command intentionally
   refuses it, because its recorded Git SHA must identify the code that created
   the artifact.
2. Install the project dependencies, including `pandas`, `openpyxl`, and the
   project model dependencies.
3. Place the dated raw snapshots in controlled storage.  The root must contain
   at least:

   ```text
   <raw-root>/
   ├── tea_board/<YYYY-MM-DD>/tea_annual_production_exports_2011_2025.xlsx
   └── faostat/<YYYY-MM-DD>/FAOSTAT producer prices.csv
   ```

   The evaluator selects the latest date-shaped snapshot directory.  Do not
   alter the input workbook/CSV after checksum verification.
4. Select a persistent, application-readable model directory.  Do not use a
   temporary build directory that will disappear when the API restarts.

## Registration procedure

Run the following from the deployed `ceynex-core` checkout.  Paths below are
examples; use the deployment platform's persistent volume paths instead.

```bash
export CEYNEX_AGRICULTURE_RAW_DIR=/srv/ceynex/raw
export CEYNEX_MODELS_DIR=/srv/ceynex/models
mkdir -p "$CEYNEX_MODELS_DIR"

git status --short
git rev-parse HEAD
python -m ceynex.models.agriculture.evaluation --register
```

`git status --short` must print nothing.  The registration JSON must contain
two entries with the deployed SHA in `git_sha`:

| Item | Target | Required source | Unit | Training window |
|---|---|---|---|---|
| tea | `export_volume` | Sri Lanka Tea Board annual total exports | kg | 2011–2025, 15 rows |
| cinnamon | `producer_price` | FAOSTAT annual Sri Lanka cinnamon producer price | USD/kg | 1991–2024, 34 rows |

The cinnamon registration must retain this limitation in `notes`:
`Liyanage/Silva purchasing-price data unavailable; FAOSTAT annual fallback used.`
It is not a reproduction of the unavailable Liyanage/Silva benchmark.

## Acceptance check

Before directing traffic to the forecast API, run:

```bash
python -c "from ceynex.models.registry import load; tea=load('agriculture','tea','export_volume'); cinnamon=load('agriculture','cinnamon','producer_price'); print('Tea:', tea.predict(1)); print('Cinnamon:', cinnamon.predict(1))"
```

Accept the release only if both outputs include `point`, `lower`, and `upper`,
with `lower <= point <= upper`, and have these exact units:

- tea: `kg`
- cinnamon: `USD/kg`

Then perform one API/query smoke test for each explicit request:

```text
forecast tea export volume for the next year
forecast cinnamon producer price for the next year
```

The tea result must remain in `kg`; the cinnamon result must remain in
`USD/kg`.  A generic request such as `forecast cinnamon exports` is an
export-value question and must not select the producer-price model.

## Failure and rollback

- If registration fails, do not create placeholder pickles or copy a model
  artifact from an untrusted machine.  Correct the raw-path, dependency, or
  clean-Git condition and rerun from the intended commit.
- If the acceptance check fails, keep the previous persistent model directory
  configured for the service, investigate off-line, and do not direct forecast
  traffic to the new artifacts.
- To reproduce a known release, check out its recorded `git_sha`, use the
  corresponding checksum-verified raw snapshots, and register into a new
  persistent model directory.  Switch `CEYNEX_MODELS_DIR` only after the same
  acceptance checks pass.
