-- Implements SRS 3.1.7, 3.1.8 and team contract 4.2 — the unified time-indexed dataset.
-- FROZEN CONTRACT. Changes require 3-way approval (M1, M2, M3).
-- Loaded automatically by docker compose on first start of the postgres volume.

CREATE TABLE IF NOT EXISTS dim_country (
  iso3  CHAR(3) PRIMARY KEY,
  m49   SMALLINT NOT NULL UNIQUE,
  name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_hs (
  hs_code     TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  sector      TEXT NOT NULL        -- agriculture | apparel
);

CREATE TABLE IF NOT EXISTS fact_trade (
  record_id        BIGSERIAL PRIMARY KEY,
  source_id        TEXT NOT NULL,          -- UN_COMTRADE, WITS, FAOSTAT, CBSL, JAAF, EDB
  sector           TEXT NOT NULL,          -- agriculture | apparel
  item             TEXT NOT NULL,          -- tea | cinnamon | rubber | coconut | knit | woven
  hs_code          TEXT,                   -- '0902', '0906', '61', '62'
  reporter_iso3    CHAR(3) NOT NULL,       -- 'LKA'
  reporter_m49     SMALLINT NOT NULL,      -- 144
  partner_iso3     CHAR(3),                -- NULL = world
  partner_m49      SMALLINT,
  period_start     DATE NOT NULL,
  period_end       DATE NOT NULL,
  frequency        TEXT NOT NULL,          -- D | W | M | Q | A
  export_volume    NUMERIC,
  volume_unit      TEXT,                   -- 'kg', 'tonne', 'pcs'
  export_value_usd NUMERIC,
  price            NUMERIC,
  price_unit       TEXT,                   -- 'USD/kg', 'LKR/kg'
  fx_usd_lkr       NUMERIC,
  ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  source_hash      TEXT NOT NULL,          -- idempotency key
  UNIQUE (source_id, item, hs_code, reporter_iso3, partner_iso3, period_start, frequency)
);

CREATE INDEX IF NOT EXISTS ix_fact_trade_item_period ON fact_trade (item, period_start);
CREATE INDEX IF NOT EXISTS ix_fact_trade_sector      ON fact_trade (sector);
CREATE INDEX IF NOT EXISTS ix_fact_trade_partner     ON fact_trade (partner_iso3);
CREATE INDEX IF NOT EXISTS ix_fact_trade_hash        ON fact_trade (source_hash);

-- Cross-validation discrepancies: FLAG, never drop (SRS 3.1.8).
-- Both conflicting source rows stay in fact_trade; this table records that they disagree.
CREATE TABLE IF NOT EXISTS dq_flag (
  flag_id      BIGSERIAL PRIMARY KEY,
  item         TEXT,
  hs_code      TEXT,
  partner_iso3 CHAR(3),
  period_start DATE,
  metric       TEXT,
  source_a     TEXT,
  value_a      NUMERIC,
  source_b     TEXT,
  value_b      NUMERIC,
  pct_diff     NUMERIC,
  severity     TEXT,                 -- minor <5% | material 5-20% | severe >20%
  detected_at  TIMESTAMPTZ DEFAULT now(),
  resolved     BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS ix_dq_flag_item     ON dq_flag (item, period_start);
CREATE INDEX IF NOT EXISTS ix_dq_flag_severity ON dq_flag (severity);

-- Pipeline observability; backs GET /api/admin/pipeline/status (SRS 3.5.4).
CREATE TABLE IF NOT EXISTS ingest_run (
  run_id       BIGSERIAL PRIMARY KEY,
  source_id    TEXT NOT NULL,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at  TIMESTAMPTZ,
  status       TEXT NOT NULL,        -- running | success | failed
  rows_written INTEGER DEFAULT 0,
  error        TEXT
);

CREATE INDEX IF NOT EXISTS ix_ingest_run_source ON ingest_run (source_id, started_at DESC);
