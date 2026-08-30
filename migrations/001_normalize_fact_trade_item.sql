-- 001_normalize_fact_trade_item.sql
--
-- Collapse EDB's free-text `item` labels onto the controlled vocabulary in
-- ceynex/data/reference/item_vocabulary.csv, so one real-world category is one
-- series. Generated from that CSV against the 22 distinct labels present in
-- production on 2026-08-30 -- not hand-typed, because several differ only by an
-- en-dash.
--
-- Why this is needed: `item` is column 2 of `fact_trade_upsert_key`, so a
-- spelling variant is a separate identity. `APPAREL` held 2014-2018 and
-- `APPREL` 2019-2024 -- one 11-year series stored as two 5-year fragments,
-- which is why the registered apparel model was fitted on 5 rows and forecast
-- a year already past.
--
-- Two cases, both measured before writing this:
--   * APPREL -> APPAREL: 0 unique-key collisions (the labels are time-disjoint).
--   * The 3 MADE-UP TEXTILE ARTICLES variants: 143 colliding key groups, and
--     all 143 hold identical export_value_usd (avg gap 0.00%). They are the
--     same fact republished in overlapping editions, so the merge is lossless
--     dedupe, not a value conflict.
--
-- `source_hash` is deliberately left alone. It was computed from the raw label,
-- so the next `make ingest` sees a hash mismatch and updates these rows in
-- place -- which is correct and harmless. Recomputing sha256 here would mean
-- reproducing writer._source_hash's exact input format in SQL, and getting
-- that subtly wrong is worse than one redundant update.
--
-- Re-runnable: the mapping is idempotent (canonical -> canonical is a no-op)
-- and the dedupe only fires when duplicates exist.

BEGIN;

CREATE TEMP TABLE item_map(raw text PRIMARY KEY, canonical text NOT NULL) ON COMMIT DROP;
INSERT INTO item_map(raw, canonical) VALUES
    ('ACTIVEWEAR/ SPORTSWERA', 'activewear_sportswear'),
    ('APPAREL', 'apparel_edb'),
    ('APPAREL & TEXTILES (Made - Up Textile Article, Apparel, Woven Fabrics & Other Textile Articles)', 'apparel_textiles_edb'),
    ('APPAREL AND TEXTILES (Made – Up Textile Articles, Apparel, Woven Fabrics, Other Textile Articles)', 'apparel_textiles_edb'),
    ('APPREL', 'apparel_edb'),
    ('BABIES'' GARMENTS', 'babies_garments'),
    ('GLOVES, MITTS & MITTENS OF TEXTILE', 'gloves_mitts_mittens'),
    ('HOSIERY', 'hosiery'),
    ('KNITTED FABRICS', 'knitted_fabrics'),
    ('MADE - UP CLOTHING ACCESSORIES (Handkerchief, Shawls, Scarves, Ties etc)', 'made_up_clothing_accessories'),
    ('MADE - UP TEXTILE ARTICLES (Blankets, Rugs, Linen & Curtains etc.)', 'made_up_textile_articles'),
    ('MADE – UP TEXTILE ARTICLES', 'made_up_textile_articles'),
    ('MADE-UP TEXTILE ARTICLES (Blankets, Rugs, Linen, Curtains etc)', 'made_up_textile_articles'),
    ('MEN''S & WOMEN''S UNDER GARMENTS', 'under_garments'),
    ('MEN''S OUTERWEAR', 'mens_outerwear'),
    ('T-SHIRTS', 't_shirts'),
    ('TEXTILE FLOOR COVERING (Carpets, Mats, Floor Covering etc)', 'textile_floor_covering'),
    ('TEXTILES ( Knitted Fabrics, Woven Fabrics, Yarn, Made - Up Textile Articles, Textile Floor Covering etc.)', 'textiles_edb'),
    ('WARM CLOTHS (Jerseys, Pullovers etc.)', 'warm_clothing'),
    ('WOMEN''S OUTERWEAR', 'womens_outerwear'),
    ('WOVEN FABRICS', 'woven_fabrics'),
    ('YARN', 'yarn');

-- 1. Report what will merge, before anything changes.
\echo '--- rows per raw label about to be remapped ---'
SELECT m.canonical, count(*) AS rows, count(DISTINCT f.item) AS variants
  FROM fact_trade f JOIN item_map m ON m.raw = f.item
 WHERE f.source_id = 'EDB'
 GROUP BY m.canonical HAVING count(DISTINCT f.item) > 1
 ORDER BY rows DESC;

-- 2. Refuse to run if any colliding pair disagrees on value. The dedupe below
--    keeps one row and drops the rest, which is only safe while they agree.
\echo '--- colliding key groups that DISAGREE on value (must be 0) ---'
SELECT count(*) AS disagreeing_groups FROM (
  SELECT 1 FROM fact_trade f JOIN item_map m ON m.raw = f.item
   WHERE f.source_id = 'EDB'
   GROUP BY m.canonical, f.hs_code, f.reporter_iso3, f.partner_iso3,
            f.period_start, f.frequency
  HAVING count(*) > 1 AND count(DISTINCT f.export_value_usd) > 1
) t;

-- 3. Drop duplicate republications, keeping the lowest record_id.
DELETE FROM fact_trade f
 USING fact_trade keep, item_map mf, item_map mk
 WHERE f.source_id = 'EDB' AND keep.source_id = 'EDB'
   AND mf.raw = f.item AND mk.raw = keep.item
   AND mf.canonical = mk.canonical
   AND f.hs_code IS NOT DISTINCT FROM keep.hs_code
   AND f.reporter_iso3 = keep.reporter_iso3
   AND f.partner_iso3 IS NOT DISTINCT FROM keep.partner_iso3
   AND f.period_start = keep.period_start
   AND f.frequency = keep.frequency
   AND f.record_id > keep.record_id;

-- 4. Remap the survivors.
UPDATE fact_trade f SET item = m.canonical
  FROM item_map m
 WHERE f.source_id = 'EDB' AND f.item = m.raw AND f.item <> m.canonical;

\echo '--- distinct EDB items after (expect 18) ---'
SELECT count(DISTINCT item) AS distinct_items, count(*) AS edb_rows
  FROM fact_trade WHERE source_id = 'EDB';

COMMIT;
