# Looking inside the databases

How to connect to PostgreSQL and Neo4j, see what is actually in them, and work
out what is wrong when something looks wrong.

**Every query in this document was run against the live system on 2026-08-19**
and the outputs shown are real. If you run one and get a different answer, that
is information — either the data moved or something broke.

> The row counts below (4,625 `fact_trade` rows, 4,625 `EXPORTS_TO` edges,
> 2015–2024) are that run's numbers and have since moved: the deployed backend
> was re-ingested and covers **2015–2025**, verified live 2026-09-03. Expect
> larger counts there, and a fresh local `make ingest` to land somewhere between.
> **What has to hold is the relationship between them, not the value** — the
> `fact_trade` count and the `EXPORTS_TO` count must still match, and §5's health
> check is written that way for exactly this reason.

Companion documents: [DATA_SOURCES.md](DATA_SOURCES.md) explains where the data
came from; this one is about looking at it once it is there.

---

## 1. Where the databases are

Two places, same schema.

| | Runs on | Postgres | Neo4j | Notes |
|---|---|---|---|---|
| **Local** | your laptop, `make up` | `localhost:5432` | `localhost:7687` | Your own sandbox. Break it freely |
| **Shared** | the database VM `8.231.116.24` | port 5432 | ports 7687 / 7474 | Real data. Everyone's demo runs off this |

The shared VM has three containers:

```
ceynex-postgres   healthy   5432
ceynex-neo4j      healthy   7474 (browser), 7687 (bolt)
ceynex-redis      healthy   6379
```

The backend API runs on a **different** VM (`8.231.64.229`, internal
`10.160.0.3`) and reaches the database VM over the private network at
`10.160.0.4`. That is why the API's health check can be green while your laptop
cannot connect to anything — they are different network paths.

### Getting to the shared databases from your laptop

Nothing is exposed to the internet on a useful port, so tunnel through SSH.
A tunnel just means "port 55432 on my laptop is really port 5432 on that VM":

```bash
ssh -f -N -L 55432:localhost:5432 database@8.231.116.24   # Postgres
ssh -f -N -L 57687:localhost:7687 database@8.231.116.24   # Neo4j
```

Local ports are deliberately `55432` and `57687` so they do not collide with
your own local stack on 5432/7687. You can have both running at once.

Close them when done — find the process and kill it by pid rather than with a
broad `pkill` pattern, which will happily match your own shell:

```bash
pgrep -af "ssh.*-L 55432"     # look first
kill <pid>                     # then kill that one
```

The passwords live in `~/db/.env` on the database VM. Read them into a variable
rather than typing them into a command that lands in your shell history:

```bash
export PGPASSWORD=$(ssh database@8.231.116.24 "grep '^POSTGRES_PASSWORD=' ~/db/.env | cut -d= -f2-")
psql -h localhost -p 55432 -U admin_ceynex -d ceynex
```

> The database VM's superuser is **`admin_ceynex`**, not `ceynex`. The local dev
> stack uses `ceynex`. Connecting with the wrong one gives you
> `password authentication failed`, which looks like a wrong password.

---

## 2. Three ways in

**psql — the command line.** Best for anything you want to copy into a document.

```bash
psql -h localhost -p 55432 -U admin_ceynex -d ceynex
```

Useful once you are in: `\dt` lists tables, `\d fact_trade` describes one,
`\x` toggles one-record-per-page (much easier to read for wide rows), `\q` quits.

**Adminer — a web page.** The local stack runs it at `http://localhost:8080`.
Point it at server `postgres`, user and password from your `.env`. Good for
clicking around, poor for anything repeatable.

**Neo4j Browser — for the graph.** `http://localhost:7474` locally. For the
shared VM you would need to tunnel 7474 as well. It draws the graph visually,
which is genuinely the fastest way to understand a relationship problem.

For the graph without a browser, run `cypher-shell` inside the container:

```bash
ssh database@8.231.116.24
NPW=$(grep '^NEO4J_PASSWORD=' ~/db/.env | cut -d= -f2-)
docker exec ceynex-neo4j cypher-shell -u neo4j -p "$NPW" 'MATCH (n) RETURN count(n);'
```

---

## 3. What is in there right now

### PostgreSQL — five tables

```sql
SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC;
```

```
 fact_trade  | 4625     the actual trade records
 dim_country |  249     country reference
 dim_hs      |   49     product code reference
 ingest_run  |    1     log of pipeline runs
 dq_flag     |    0     flagged data problems
```

`dq_flag` being empty is expected, not suspicious: it fills when two sources
disagree about the same figure, and there is currently only one source.

### Neo4j — five node types

```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC;
```

```
Country          201        EXPORTS_TO     4625
HSCode             9        COVERED_BY       13
TradeAgreement     6        CLASSIFIED_AS     5
Commodity          3
ApparelCategory    2
```

**`EXPORTS_TO` is 4,625 and `fact_trade` is 4,625.** That is not a coincidence —
each trade record becomes one relationship. It is also the single most useful
health check in this document (see §5).

Fewer `Country` nodes (201) than `dim_country` rows (249) is also expected: the
graph only holds countries Sri Lanka actually exported to.

---

## 4. Questions worth asking

### Is the data complete?

```sql
SELECT extract(year FROM period_start)::int AS year,
       count(*) AS rows, count(DISTINCT item) AS items,
       round(sum(export_value_usd)/1e9, 2) AS usd_bn
FROM fact_trade GROUP BY 1 ORDER BY 1;
```

```
 year | rows | items | usd_bn
 2015 |  494 |     5 |   6.01
 2016 |  503 |     5 |   6.02
 2017 |  505 |     5 |   6.47
 2019 |  535 |     5 |   6.70      <- note the jump
 2020 |  525 |     5 |   5.68
 2021 |  534 |     5 |   6.96
 2022 |  510 |     5 |   7.29
 2023 |  525 |     5 |   5.99
 2024 |  494 |     5 |   6.22
```

**This one query tells you three things.**

*2018 is missing.* Not a bug — Sri Lanka reported nothing to Comtrade that year.
Any growth rate or forecast crossing 2018 has a hole in it.

*The totals are believable.* Around USD 6 billion a year for these five products,
against Sri Lanka's total exports of roughly USD 12–13 billion. Apparel alone is
about USD 5 billion. If this column read 60 or 0.6, something is wrong by a
factor of a thousand — usually a units mistake.

*2020 dips.* That is COVID, visible in the data. Real-world events showing up
where you expect them is good evidence the pipeline is not mangling anything.

### Did the last pipeline run work?

```sql
SELECT * FROM ingest_run ORDER BY started_at DESC LIMIT 5;
```

```
 run_id       | 1
 source_id    | UN_COMTRADE
 started_at   | 2026-08-18 07:32:57+00
 finished_at  | 2026-08-18 07:32:57+00
 status       | success
 rows_written | 4625
 error        |
```

**A failed run still appears here.** The row is committed as soon as the run
starts, precisely so a crash does not roll away the evidence that it happened.
If `status` is `failed`, read `error`. If `finished_at` is empty, the run died
hard — check the container logs.

### Who buys what?

```sql
SELECT partner_iso3, round(sum(export_value_usd)/1e6, 1) AS usd_m
FROM fact_trade
WHERE sector = 'apparel' AND extract(year FROM period_start) = 2024
GROUP BY 1 ORDER BY 2 DESC LIMIT 5;
```

```
 USA | 1855.9
 GBR |  654.1
 ITA |  467.0
 DEU |  248.1
 NLD |  215.9
```

The United States taking about three times the United Kingdom, with Italy and
Germany behind — that is what Sri Lankan apparel exports actually look like. A
result that does not resemble this means the country codes are wrong (§5).

### What does the graph know about trade deals?

```cypher
MATCH (h:HSCode)-[:COVERED_BY]->(t:TradeAgreement)
WHERE h.code IN ['610910', '6109', '61']
RETURN t.name AS agreement, h.code AS matched_on;
```

```
 ISFTA    | 61
 UK DCTS  | 61
 GSP+     | 61
```

Coverage is recorded at chapter level (`61` = knitted clothing) and a question
about a specific product (`610910`, T-shirts) matches through it. That is why
the query asks for all three levels at once.

> **Every agreement is marked `unverified`.** Nobody has yet checked these
> against the official EU regulation. Under the team rule, no figure derived
> from them goes in a report until someone has.

---

## 5. Six checks that should always pass

Run these when something feels off. Each has a known-correct answer.

```sql
-- 1. No group rows. "World" (0) and "EU" (97) alongside individual
--    countries would double-count every total.  Expected: 0
SELECT count(*) FROM fact_trade WHERE partner_m49 IN (0, 97);

-- 2. Every partner resolved to a country code.  Expected: 0
--    Non-zero means partner_aliases.csv is missing an entry and
--    exports are being silently dropped.
SELECT count(*) FROM fact_trade
WHERE partner_m49 IS NOT NULL AND partner_iso3 IS NULL;

-- 3. The USA is present.  Expected: 45  (5 products x 9 years)
SELECT count(*) FROM fact_trade WHERE partner_iso3 = 'USA';

-- 4. No duplicate records.  Expected: 0
SELECT count(*) FROM (
  SELECT source_id, item, hs_code, reporter_iso3, partner_iso3,
         period_start, frequency
  FROM fact_trade
  GROUP BY 1,2,3,4,5,6,7 HAVING count(*) > 1
) d;

-- 5. Leading zeros intact.  Expected: 0902, 0906, 4001, 61, 62
--    Seeing "902" means the HS code was treated as a number somewhere.
SELECT DISTINCT hs_code FROM fact_trade ORDER BY 1;
```

```cypher
-- 6. Graph matches the database.  Expected: same number as fact_trade (4625)
MATCH ()-[e:EXPORTS_TO]->() RETURN count(e);
```

**Check 3 has history.** The USA arrives from Comtrade as country `842`, but the
international standard says `840`. Before that was handled, every USA row was
dropped as an unknown partner — **USD 6.66 billion**, the largest apparel market,
gone with a log line and no error. Checks 2 and 3 exist to catch that class of
failure returning.

**Check 6 is the best single indicator.** If Postgres and Neo4j disagree, the
graph is stale: someone ran `make ingest` without `make kg-load`. Agents read
the graph, so the API would answer confidently from old data.

Also worth knowing: no orphan countries.

```cypher
MATCH (c:Country) WHERE NOT ()-[:EXPORTS_TO]->(c) RETURN count(c);
-- Expected: 0. Countries are only created when a trade flow needs them.
```

---

## 6. When something is wrong

### Nothing connects

```bash
ssh database@8.231.116.24 "docker ps --format '{{.Names}}\t{{.Status}}'"
```

All three should say `healthy`. If a container is restarting:

```bash
ssh database@8.231.116.24 "docker logs ceynex-postgres --tail 50"
```

### The API says a database is down

```bash
ssh backend@8.231.64.229 "curl -s http://localhost:8000/health"
```

```json
{"status":"ok","neo4j":true,"postgres":true,"llm":false,
 "detail":{"fact_trade_rows":4625,"reasoning":"degraded: no API key, figures only"}}
```

Each dependency is reported separately on purpose. **`llm: false` is not a
fault** — no OpenAI key is configured, so the system runs its degraded path and
returns figures and evidence without generated prose. That is required
behaviour, not a broken service.

If `postgres` or `neo4j` is `false` while you can reach them from your laptop,
the problem is between the two VMs, not in the database. Check the firewall
rules rather than the database.

### Answers look stale or empty

In order:

1. `SELECT count(*) FROM fact_trade;` — is the data there at all?
2. `MATCH ()-[e:EXPORTS_TO]->() RETURN count(e);` — does the graph agree?
3. If they differ, run `make kg-load`.
4. `docker logs ceynex-api --tail 50` on the backend VM.

### A number looks wrong

The raw responses from the UN are kept under `data/raw/comtrade/<date>/`,
unedited. Go back to the original and check whether the wrong number came from
them or from us. That is the whole reason those files are kept.

---

## 7. Rules for touching the shared database

It is the only copy, and two teammates depend on it.

- **Read freely.** `SELECT` and `MATCH` cannot hurt anything.
- **Do not `DELETE`, `DROP` or `DETACH DELETE`.** Rebuilding from scratch is
  `make db-init ingest kg-load`, but everyone else is broken until it finishes.
- **Writing is done by the pipeline, not by hand.** `make ingest` is safe to
  re-run — the writer upserts, so twice gives the same row count as once, not
  double.
- **Try it locally first.** `make up` gives you an identical stack you can
  destroy.
- **Rotating a password?** Tell the team. It is in the database VM's `.env`, the
  backend VM's `.env`, and anyone's local tunnels.
