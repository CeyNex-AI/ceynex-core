# Deferred

Work that is consciously **not** in this repo, with what depends on it. The
point of writing this down is that a marker or a teammate can tell the
difference between a gap someone decided on and a gap nobody noticed.

Deviations from the SAD that *were* built live in
[ARCHITECTURE_DELTA.md](ARCHITECTURE_DELTA.md). This file is the other half:
things not built at all.

---

## Owned by M3, deliberately not pre-empted

`ceynex/api/` was seeded by M2 so the orchestrator is reachable for the
mid-evaluation demo, and is M3's from then on. Routes now exist for `GET
/health`, `POST /api/query`, auth (`ceynex/api/routes/auth.py`), history
(`ceynex/api/routes/history.py`), and admin
(`ceynex/api/routes/admin.py`) — and nothing else beyond that. Specifically
**not** built, because building them would mean guessing at M3's design and
then arguing about it:

| Deferred | Spec | Consequence today |
|---|---|---|
| Help and guidance content | SRS 3.5.5 | — |
| `web/` frontend | SAD §6 | The frontend VM is intentionally empty |

**Saved / bookmarked queries (SRS 3.5.2, second half) are built**: `saved` is
a column on `query_history` (`ceynex/api/history.py`), not a second table — a
saved query is a history entry, just flagged, and every query that could ever
be saved already has a row there from the moment it was asked. `POST
/api/history/{id}/save` and `.../unsave` toggle it, scoped to the caller's own
`user_email` in the same `UPDATE` (never a separate ownership check, same
reasoning as `auth.authenticate` never distinguishing "no such user" from
"wrong password") — a 404 covers both "no such entry" and "not yours", not
just the first. `GET /api/history?saved=true` filters the existing list route
rather than adding a second one.

**Admin routes (SRS 3.5.4) are built**: `GET /api/admin/models`, `POST
/api/admin/retrain`, `POST /api/admin/pipeline/ingest`, `GET
/api/admin/pipeline/status`, `GET /api/admin/dq-flags`, `POST
/api/admin/dq-flags/{id}/resolve` (`ceynex/api/routes/admin.py`,
`ceynex/api/admin.py`). All six require the `admin` role via `require_admin` —
403 for any other signed-in role, 401 for none. Retrain and ingest wrap the
real hooks (`ceynex.models.registry.retrain`,
`ceynex.data.pipeline.run_source`) rather than reimplementing them, and both
run via `asyncio.to_thread` with the HTTP response waiting on the real work —
a deliberate simplification at this data scale, not a queue; see
`ceynex/api/routes/admin.py`'s module docstring for when that stops being
true. Retrain only works on an already-registered `sector/item/target` (404
otherwise) — it refits the same model class on fresh data, it does not train
something new from a bare request.

**Real user accounts and RBAC are built; the four fixed demo accounts are
gone.** `ceynex/api/users.py` is a real `users` table (bcrypt password hashes,
a `role` column over the same four roles, a `disabled_at` flag), additive to
the frozen contracts schema the same way `query_history` is. `POST
/api/auth/signup` self-registers an account at the default role
(`researcher`) and logs it straight in; `POST /api/auth/login` checks the
stored hash. A signed-in user changes their own password at `POST
/api/account/password` (current password required). Admin-only routes on the
admin router provision an account at any role (`POST /api/admin/users`), list
every account (`GET /api/admin/users`), move one between roles (`POST
/api/admin/users/{id}/role`), disable or re-enable one (`POST
/api/admin/users/{id}/{disable,enable}`), and reset a locked-out user's
password (`POST /api/admin/users/{id}/password`, no current-password check) —
each audited first, with a last-enabled-admin guard so a deployment can't lock
itself out.
A fresh deployment starts with zero users: seed the first admin with
`CEYNEX_BOOTSTRAP_ADMIN=email:password` (read once by `ensure_table()` at
startup) or `python -m ceynex.api.users create-admin <email> <password>`. The
login JWT still carries the role as a claim (8 h TTL), so a role change or
disable catches an already-signed-in session only within that window;
immediate for new logins and for API keys (`auth.role_for_email` re-derives
live).

**`POST /api/query` itself still does not require a token.** `require_user`
(`ceynex/api/routes/auth.py`) gates every route that needs a signed-in user,
but wiring it onto `/api/query` was a deliberate choice left for later: the
four roles gate which UI pages render and the admin routes, not *what* a query
can see, so requiring a token there today would add a login wall without
changing any behaviour behind it. `POST /api/query` instead takes an *optional*
token (`get_optional_user`) — signed in or not, a query still answers; being
signed in only additionally attributes it to that user for history. Revisit
the hard requirement once a route needs to tell users apart to change *what*
it returns.

**Query history (SRS 3.5.2, first half) is built**: `ceynex/api/history.py`
records every authenticated query into a `query_history` table (additive to
the frozen contracts schema, not part of it — see that module's docstring for
why), and `GET /api/history` lists a signed-in user's own past queries. An
anonymous query, or one with an invalid/expired token, still answers
normally — it simply is not recorded, silently, by design.

## WITS tariff ingestion — cut

**Decided 2026-08-19. Spec touched: SRS 3.1.7, 3.1.8.**

The M2 plan lists WITS as the first thing to cut under schedule pressure, with a
static GSP+ table as the documented fallback. It is now cut outright: no
`ceynex/data/connectors/wits.py`, and no committed tariff table either.

**Why.** The schedule is roughly two weeks behind the written plan, and the
three things the plan says may never be cut — the orchestrator, the
single-graph constraint, the 30-question evaluation — are worth more marks than
a second connector. The orchestrator and the graph constraint are done; the
evaluation is not, and it is the headline deliverable. WITS is what pays for it.

**What actually degrades.** Less than it sounds, and precisely one thing:

- **FX shocks** are unaffected — no tariff rate is involved.
- **Explicit tariff scenarios** ("if the EU raises tariffs on tea by 10%") are
  unaffected: the rate comes from the question, not from WITS.
- **Preference-loss scenarios** ("if Sri Lanka loses GSP+") need an MFN rate to
  re-impose, and that is the number WITS would have supplied. It is currently a
  documented constant in `config/elasticities.yaml`
  (`agreement_loss_mfn_tariff`, default 9.5%), surfaced in the agent's
  `assumptions` list on every run rather than hidden as a literal.

So the honest description is: **preference-loss magnitudes rest on a literature
constant rather than a queried tariff schedule, and say so.** Whether GSP+
*covers* a given HS code is still resolved from the knowledge graph and is not
affected by this cut.

**What does not happen.** The agent does not silently invent coverage. With no
`TradeAgreement` coverage in the graph, `_simulate_agreement_loss` returns no
outcome and the agent reports that the simulation cannot be completed, citing
the Cypher that found nothing — SAD §4.1. Four tests in
`tests/agents/test_trade_economics.py` hold that line, including one asserting
the refusal cites the coverage query rather than some other query that happened
to be in scope.

**To undo the cut,** add a `WITSConnector(DataSourceConnector)` and register it
in `CONNECTORS` in `ceynex/data/pipeline.py`; nothing else changes shape.

### Amended 2026-08-28: a second source, not a replacement (D10)

Policy-document retrieval now supplies an MFN rate **where a retrieved document
states one unambiguously**, and `trade_economics` cites the sentence it came
from. The cut still stands, and the difference matters:

- **WITS would have been a tariff schedule** — every rate, every line, queryable.
- **This is one rate, from prose, when the prose happens to say so.**
  `ceynex/retrieval/rates.py` requires the percentage, a tariff word, and the
  goods to appear in the *same sentence*, and refuses outright when two
  different rates qualify. It is built to return None more often than not,
  because a plausible wrong rate carrying a real citation is worse than the
  documented constant — `orchestrator/grounding.py` cannot tell the two apart.

So the honest description is now: **preference-loss magnitudes rest on a
literature constant unless a policy document states a rate for those goods, and
the answer says which of the two it used, on every run.** Neither is a queried
tariff schedule, and neither is verified — the rate ships `unverified` like every
other figure derived from the reference CSVs.

`agreement_loss_mfn_tariff` stays in `config/elasticities.yaml` and is still the
fallback. Deleting it once retrieval existed would have made the system's answer
depend on whether a search happened to hit.

## Data owned by teammates

`fact_trade` and the sector nodes are populated by M1 (agriculture) and M3
(apparel) through the same `UnifiedDatasetWriter` and `MERGE`-only loaders, so
their loads add to mine rather than colliding with them. M2 seeds only
`dim_country`, `dim_hs`, the `TradeAgreement` nodes, and the Comtrade extract.

## Measured but not measured under load

**SRS 3.4.2 wants 50 concurrent users. Nothing has tested that.** The latency
figures in [EVALUATION.md](EVALUATION.md) are single-user, sequential by design
so the numbers mean something per query. They pass their budgets with 5–15x
headroom, and that headroom is partly because no LLM key was configured when
they were taken, so the system never paid for a model call.

**Retaken on 2026-08-28 with a key configured, and one budget now fails.**
Single-sector p95 came out at 14.6 s against a 10 s budget (SRS 3.4.1), where
the keyless run had posted 2.7 s. That headroom was never real — it was the
degraded path being reported as the system. See EVALUATION.md §1.

**Throughput at 50 concurrent users (SRS 3.4.2) is still untested**, and is now
the more important of the two: the single-user numbers no longer have headroom
to spare.

One thing that *does* now exist between a load test and a real outage: the SRS
3.4.6 rate limiter caps any single caller at 30 queries/minute, so the
50-concurrent-user figure is about 50 distinct users, not one script.

## Merge coherence not yet rated

`eval/coherence.py` builds the blind rating sheets and scores them. It needs
three human raters and the session has not happened, so SRS 3.1.2's "one
coherent answer, not a list of per-agent responses" is currently supported by
the merger's design and its unit tests, not by a measurement.

**Now unblocked and now the longest-lead item.** It was waiting on the two
sector agents (landed 26 Aug) and on prose generation being on (it is). Rating
prose the LLM never wrote would have measured the deterministic composer, which
is not what SRS 3.1.2 is about. Everything else outstanding is a command; this
one needs three people's calendars, so book it before writing anything else.

## Audit logging (SRS 3.4.7) — administrative half now built

**Found 2026-08-28 while implementing the rate limiter, not previously
recorded anywhere. The administrative half was built 2026-09-04.**

SRS 3.4.7 requires "an audit log of all user queries and all administrative
actions, such as changes to user accounts or manual interventions in the data
pipeline". Two different mechanisms cover the two halves:

- **User queries** are recorded, for signed-in callers only, by
  `ceynex/api/history.py`. That table was built for SRS 3.5.2 (the user's own
  history), so it is scoped to the caller and has no retention or tamper
  story. It is a feature that happens to leave a trail, not an audit log.
- **Administrative actions** are now recorded by `ceynex/api/audit.py`'s
  `audit_log` table. `POST /api/admin/retrain`, `POST /api/admin/pipeline/ingest`
  and `POST /api/admin/dq-flags/{id}/resolve` each write an `(actor_email,
  action, target, logged_at)` row via `routes/admin.py`'s `_audit()` helper
  **before** performing the mutation, and `GET /api/admin/audit-log` (also
  behind `require_admin`) lists them back, newest first.

  This is deliberately not opportunistic the way `history.record()` is: a
  lost history row costs nothing, but an admin mutation with no audit row is
  exactly the failure this section used to warn about — "an audit log that
  misses some actions is worse than none, it invites the reader to trust a
  record that is not complete." So `audit.record()` lets `psycopg.Error`
  propagate, and `_audit()` turns that into a 503 *before* the mutation runs —
  a Postgres outage blocks the admin action rather than letting it through
  unlogged. `tests/api/test_admin.py`'s
  `test_an_unwritable_audit_log_blocks_retrain_rather_than_running_it_unlogged`
  holds that line directly, spying on `_do_retrain` to prove it is never
  called when the audit write fails.

  Not covered by this table, deliberately out of scope for SRS 3.4.7's
  "administrative actions" wording: login/logout, and the read-only admin
  routes (`GET /models`, `/pipeline/status`, `/dq-flags`, `/llm/status`) —
  none of them mutate state. No retention policy or export tooling exists yet
  either; the table is append-only Postgres, nothing more.

## Rate limiting (SRS 3.4.6) — built, with one stated exposure

`ceynex/api/rate_limit.py`, wired onto `POST /api/query` only. Redis-backed
when `REDIS_URL` is set (the deployed image runs two uvicorn workers, so a
per-process counter would permit double the configured limit), per-process
otherwise.

**It fails open.** If Redis is unreachable the request is allowed and a warning
is logged, so a Redis outage means abuse is unthrottled until it is restored.
That is the deliberate direction — the alternative is a rate-limit store outage
taking down query submission entirely, which causes the unavailability the
limiter exists to prevent — but it is an exposure and is recorded here rather
than left to be discovered.

Also unlimited by design, each needing its own justification before being
throttled: login, a user reading their own history, and the admin routes.

## Operational

- **`OPENAI_API_KEY` is now set** (and an OpenRouter free-tier failsafe sits
  behind it), so the system no longer runs the SRS 3.4.3 degraded path by
  default on a machine that has the `.env`. Degraded mode is still reachable
  deliberately — `--no-llm`, or `make eval-degraded` — and still tested. Both
  paths are now measured side by side in EVALUATION.md §1. Whether the deployed
  VM's own `.env` carries the key is not visible from a checkout and should be
  confirmed on the box, not assumed from this file.
- **`COMTRADE_API_KEY` is unset.** Comtrade uses the keyless public preview
  endpoint, which returns real data at a lower rate limit.
- **SSH (22) and RDP (3389) are open to `0.0.0.0/0`** on the `ceynex-dev` VPC.
  Not closed unilaterally because restricting SSH to a single address could lock
  out two teammates. RDP serves no purpose on these Linux VMs and can be deleted
  safely.
- **Data-tier credentials have not been rotated** since being committed as
  `.env.example` values in `DevOps/`.

## News sidecar (D11)

- **The news endpoints are unauthenticated.** `GET /api/news/search` and
  `/api/news/trending` use `get_optional_user` and gate nothing, matching
  `POST /api/query`, which deliberately answers anonymous callers. Gating the
  sidecar while the main event stays open would be incoherent; both should be
  closed together or not at all. They have their own rate-limit allowance under
  a `news:` identity prefix, so abuse of one cannot exhaust the other.
- **The relevance floor is measured but not a clean boundary.**
  `relevance.min_score = -8.0` comes from scoring 150 real indexed headlines
  against 11 real questions on the deployed box (the table is in
  `news/relevance.py`). It is the highest cut that rejects every out-of-scope
  question tried. It is *not* a separator: "Will it rain in Colombo tomorrow?"
  reached -8.14 on the word Colombo alone — above three genuine questions and
  0.14 from the cut. With a corpus of 150 mostly-unrelated headlines nothing
  scores well, so this should be re-measured once the collection has run for a
  few days. The UI shows three coarse buckets rather than a number precisely
  because the number does not support finer claims.
- **Cross-outlet story deduplication is not attempted.** One wire story running
  on forty sites is forty points in `ceynex_news`. Near-identical titles are
  collapsed in a *response* only; entity resolution across outlets is not
  solvable from a headline at acceptable cost.
- **`store.blocked_domains` is empty.** GDELT indexes everything, and
  `sourcelang:english` is not a quality filter. The mechanism exists because it
  is impossible to retrofit under demo pressure, not because anything is
  currently excluded.
- **The watchlist queries have not been validated against live GDELT.** They are
  syntactically checked by `tests/news/test_config.py` and printable with
  `python -m ceynex.news.refresh --dry-run`, but a query GDELT silently rejects
  returns a plain-text 200 that looks identical to "no coverage". Run
  `make news-refresh` once and read the report before trusting the panel.
- **GDELT's rate limit is undocumented and real.** It returns HTTP 429 under
  concurrent requests. `news/throttle.py` gates outbound calls to one every five
  seconds by convention, not by measurement — the actual ceiling is unknown.
- **The deployed backend calls GDELT over plain HTTP.** `api.gdeltproject.org`
  resets every connection to :443 from that VM while answering on :80 normally;
  github.com and api.openai.com are reachable over TLS from the same host, so
  the fault is that endpoint's rather than the network's.
  `CEYNEX_GDELT_BASE_URL` in `ceynex-infra/backend/docker-compose.yml` carries
  the override and `config/news.yaml` still defaults to https. What crosses in
  clear text is the user's question and a list of public headlines — this API
  has no key, no token and no account. The integrity risk is bounded by news
  never being evidence (D11). Remove the override once :443 answers.
