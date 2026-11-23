# FPL AI Manager

An ML-powered Fantasy Premier League assistant: ingests player/team/fixture
data, engineers leak-free time-series features, predicts each player's
points for upcoming gameweeks, and uses integer-programming optimisers to
recommend the best squad or transfers.

This is a **scaffold** — the architecture, data models, and tricky logic
(no-leakage rolling features, Elo ratings, ILP optimisers) are implemented;
the data-ingestion `TODO`s in `app/scheduler/jobs.py` and the DB-wiring in
a couple of routes are where you plug in real calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data collection                                             │
│  app/collectors/  fpl_client.py · understat_client.py         │
│                    player_matcher.py (reconciles the two)      │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Storage                                                      │
│  Postgres via SQLAlchemy (app/db/models.py)                   │
│  Redis for caching (app/cache/redis_cache.py)                 │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature engineering (Polars)                                 │
│  app/features/  rolling_features.py (shift(1) rolling windows) │
│                 elo_rating.py (xG-driven team attack/defence)  │
│                 feature_pipeline.py (joins it all together)    │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  ML (app/ml/)                                                 │
│  train.py — LightGBM, time-based validation split             │
│  predict.py — scores the upcoming gameweek                    │
│  model_registry.py — tracks trained model versions             │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Optimisation (app/optimizer/) — PuLP / MILP                  │
│  squad_optimizer.py — best 15 from scratch (wildcard)          │
│  transfer_optimizer.py — best transfers from an existing squad │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  API (FastAPI, app/main.py + app/api/routes/)                 │
└─────────────────────────────────────────────────────────────┘
```

Scheduled jobs (`app/scheduler/jobs.py`, run via APScheduler in the
`worker` container) keep data fresh and periodically retrain the model —
see that file for the cron schedule.

## Why these choices

- **Polars over pandas** for feature engineering: lazy evaluation + columnar
  storage make rolling-window operations over hundreds of thousands of
  player-gameweek rows meaningfully faster, and the `.shift(1)` pattern used
  throughout `rolling_features.py` is how leakage is prevented — every
  historical aggregate is computed from strictly-prior gameweeks only.
- **LightGBM over a full AutoML stack**: fast to train, easy to inspect and
  debug, and a reasonable starting point for ~500 players × ~200 features.
  `model_registry.py` keeps the training interface swappable — if you later
  want to try AutoGluon, XGBoost, or an ensemble, only `train.py`/`predict.py`
  need to change.
- **PuLP (MILP) for both optimisers**: squad/transfer selection has several
  simultaneous hard constraints (budget, exact position counts, max-3-per-club,
  starting-XI formation rules). A greedy heuristic can get stuck in an
  infeasible corner; an ILP solver explores the full feasible space and
  returns a provably optimal answer, and at this scale (~700 players, 15
  picks) it solves in well under a second with the bundled CBC solver.
- **Postgres + Redis + Docker Compose**: matches a "real" deployment shape —
  durable relational storage for historical stats, a cache layer so the API
  doesn't re-hit the FPL/Understat APIs on every request, and one command
  (`docker-compose up`) to bring the whole stack up locally.
- **Understat for xG**: it's free, but it's an unofficial scrape of embedded
  JSON (see the caveat in `understat_client.py`) — it can break if Understat
  changes their site, and you should keep request volume low and respect
  their terms of use. If this becomes a reliability problem, swapping in a
  paid provider (Sportmonks, Opta) only touches `app/collectors/` and the
  `PlayerMapping` table — nothing downstream needs to change.

## Getting started

```bash
cp .env.example .env
docker-compose up --build
```

This starts Postgres, Redis, the FastAPI app (port 8000), and the
scheduler worker. On first boot the API creates tables directly from the
models (`Base.metadata.create_all`) as a dev convenience — switch to Alembic
migrations (`alembic upgrade head`) once you're iterating on schema changes
with real data in place.

Then, one-time setup:

```bash
# populate teams & players from the official FPL API
docker-compose exec api python -m scripts.seed_initial_data

# once you have a few gameweeks of historical stats in the DB:
docker-compose exec api python -m scripts.train_model
```

Run tests:

```bash
pip install -r requirements.txt
pytest
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | DB/Redis/model status |
| GET | `/teams` | List Premier League teams |
| GET | `/players` | List players (filter by `position`, `team_id`) |
| GET | `/players/{id}` | Single player detail |
| GET | `/players/{id}/history` | Player's gameweek-by-gameweek stats |
| GET | `/fixtures?gameweek=` | Fixtures, optionally filtered |
| GET | `/predictions/gameweek/{gw}` | All players' predicted points for a gameweek |
| GET | `/predictions/player/{id}` | One player's predicted points across gameweeks |
| POST | `/optimizer/squad` | Build the best 15-player squad from scratch (wildcard) |
| POST | `/optimizer/transfers` | Best transfers for an existing squad (needs `manager_id`) |
| GET | `/manager/{manager_id}` | A manager's FPL entry info (pulled live from FPL API) |
| GET | `/manager/{manager_id}/picks/{gameweek}` | A manager's squad for a given gameweek |

Interactive docs at `http://localhost:8000/docs` once the API is running.

## What's stubbed vs. implemented

**Implemented, with real logic:**
- SQLAlchemy models for the full data model
- FPL API client (bootstrap-static, fixtures, player summaries, manager entry/picks)
- Understat scraper + the FPL↔Understat player matcher (team-constrained fuzzy matching)
- No-leakage rolling/composite feature engineering, with a unit test guarding the leakage property
- xG-driven team Elo rating system
- LightGBM training with time-based (not random) validation split, plus a file-based model registry
- Both ILP optimisers (squad-from-scratch and transfer recommendation), including the
  free-transfers-vs-hit trade-off

**Stubbed — wire these in next:**
- The four job bodies in `app/scheduler/jobs.py` (currently log-and-return; need to call the
  collectors above and upsert into the DB)
- `build_feature_matrix` is called in `scripts/train_model.py` with empty Understat/Elo/fixture
  frames — once ingestion jobs populate those tables, pass real data through
- `/optimizer/transfers` needs the manager's live squad + bank balance from
  `FPLClient.get_manager_picks()` wired into the route
- Alembic migrations: the env is configured, but no initial revision has been generated yet
  (`alembic revision --autogenerate -m "initial schema"`)

## Suggested build order

1. Seed teams/players (`scripts/seed_initial_data.py`) — gets `/teams` and `/players` working end to end.
2. Implement `ingest_latest_gameweek_results` — once a few gameweeks of real `PlayerGameweekStat`
   rows exist, `/players/{id}/history` comes alive and you can sanity-check the data.
3. Implement the Understat ingestion + player matching job, and an Elo-ratings update job that
   runs `TeamEloRatingSystem.update_after_match` after each finished fixture.
4. Run `scripts/train_model.py` once you have several gameweeks of data — check the MAE/R² it
   logs look sane (the reference project this was scaffolded against reports ~1.2 MAE / ~0.67 R²
   on total points, which is a reasonable target to benchmark against).
5. Wire `refresh_predictions` to call `ml.predict.predict_gameweek` and persist to the
   `predictions` table — this unblocks both `/predictions/*` and `/optimizer/*`.
6. Finish `/optimizer/transfers` by fetching the manager's real squad via `FPLClient`.

## Notes / caveats

- FPL's API is unauthenticated and undocumented but has been stable for years; it can still change
  without notice — the client has no retry/backoff logic yet, add some before relying on it for
  anything time-critical (e.g. a pre-deadline refresh job).
- The transfer optimiser uses each owned player's `now_cost` as a stand-in for sell price. FPL's
  real sell-price rule (you only keep half of any price rise since you bought a player) needs the
  manager's actual purchase prices from their entry data for full accuracy.
- Elo ratings, rolling windows, and the LightGBM config here are reasonable starting points, not
  tuned — expect to iterate on window sizes, feature set, and model hyperparameters against your
  own validation results.
