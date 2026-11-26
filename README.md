# FPL AI Manager

An ML-powered Fantasy Premier League assistant: ingests player/team/fixture
data, engineers leak-free time-series features, predicts each player's
points for upcoming gameweeks, and uses integer-programming optimisers to
recommend the best squad or transfers.

**Status: working end-to-end.** The full pipeline runs on FPL data alone —
scheduled ingestion, LightGBM predictions, and both optimisers are live.
xG features (Understat) and Elo ratings are scaffolded but dormant while
understat.com is offline (see "Data sources" below).

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
`worker` container) keep data fresh and periodically retrain the model.
Current schedule:

| Job | Frequency | What it does |
|---|---|---|
| `ingest_bootstrap_static` | every 4h | Sync teams/players, refresh prices + injury status |
| `ingest_latest_gameweek_results` | daily 03:00 | Sync fixtures + per-gameweek player stats |
| `refresh_predictions` | 06:00 & 18:00 | Predict points for the next unplayed gameweek |
| `retrain_model` | Mondays 04:00 | Full retrain as more gameweeks accumulate |

## How the ML works

**Features** (`app/features/`): for each player-gameweek we compute
rolling means over 3/5/10-gameweek windows for points, minutes, ICT index,
bonus, bps, etc., plus short-vs-long trend terms. Every rolling aggregate
uses `.shift(1)` so gameweek N's features only see gameweeks < N — this is
the anti-leakage rule, enforced by a unit test.

**Model** (`app/ml/train.py`): LightGBM regressor predicting
`total_points` directly. Validation uses a **time-based split** (last 4
gameweeks held out) rather than random k-fold, because random splits leak
future information. Trained models are versioned in `models/` with a
`registry.json`.

**Prediction** (`app/ml/predict.py` + `refresh_predictions` job): appends
one synthetic row per player for the upcoming gameweek to the historical
stats frame, runs the *same* `build_feature_matrix`, and predicts only the
future rows. Training and inference share one code path, so features can't
drift apart.

**Optimisers** (`app/optimizer/`): PuLP/CBC mixed-integer programs that
maximise starting-XI predicted points (captain counts double) subject to
FPL's real rules — 15 players, position counts, budget, max-3-per-club,
valid formations. The transfer optimiser additionally weighs the −4pt cost
of hits beyond free transfers, so "take a hit or not" is an optimisation
decision.

## Why these choices

- **Polars over pandas** for feature engineering: lazy evaluation + columnar
  storage make rolling-window operations over hundreds of thousands of
  player-gameweek rows meaningfully faster, and the `.shift(1)` pattern used
  throughout `rolling_features.py` is how leakage is prevented.
- **LightGBM over a full AutoML stack**: fast to train, easy to inspect and
  debug, and a reasonable starting point for ~700 players × ~20 features.
  `model_registry.py` keeps the training interface swappable.
- **PuLP (MILP) for both optimisers**: squad/transfer selection has several
  simultaneous hard constraints. A greedy heuristic can get stuck in an
  infeasible corner; an ILP solver returns a provably optimal answer in
  well under a second at this scale.
- **Postgres + Redis + Docker Compose**: durable relational storage,
  a cache layer, and one command to bring the stack up.

## Getting started

```bash
cp .env.example .env        # set DEFAULT_MANAGER_ID to your FPL team id
docker-compose up --build -d
```

This starts Postgres, Redis, the FastAPI app (port 8000), and the
scheduler worker. Tables are created from the models on first boot
(`Base.metadata.create_all`) — switch to Alembic migrations
(`alembic revision --autogenerate -m "..."` then `alembic upgrade head`)
once the schema stabilises.

One-time seed (only needed on a fresh DB):

```bash
docker-compose exec api python -m scripts.seed_initial_data
```

Run tests:

```bash
pip install -r requirements.txt
pytest
```

## Essential commands

Everything below assumes `cd` into the project root. **After editing code,
you must rebuild** — the image bakes in source (`docker-compose up --build -d`).

```bash
# Stack control
docker-compose up --build -d      # build + start all services
docker-compose ps                 # check health
docker-compose logs -f worker     # watch scheduled jobs live
docker-compose logs api --tail 50 # API logs / tracebacks

# Manual job runs (same functions the cron calls)
docker-compose exec worker python -c "
import asyncio, logging
logging.basicConfig(level='INFO')
from app.scheduler.jobs import ingest_latest_gameweek_results
asyncio.run(ingest_latest_gameweek_results())
"
# same pattern for: ingest_bootstrap_static, refresh_predictions, retrain_model

# Training
docker-compose exec api python -m scripts.train_model

# Database inspection
docker-compose exec postgres psql -U fpl -d fpl -c "SELECT count(*) FROM player_gameweek_stats;"

# API smoke tests
curl -s localhost:8000/health
curl -s "localhost:8000/players?position=MID" | head -c 500
curl -s "localhost:8000/predictions/gameweek/12" | head -c 500
curl -s -X POST localhost:8000/optimizer/squad \
  -H "Content-Type: application/json" -d '{"gameweek": 12}'
curl -s -X POST localhost:8000/optimizer/transfers \
  -H "Content-Type: application/json" -d '{"gameweek": 12}'
```

Interactive API docs at `http://localhost:8000/docs`.

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
| POST | `/optimizer/squad` | Best 15-player squad from scratch (wildcard) |
| POST | `/optimizer/transfers` | Best transfers for your squad (uses `DEFAULT_MANAGER_ID`) |
| GET | `/manager/{manager_id}` | A manager's FPL entry info (live from FPL API) |
| GET | `/manager/{manager_id}/picks/{gameweek}` | A manager's squad for a gameweek |

## Current accuracy (as of GW11, Nov 2026)

First full-season validation: **real-world MAE 1.455** vs validation MAE
1.39 — the model generalises as expected (no leakage). Training metrics on
the latest holdout: MAE ~1.36, R² ~0.25. These are FPL-stats-only numbers;
xG features should improve them once a data source is available.

## What's implemented vs. dormant

**Live:**
- All four scheduler jobs (bootstrap sync, fixture+stat ingestion,
  prediction refresh, weekly retrain)
- No-leakage rolling/composite features + unit test
- LightGBM training with time-based split + versioned model registry
- Both ILP optimisers, including free-transfers-vs-hit trade-off
- `/optimizer/transfers` pulls your real squad + bank from the FPL API
  (`DEFAULT_MANAGER_ID` in `.env`)

**Dormant (scaffolded, awaiting data source):**
- Understat xG ingestion + `PlayerMatcher` (E3a–c in `jobs.py`)
- xG-driven Elo ratings (E4 in `jobs.py`) — needs xG input
- `fixture_difficulty` column exists but is NULL — FPL doesn't expose FDR;
  Elo opponent ratings are the intended replacement

## Data sources

- **FPL API** (`fantasy.premierleague.com/api`) — unauthenticated,
  undocumented but stable for years. No retry/backoff yet; add some before
  relying on it for deadline-critical jobs.
- **Understat** — currently **offline since 2 Nov 2026** (site-wide outage,
  not a code issue). The scraper + matcher are written and ready; when the
  site returns, uncomment the E3/E4 calls in
  `ingest_latest_gameweek_results`. Alternatives (FotMob unofficial API,
  Sportmonks paid) would only touch `app/collectors/`.

## Notes / caveats

- The transfer optimiser uses `now_cost` as a sell-price proxy. FPL's real
  sell-price rule (half of price rises since purchase) isn't exposed by the
  current API responses — would need purchase-price tracking over time.
- `refresh_predictions` predicts every rostered player including injured
  ones (`status != 'a'`); the optimiser/user decides whether to pick them.
- Rolling windows (3/5/10), LightGBM hyperparameters, and feature set are
  reasonable starting points, not tuned — iterate against validation MAE.
- FPL returns some numeric fields as strings — ingestion casts
  `int()`/`float()` explicitly; keep that habit for new fields.
