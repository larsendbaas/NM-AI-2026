# Astar Island Solver

This workspace now contains a complete Python CLI for the Astar Island challenge. It is designed for the live API, uses only the Python standard library, and persists round state locally so you can fetch, collect, predict, and submit without losing progress.

## What It Does

- fetches the active round and saves the full round detail
- imports any existing `docs/sim_seed*.json` samples as bootstrap observations
- spends the live query budget using a deterministic plan
- learns a smoothed probabilistic transition model from queried viewports
- writes one `40 x 40 x 6` prediction tensor per seed
- optionally submits all predictions to the API

## Solver Strategy

The implementation assumes the hidden parameters are shared across all seeds within a round and uses that aggressively:

1. `coverage` queries tile each 40x40 seed with 9 overlapping 15x15 windows.
2. The remaining 5 queries are used as `hotspot` samples on the most dynamic windows, biased toward settlement-heavy regions.
3. Every observed cell updates pooled transition statistics keyed by initial local context:
   - raw terrain
   - coastal vs inland
   - nearby settlements, ports, ruins, and forests
   - nearest ocean / settlement / ruin distance bins
4. Unobserved cells use a heuristic prior based on the mechanics docs.
5. Observed cells, specific context buckets, and broad terrain buckets are blended into one probability distribution with a 0.01 floor for KL safety.

This is not a trivial baseline. It is meant to work with the real round structure you already fetched into `docs/`.

## Files

- `astar_island/api.py`: API client
- `astar_island/features.py`: feature extraction from initial states
- `astar_island/planner.py`: 50-query plan
- `astar_island/model.py`: probabilistic predictor
- `astar_island/solver.py`: orchestration
- `astar_island/cli.py`: command-line interface

## Usage

Set your token locally:

```powershell
$env:AINM_TOKEN="YOUR_REAL_TOKEN"
```

Fetch and initialize the active round:

```powershell
py -m astar_island fetch
```

Collect live simulations with the remaining budget:

```powershell
py -m astar_island collect
```

Build prediction tensors from the saved observations:

```powershell
py -m astar_island predict
```

Submit all 5 seeds:

```powershell
py -m astar_island submit
```

Or do the full run:

```powershell
py -m astar_island solve --submit
```

Useful safe variants:

```powershell
py -m astar_island collect --max-queries 5
py -m astar_island solve --dry-run
```

## Where Output Goes

Each round gets its own directory:

```text
runs/<round_id>/
```

That directory stores:

- `round.json`
- `round_detail.json`
- `query_plan.json`
- `observations.jsonl`
- `prediction_seed_0.json` ... `prediction_seed_4.json`
- `prediction_report.json`
- `submit_response.json`

## Important Notes

- The solver imports your existing `docs/sim_seed*.json` files automatically.
- `collect` will not deliberately exceed the 50-query plan.
- Predictions always keep a minimum class floor of `0.01`.
- If you already spent queries manually outside the solver, the API budget is still authoritative.

## Suggested Live Flow

1. Rotate your token, because it was pasted into chat.
2. Set `AINM_TOKEN` locally.
3. Run `py -m astar_island fetch`.
4. Run `py -m astar_island collect`.
5. Run `py -m astar_island predict`.
6. Inspect the files in `runs/<round_id>/`.
7. Run `py -m astar_island submit` when you are satisfied.
