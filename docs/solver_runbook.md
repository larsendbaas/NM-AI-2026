# Astar Island Solver Runbook

## Before Running

1. Log out and back in to `app.ainm.no`.
2. Copy a fresh `access_token`.
3. Set it in PowerShell:

```powershell
$env:AINM_TOKEN="YOUR_FRESH_TOKEN"
```

## Commands

Fetch the active round into `runs/<round_id>/`:

```powershell
py -m astar_island fetch
```

Use the remaining live query budget:

```powershell
py -m astar_island collect
```

Generate predictions from the saved observations:

```powershell
py -m astar_island predict
```

Submit all 5 seeds:

```powershell
py -m astar_island submit
```

## Safe Checks

Preview the collection plan without spending more queries:

```powershell
py -m astar_island collect --dry-run --max-queries 5
```

Generate predictions without touching the API:

```powershell
py -m astar_island predict --round-id <round_id>
```

## Output

The solver stores everything in:

```text
runs/<round_id>/
```

Important files:

- `observations.jsonl`: collected simulation samples
- `query_plan.json`: the 50-query plan
- `prediction_seed_*.json`: submission tensors
- `prediction_report.json`: summary of what was used to build predictions

## Model Summary

- 45 coverage queries: 9 windows per seed
- 5 hotspot queries: one extra high-value window per seed
- pooled transition statistics across all seeds
- heuristic priors for unobserved cells
- 0.01 probability floor on every class
