"""Command-line interface for the Astar Island solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import AstarIslandClient
from .solver import AstarIslandSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace root containing docs/ and where run state should be stored.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch", help="Fetch and save the active round detail.")

    collect = subparsers.add_parser("collect", help="Use the remaining live budget to collect simulations.")
    collect.add_argument("--round-id", help="Round id. Defaults to the active round.")
    collect.add_argument("--max-queries", type=int, help="Limit how many live queries to execute.")
    collect.add_argument("--dry-run", action="store_true", help="Plan queries without calling the API.")

    predict = subparsers.add_parser("predict", help="Build predictions from saved observations.")
    predict.add_argument("--round-id", help="Round id. Defaults to the active round.")

    submit = subparsers.add_parser("submit", help="Submit saved predictions for all seeds.")
    submit.add_argument("--round-id", help="Round id. Defaults to the active round.")

    solve = subparsers.add_parser("solve", help="Fetch, collect, predict, and optionally submit.")
    solve.add_argument("--max-queries", type=int, help="Limit live collection.")
    solve.add_argument("--dry-run", action="store_true", help="Do not call /simulate.")
    solve.add_argument("--submit", action="store_true", help="Submit after building predictions.")

    return parser


def resolve_round(solver: AstarIslandSolver, round_id: str | None) -> tuple[dict, object]:
    if round_id:
        return solver.load_round(round_id)
    return solver.fetch_active_round()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    solver = AstarIslandSolver(workspace_root=workspace, client=AstarIslandClient())

    if args.command == "fetch":
        detail, store = solver.fetch_active_round()
        imported = solver.import_bootstrap_samples(detail["id"], store)
        print(f"Fetched round {detail['id']} into {store.root}")
        if imported:
            print(f"Imported {imported} bootstrap simulation sample(s).")
        return

    if args.command == "collect":
        detail, store = resolve_round(solver, args.round_id)
        imported = solver.import_bootstrap_samples(detail["id"], store)
        result = solver.collect(detail, store, max_queries=args.max_queries, dry_run=args.dry_run)
        print(f"Collected {result.executed_queries} query(s). Remaining budget: {result.remaining_budget}.")
        if imported:
            print(f"Imported {imported} bootstrap sample(s) first.")
        return

    if args.command == "predict":
        detail, store = resolve_round(solver, args.round_id)
        imported = solver.import_bootstrap_samples(detail["id"], store)
        report = solver.predict(detail, store)
        print(
            f"Predictions saved for {report['seeds']} seeds "
            f"using {report['queries_loaded']} query samples and {report['observed_cells']} observed cells."
        )
        if imported:
            print(f"Imported {imported} bootstrap sample(s) first.")
        return

    if args.command == "submit":
        detail, store = resolve_round(solver, args.round_id)
        responses = solver.submit(detail, store)
        print(f"Submitted {len(responses)} predictions for round {detail['id']}.")
        return

    if args.command == "solve":
        detail, store = solver.fetch_active_round()
        imported = solver.import_bootstrap_samples(detail["id"], store)
        result = solver.collect(detail, store, max_queries=args.max_queries, dry_run=args.dry_run)
        report = solver.predict(detail, store)
        print(
            f"Round {detail['id']}: collected {result.executed_queries} query(s), "
            f"remaining budget {result.remaining_budget}, predictions written for {report['seeds']} seeds."
        )
        if imported:
            print(f"Imported {imported} bootstrap sample(s).")
        if args.submit and not args.dry_run:
            responses = solver.submit(detail, store)
            print(f"Submitted {len(responses)} predictions.")
        return

    parser.error(f"Unknown command {args.command!r}")
