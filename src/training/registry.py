"""Manage model versions in the MLflow Model Registry with aliases.

Aliases (MLflow's replacement for the deprecated Staging/Production stages):
    @champion    the version serving production traffic
    @challenger  a candidate waiting to be compared against the champion
    @previous    the last champion, kept for one-step rollback

Usage:
    python -m src.training.registry register --run-name lgbm-poisson-31leaves
    python -m src.training.registry register --best
    python -m src.training.registry promote
    python -m src.training.registry rollback
    python -m src.training.registry archive --version 3
    python -m src.training.registry list
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.features.training_data import FEATURE_SERVICE
from src.training.train import EXPERIMENT

MODEL_NAME = "bike-demand-model"
SELECTION_METRIC = "valid_mae"  # lower is better; the test week stays for final reporting
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def alias_version(c: MlflowClient, alias: str):
    try:
        return c.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def version_metric(c: MlflowClient, version) -> float:
    return c.get_run(version.run_id).data.metrics[SELECTION_METRIC]


def find_run(c: MlflowClient, run_name: str | None):
    exp = c.get_experiment_by_name(EXPERIMENT)
    # Only runs that logged a model can be registered (the baseline has none).
    query = "tags.mlflow.runName = '{}'".format(run_name) if run_name else "params.model_type = 'lightgbm'"
    runs = c.search_runs(
        [exp.experiment_id], filter_string=query,
        order_by=[f"metrics.{SELECTION_METRIC} ASC"], max_results=1,
    )
    if not runs:
        raise SystemExit(f"No run found for {run_name or 'best lightgbm'}")
    return runs[0]


def register(c: MlflowClient, run_name: str | None) -> None:
    run = find_run(c, run_name)
    # MLflow 3 stores models as LoggedModels linked to the run, addressed as models:/<model_id>.
    model_id = run.outputs.model_outputs[0].model_id
    mv = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)

    c.update_registered_model(
        MODEL_NAME, description="Hourly rentals per Citi Bike station (next hour)."
    )
    c.update_model_version(
        MODEL_NAME, mv.version,
        description=f"From run '{run.info.run_name}' ({SELECTION_METRIC}={run.data.metrics[SELECTION_METRIC]:.3f})",
    )
    # Version-level tags carry lineage into the registry, so serving knows which features to fetch.
    for key, value in {
        "feature_service": run.data.tags.get("feature_service", FEATURE_SERVICE),
        "run_name": run.info.run_name,
        SELECTION_METRIC: f"{run.data.metrics[SELECTION_METRIC]:.4f}",
        "status": "candidate",
    }.items():
        c.set_model_version_tag(MODEL_NAME, mv.version, key, value)

    c.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} from run '{run.info.run_name}' -> @challenger")


def promote(c: MlflowClient, version: str | None, force: bool) -> None:
    candidate = c.get_model_version(MODEL_NAME, version) if version else alias_version(c, "challenger")
    if candidate is None:
        raise SystemExit("Nothing to promote: no @challenger and no --version given")
    champion = alias_version(c, "champion")

    if champion and champion.version == candidate.version:
        print(f"v{candidate.version} is already @champion")
        return

    if champion and not force:
        new, old = version_metric(c, candidate), version_metric(c, champion)
        if new >= old:
            c.set_model_version_tag(MODEL_NAME, candidate.version, "status", "rejected")
            print(f"Gate failed: v{candidate.version} {SELECTION_METRIC}={new:.3f} "
                  f"is not better than champion v{champion.version} ({old:.3f}). Champion unchanged.")
            return
        print(f"Gate passed: v{candidate.version} {SELECTION_METRIC}={new:.3f} < v{champion.version} {old:.3f}")

    if champion:
        c.set_registered_model_alias(MODEL_NAME, "previous", champion.version)
        c.set_model_version_tag(MODEL_NAME, champion.version, "status", "retired")
    c.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
    c.set_model_version_tag(MODEL_NAME, candidate.version, "status", "production")
    if (ch := alias_version(c, "challenger")) and ch.version == candidate.version:
        c.delete_registered_model_alias(MODEL_NAME, "challenger")
    print(f"v{candidate.version} -> @champion" + (f" (v{champion.version} -> @previous)" if champion else ""))


def rollback(c: MlflowClient) -> None:
    previous, champion = alias_version(c, "previous"), alias_version(c, "champion")
    if previous is None:
        raise SystemExit("No @previous version to roll back to")
    c.set_registered_model_alias(MODEL_NAME, "champion", previous.version)
    c.set_model_version_tag(MODEL_NAME, previous.version, "status", "production")
    if champion:
        c.set_registered_model_alias(MODEL_NAME, "previous", champion.version)
        c.set_model_version_tag(MODEL_NAME, champion.version, "status", "rolled_back")
    print(f"Rolled back: v{previous.version} -> @champion" + (f", v{champion.version} -> @previous" if champion else ""))


def archive(c: MlflowClient, version: str) -> None:
    mv = c.get_model_version(MODEL_NAME, version)
    if "champion" in mv.aliases:
        raise SystemExit(f"v{version} is @champion; promote another version before archiving it")
    for alias in mv.aliases:
        c.delete_registered_model_alias(MODEL_NAME, alias)
    c.set_model_version_tag(MODEL_NAME, version, "status", "archived")
    print(f"Archived v{version}")


def list_versions(c: MlflowClient) -> None:
    versions = sorted(c.search_model_versions(f"name = '{MODEL_NAME}'"), key=lambda v: int(v.version))
    # Search results omit aliases; the registered model holds the alias -> version map.
    by_version: dict[str, list[str]] = {}
    for alias, ver in c.get_registered_model(MODEL_NAME).aliases.items():
        by_version.setdefault(str(ver), []).append(f"@{alias}")
    print(f"{'ver':<4} {'aliases':<22} {'status':<12} {SELECTION_METRIC:<10} run")
    for v in versions:
        aliases = ",".join(sorted(by_version.get(v.version, []))) or "-"
        print(f"v{v.version:<3} {aliases:<22} {v.tags.get('status', '-'):<12} "
              f"{v.tags.get(SELECTION_METRIC, '-'):<10} {v.tags.get('run_name', '-')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register", help="register a run's model and mark it @challenger")
    who = reg.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-name")
    who.add_argument("--best", action="store_true", help=f"lightgbm run with the lowest {SELECTION_METRIC}")
    pro = sub.add_parser("promote", help="make @challenger (or --version) the @champion if it passes the gate")
    pro.add_argument("--version")
    pro.add_argument("--force", action="store_true", help="skip the metric gate")
    sub.add_parser("rollback", help="swap @champion and @previous")
    arc = sub.add_parser("archive", help="remove aliases and mark a version archived")
    arc.add_argument("--version", required=True)
    sub.add_parser("list")
    args = parser.parse_args()

    c = client()
    if args.cmd == "register":
        register(c, None if args.best else args.run_name)
    elif args.cmd == "promote":
        promote(c, args.version, args.force)
    elif args.cmd == "rollback":
        rollback(c)
    elif args.cmd == "archive":
        archive(c, args.version)
    list_versions(c)


if __name__ == "__main__":
    main()
