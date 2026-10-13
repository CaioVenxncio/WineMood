"""Command-line interface: ``winemood data {download,verify,validate,split,eda}``."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from winemood.config import REPO_ROOT, Config, load_config
from winemood.data.acquire import AcquisitionError, download_dataset, verify_checksum
from winemood.data.ingest import load_reviews
from winemood.data.labels import assign_labels, get_scheme
from winemood.data.protocol import (
    drop_exact_duplicates,
    load_split_ids,
    run_split_protocol,
    write_manifest_ids,
)
from winemood.data.schema import validate_reviews
from winemood.logging_utils import configure_logging, get_logger
from winemood.models.artifact import ArtifactVerificationError
from winemood.seeding import set_global_seed

logger = get_logger(__name__)

SPLIT_STRATEGIES = {
    "grouped_v1": "grouped",
    "taster_holdout_v1": "taster_holdout",
    # H3 comparison split (Phase 3): same rows and sizes as taster_holdout_v1.
    "taster_random_v1": "taster_random",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _verified_csv(cfg: Config) -> Path:
    verify_checksum(cfg.data.raw_csv, cfg.data.sha256 or None)
    return cfg.data.raw_csv


def cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    if not cfg.data.url:
        logger.error("no [data] url configured")
        return 2
    digest = download_dataset(
        cfg.data.url, cfg.data.raw_csv, cfg.data.sha256 or None, overwrite=args.force
    )
    print(f"OK {cfg.data.raw_csv} sha256={digest}")
    return 0


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    digest = verify_checksum(cfg.data.raw_csv, cfg.data.sha256 or None)
    print(f"OK sha256={digest}" + ("" if cfg.data.sha256 else " (no pinned checksum)"))
    return 0


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    df = load_reviews(_verified_csv(cfg), cfg.data.id_column)
    report = validate_reviews(df, cfg.data.text_column, cfg.data.target_column)
    out = cfg.data.reports_dir / "validation.json"
    _write_json(out, report.to_dict())
    for w in report.warnings:
        print(f"WARNING {w}")
    for e in report.errors:
        print(f"ERROR {e}")
    print(f"{'OK' if report.ok else 'FAILED'}: {report.n_rows} rows; report written to {out}")
    return 0 if report.ok else 1


def _split_config(cfg: Config, strategy: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "seed": cfg.seed,
        "split": {k: v for k, v in asdict(cfg.split).items() if k not in ("deduplicate",)},
        "neardup": asdict(cfg.neardup),
    }


def cmd_split(cfg: Config, args: argparse.Namespace) -> int:
    set_global_seed(cfg.seed, include_tensorflow=False)
    df = load_reviews(_verified_csv(cfg), cfg.data.id_column)
    scheme = get_scheme(cfg.split.stratify_on)
    strata = assign_labels(df[cfg.data.target_column], scheme).to_numpy()
    neardup = (
        {k: v for k, v in asdict(cfg.neardup).items() if k != "enabled"}
        if cfg.neardup.enabled
        else None
    )
    manifests: dict[str, Any] = {}
    for name, strategy in SPLIT_STRATEGIES.items():
        result = run_split_protocol(
            df,
            name=name,
            text_column=cfg.data.text_column,
            target_column=cfg.data.target_column,
            id_column=cfg.data.id_column,
            strata=strata,
            strata_names=scheme.class_names,
            dataset_sha256=cfg.data.sha256,
            config=_split_config(cfg, strategy),
            seed=cfg.seed,
            neardup=neardup,
            group_by_title=cfg.split.group_by_title,
            n_folds=cfg.split.n_folds,
            test_folds=cfg.split.test_folds,
            validation_folds=cfg.split.validation_folds,
            strategy=strategy,
        )
        manifests[name] = result.manifest.to_dict()
        if not args.check:
            split_dir = cfg.data.processed_dir / "splits" / name
            write_manifest_ids(
                split_dir,
                result.data,
                result.data["split"].to_numpy(),
                cfg.data.id_column,
                groups=result.data["group"].to_numpy(),
            )
            if name == "grouped_v1":
                _write_pairs(cfg, result)
        print(f"{name}: {result.manifest.counts}")

    if args.check:
        if not cfg.data.fingerprints.is_file():
            print(f"FAILED: no committed fingerprints at {cfg.data.fingerprints}")
            return 1
        committed = json.loads(cfg.data.fingerprints.read_text(encoding="utf-8"))
        ok = all(
            committed.get(n, {}).get("id_sha256") == m["id_sha256"]
            and committed.get(n, {}).get("config_hash") == m["config_hash"]
            for n, m in manifests.items()
        )
        print("OK: splits reproduce the committed fingerprints" if ok else "FAILED: mismatch")
        return 0 if ok else 1
    _write_json(cfg.data.fingerprints, manifests)
    print(f"fingerprints written to {cfg.data.fingerprints}")
    return 0


def _write_pairs(cfg: Config, result: Any) -> None:
    out = cfg.data.reports_dir / "near_duplicate_pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = result.data
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["review_id_a", "review_id_b", "jaccard", "split_a", "split_b"])
        for i, j, sim in result.near_dup_pairs:
            writer.writerow(
                [
                    int(data.at[i, cfg.data.id_column]),
                    int(data.at[j, cfg.data.id_column]),
                    f"{sim:.4f}",
                    data.at[i, "split"],
                    data.at[j, "split"],
                ]
            )


def cmd_eda(cfg: Config, args: argparse.Namespace) -> int:
    from winemood.analysis import plots
    from winemood.analysis.eda import descriptive_summary, modelling_summary, word_counts
    from winemood.analysis.report import render

    set_global_seed(cfg.seed, include_tensorflow=False)
    text, target, idc = cfg.data.text_column, cfg.data.target_column, cfg.data.id_column
    raw = load_reviews(_verified_csv(cfg), idc)
    dedup, _ = drop_exact_duplicates(raw, text, target, idc)

    split_dir = cfg.data.processed_dir / "splits" / "grouped_v1"
    train_ids = load_split_ids(split_dir, "train")
    val_ids = load_split_ids(split_dir, "validation")
    train = dedup[dedup[idc].isin(train_ids)]
    validation = dedup[dedup[idc].isin(val_ids)]
    if len(train) != len(train_ids) or len(validation) != len(val_ids):
        logger.error("split manifests do not match the dataset; re-run 'winemood data split'")
        return 1

    summary = descriptive_summary(raw, dedup, text_column=text, target_column=target)
    model = modelling_summary(
        train,
        validation,
        text_column=text,
        target_column=target,
        min_class_fraction=cfg.targets.min_class_fraction,
        seed=cfg.seed,
    )
    manifests = (
        json.loads(cfg.data.fingerprints.read_text(encoding="utf-8"))
        if cfg.data.fingerprints.is_file()
        else {}
    )
    out_dir = cfg.data.reports_dir / "eda"
    _write_json(out_dir / "summary.json", {"descriptive": summary, "modelling": model})
    figures = plots.make_all(
        summary,
        model,
        word_counts(dedup[text]).to_numpy(),
        train[target].to_numpy(),
        word_counts(train[text]).to_numpy(),
        out_dir / "figures",
    )
    report_md = render(summary, model, manifests, figure_dir="figures")
    (out_dir / "data-report.md").write_text(report_md, encoding="utf-8")
    print(f"EDA written to {out_dir} ({len(figures)} figures)")

    if args.publish:
        docs = Path(args.docs_dir)
        (docs / "figures").mkdir(parents=True, exist_ok=True)
        for f in figures:
            shutil.copy2(f, docs / "figures" / f.name)
        (docs / "data-report.md").write_text(report_md, encoding="utf-8")
        # Aggregates only (counts, n-grams, statistics) for the application.
        _write_json(
            cfg.experiments.results_dir / "eda" / "summary.json",
            {"descriptive": summary, "modelling": model},
        )
        print(f"published to {docs / 'data-report.md'} and results/eda/summary.json")
    if not model["merge_rule"]["matches_frozen_scheme"]:
        print("WARNING: merge rule no longer matches we_category_merged_v1; review labels.py")
        return 1
    return 0


def _runner(cfg: Config):
    from winemood.experiments.runner import Runner
    from winemood.experiments.spec import load_plan

    return Runner(cfg, load_plan(cfg.experiments.spec))


def cmd_exp_list(cfg: Config, args: argparse.Namespace) -> int:
    runner = _runner(cfg)
    scope = set(runner.plan.test_scope)
    for spec in runner.plan.experiments.values():
        done = "done" if runner.validation_path(spec.id).is_file() else "    "
        print(
            f"{done}  {spec.id:28s} {spec.task}  {spec.family:7s} {spec.split:18s} "
            f"{'TEST' if spec.id in scope else 'val '}  {spec.description}"
        )
    return 0


def cmd_exp_run(cfg: Config, args: argparse.Namespace) -> int:
    runner = _runner(cfg)
    set_global_seed(cfg.seed, include_tensorflow=False)
    ids = list(runner.plan.experiments) if args.all else args.ids
    if args.family:
        ids = [i for i in ids if runner.plan.get(i).family in args.family]
    for exp_id in ids:
        if args.skip_done and runner.validation_path(exp_id).is_file():
            print(f"skip {exp_id} (already run)")
            continue
        result = runner.run(exp_id, pilot=args.pilot)
        if args.pilot:
            epochs = result["pilot_epoch_seconds"]
            print(
                f"PILOT {exp_id}: 1 epoch = {epochs[0]:.0f} s (seq_len {result['seq_len']}); "
                f"{result['n_configs']} configs + {result['n_seeds'] - 1} extra seeds"
            )
        else:
            task = runner.plan.tasks[result["task"]]
            value = result["validation_metrics"][task.selection_metric]
            print(f"{exp_id}: validation {task.selection_metric} = {value:.4f}")
    return 0


def cmd_exp_select(cfg: Config, args: argparse.Namespace) -> int:
    selection = _runner(cfg).select(allow_missing=args.allow_missing)
    print(f"final model: {selection['final_model']}")
    for c in selection["comparisons_on_validation"]:
        print(f"  {c}")
    print(
        f"frozen {sum(e['status'] == 'frozen' for e in selection['test_scope'])} test-scope entries"
    )
    return 0


def cmd_exp_final(cfg: Config, args: argparse.Namespace) -> int:
    outputs = _runner(cfg).final(allow_repeat=args.allow_repeat)
    print(f"evaluated {len(outputs)} entries on test; see results/test/ and the access log")
    return 0


def cmd_exp_report(cfg: Config, args: argparse.Namespace) -> int:
    from winemood.evaluation.report import build_report

    out = build_report(_runner(cfg), docs_dir=Path(args.docs_dir))
    print(f"report written: {out}")
    return 0


def cmd_model_package(cfg: Config, args: argparse.Namespace) -> int:
    from winemood.models.artifact import package

    runner = _runner(cfg)
    spec = runner.plan.get(args.experiment)
    out = Path(args.out) if args.out else REPO_ROOT / "dist" / "models" / f"{args.experiment}.zip"
    manifest = package(runner.artifact_dir(spec), out, cfg.experiments.results_dir)
    print(f"packaged {manifest['experiment']} ({len(manifest['files'])} files) -> {out}")
    return 0


def cmd_model_install(cfg: Config, args: argparse.Namespace) -> int:
    from winemood.models.artifact import install

    manifest = install(Path(args.archive), Path(args.target))
    print(f"installed and verified {manifest['experiment']} -> {args.target}")
    return 0


def cmd_model_verify(cfg: Config, args: argparse.Namespace) -> int:
    from winemood.models.artifact import verify

    manifest = verify(Path(args.directory))
    print(f"OK {manifest['experiment']}: {len(manifest['files'])} files verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="winemood", description=__doc__)
    parser.add_argument("--config", default=None, help="path to a TOML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    top = parser.add_subparsers(dest="group", required=True)
    data = top.add_parser("data", help="dataset acquisition, validation, splitting and EDA")
    sub = data.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and verify the raw CSV")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=cmd_download)
    sub.add_parser("verify", help="verify the raw CSV checksum").set_defaults(func=cmd_verify)
    sub.add_parser("validate", help="write a validation report").set_defaults(func=cmd_validate)
    p = sub.add_parser("split", help="run the split protocol and write manifests")
    p.add_argument(
        "--check", action="store_true", help="recompute and compare with committed fingerprints"
    )
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("eda", help="exploratory analysis (requires 'split' first)")
    p.add_argument("--publish", action="store_true", help="copy report and figures to docs/")
    p.add_argument("--docs-dir", default=str(REPO_ROOT / "docs"))
    p.set_defaults(func=cmd_eda)

    exp = top.add_parser("exp", help="experiments: run, select, final test evaluation, report")
    esub = exp.add_subparsers(dest="command", required=True)
    esub.add_parser("list", help="list experiments and status").set_defaults(func=cmd_exp_list)
    p = esub.add_parser("run", help="train + validate experiments (never touches test)")
    p.add_argument("ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--family", action="append", choices=["linear", "lstm", "encoder"])
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--pilot", action="store_true", help="LSTM: time one epoch, save nothing")
    p.set_defaults(func=cmd_exp_run)
    p = esub.add_parser("select", help="freeze the test-scope configurations")
    p.add_argument("--allow-missing", action="store_true", help="record unrun entries as 'not run'")
    p.set_defaults(func=cmd_exp_select)
    p = esub.add_parser("final", help="single logged evaluation of the frozen scope on test")
    p.add_argument("--allow-repeat", action="store_true")
    p.set_defaults(func=cmd_exp_final)
    p = esub.add_parser("report", help="statistics, error analysis, figures and docs")
    p.add_argument("--docs-dir", default=str(REPO_ROOT / "docs"))
    p.set_defaults(func=cmd_exp_report)

    model = top.add_parser("model", help="package, install and verify model artifacts")
    msub = model.add_subparsers(dest="command", required=True)
    p = msub.add_parser("package", help="zip an experiment's model with a checksum manifest")
    p.add_argument("experiment", nargs="?", default="E04_tfidf_word")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_model_package)
    p = msub.add_parser("install", help="verify and extract a packaged model")
    p.add_argument("archive")
    p.add_argument("--target", default=str(REPO_ROOT / "models" / "e04"))
    p.set_defaults(func=cmd_model_install)
    p = msub.add_parser("verify", help="check an installed model against its manifest")
    p.add_argument("directory", nargs="?", default=str(REPO_ROOT / "models" / "e04"))
    p.set_defaults(func=cmd_model_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO")
    cfg = load_config(args.config)
    try:
        return int(args.func(cfg, args))
    except (AcquisitionError, FileNotFoundError, PermissionError) as exc:
        logger.error(str(exc))
        return 2
    except ArtifactVerificationError as exc:
        logger.error(str(exc))
        return 3


if __name__ == "__main__":
    sys.exit(main())
