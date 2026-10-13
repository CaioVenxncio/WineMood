"""Experiment execution: validation grids, seeds, selection and final test evaluation.

Discipline enforced here:
* fitting uses the training frame only; grids are scored on validation;
* ``run`` never touches the test partition (``load_split`` refuses it);
* ``select`` freezes the configurations in the pre-declared test scope;
* ``final`` evaluates exactly that scope once, behind the test-access guard.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from winemood.config import Config
from winemood.data.labels import RegressionTarget, assign_labels, get_scheme
from winemood.evaluation.bootstrap import paired_difference, regression_stats, REGRESSION_METRICS
from winemood.evaluation.metrics import (
    classification_metrics,
    clip_scores,
    regression_metrics,
    scores_to_classes,
)
from winemood.experiments.data import SplitData, load_split
from winemood.experiments.spec import ExperimentPlan, ExperimentSpec, TaskSpec, config_hash
from winemood.experiments.tracking import (
    Timer,
    append_registry,
    guard_test_access,
    log_test_access,
    now_iso,
    read_json,
    run_metadata,
    write_json,
)
from winemood.features.pipeline import FeaturePipeline, FeatureSpec
from winemood.features.text import sequence_text
from winemood.logging_utils import get_logger
from winemood.models.linear import Head
from winemood.preprocessing import TextPreprocessor, preprocessing_config

logger = get_logger(__name__)

# Complexity order used by the model-selection rule (docs/model-selection.md).
T1_CANDIDATES = ("E04_tfidf_word", "E12_encoder", "E10_lstm")


class ExperimentError(RuntimeError):
    pass


def seed_summary(y: np.ndarray, seed_preds: dict[int, np.ndarray]) -> dict[str, Any]:
    """Headline metrics for multi-seed models: each metric is the mean over seeds
    (protocol §6), with its across-seed SD; ``ensemble_*`` keys describe the
    seed-averaged prediction and are diagnostics only."""
    per_seed = [regression_metrics(y, p) for p in seed_preds.values()]
    keys = [k for k in per_seed[0] if k != "n"]
    out: dict[str, Any] = {"n": per_seed[0]["n"], "n_seeds": len(per_seed)}
    for k in keys:
        values = np.array([m[k] for m in per_seed], dtype=float)
        out[k] = float(values.mean())
        out[f"{k}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ensemble = regression_metrics(y, np.mean(list(seed_preds.values()), axis=0))
    out.update({f"ensemble_{k}": v for k, v in ensemble.items() if k != "n"})
    return out


class Runner:
    def __init__(self, cfg: Config, plan: ExperimentPlan, *, stopwords=None) -> None:
        self.cfg = cfg
        self.plan = plan
        self.stopwords = stopwords
        self.exp_dir = cfg.experiments.dir
        self.results = cfg.experiments.results_dir
        self._encoders: dict[str, Any] = {}

    # -- paths -------------------------------------------------------------------
    def artifact_dir(self, spec: ExperimentSpec) -> Path:
        return self.exp_dir / spec.split / spec.id

    def validation_path(self, exp_id: str) -> Path:
        return self.results / "validation" / f"{exp_id}.json"

    def test_path(self, exp_id: str) -> Path:
        return self.results / "test" / f"{exp_id}.json"

    # -- helpers -----------------------------------------------------------------
    def task(self, spec: ExperimentSpec) -> TaskSpec:
        return self.plan.tasks[spec.task]

    def targets(self, task: TaskSpec, frame: pd.DataFrame) -> np.ndarray:
        points = frame[self.cfg.data.target_column]
        if isinstance(task.scheme, RegressionTarget):
            return points.to_numpy(dtype=float)
        return assign_labels(points, task.scheme).to_numpy()

    def base_config(self, spec: ExperimentSpec) -> dict[str, Any]:
        if not spec.base:
            return {}
        path = self.validation_path(spec.base)
        if not path.is_file():
            raise ExperimentError(f"Run base experiment '{spec.base}' before '{spec.id}'")
        return read_json(path)["selected_config"]

    def _score(self, task: TaskSpec, y, pred, proba=None) -> dict[str, Any]:
        if task.is_regression:
            return regression_metrics(y, pred)
        return classification_metrics(y, pred, task.scheme, proba)

    def _better(self, task: TaskSpec, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return a > b if task.higher_is_better else a < b

    def _encoder(self, name: str, revision: str | None = None):
        """Encoders are cached per name; a pinned ``revision`` (from a saved run)
        replaces any previously loaded copy so test-time embeddings use exactly
        the weights recorded at validation time."""
        cached = self._encoders.get(name)
        if cached is None or (revision and cached.revision != revision):
            from winemood.models.encoder import FrozenEncoder

            self._encoders[name] = FrozenEncoder(name, revision=revision, threads=self.cfg_threads())
        return self._encoders[name]

    @staticmethod
    def cfg_threads() -> int | None:
        import os

        return os.cpu_count()

    def _embeddings(self, pipe: FeaturePipeline, frame: pd.DataFrame) -> np.ndarray | None:
        if pipe.spec.kind != "encoder":
            return None
        from winemood.models.encoder import cached_embeddings

        encoder = self._encoder(pipe.spec.encoder_model)
        return cached_embeddings(
            encoder,
            pipe.raw_masked_texts(frame),
            frame[self.cfg.data.id_column].to_numpy(),
            self.exp_dir / "cache" / "encoder",
        )

    # -- run (train + validation only) ---------------------------------------------
    def run(self, exp_id: str, *, pilot: bool = False) -> dict[str, Any]:
        spec = self.plan.get(exp_id)
        task = self.task(spec)
        meta = run_metadata(spec.id, split=spec.split, task=spec.task, family=spec.family)
        data = load_split(self.cfg, spec.split)
        meta["dataset_sha256"] = data.dataset_sha256
        meta["split_id_sha256"] = data.id_sha256
        with Timer() as timer:
            if spec.family == "lstm":
                result = self._run_lstm(spec, task, data, pilot=pilot)
            else:
                result = self._run_linear(spec, task, data)
        meta.update({"finished": now_iso(), "seconds": round(timer.seconds, 1), "status": "ok"})
        if pilot:
            return {**result, "meta": meta}
        payload = {
            "experiment": spec.id,
            "description": spec.description,
            "hypothesis": spec.hypothesis,
            "task": spec.task,
            "target": task.target,
            "family": spec.family,
            "split": spec.split,
            "base": spec.base,
            **result,
            "meta": meta,
        }
        write_json(self.validation_path(spec.id), payload)
        write_json(self.artifact_dir(spec) / "run.json", payload)
        append_registry(
            self.exp_dir,
            {
                "run_id": meta["run_id"],
                "experiment": spec.id,
                "split": spec.split,
                "status": "ok",
                "seconds": meta["seconds"],
                "selected_hash": result["config_hash"],
                "metric": task.selection_metric,
                "value": result["validation_metrics"][task.selection_metric],
            },
        )
        logger.info(
            "experiment complete",
            extra={
                "experiment": spec.id,
                task.selection_metric: result["validation_metrics"][task.selection_metric],
                "seconds": meta["seconds"],
            },
        )
        return payload

    def _predict_head(self, task: TaskSpec, head: Head, x):
        if task.is_regression:
            return clip_scores(head.predict(x)), None
        proba = head.predict_proba(x)
        pred = head.predict(x) if head.kind == "stratified" else np.argmax(proba, axis=1)
        return pred, proba

    def _run_linear(self, spec: ExperimentSpec, task: TaskSpec, data: SplitData) -> dict[str, Any]:
        train, val = data.frames["train"], data.frames["validation"]
        y_tr, y_va = self.targets(task, train), self.targets(task, val)
        n_classes = None if task.is_regression else task.scheme.n_classes
        configs = spec.grid_configs(self.base_config(spec))
        by_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in configs:
            by_features[json.dumps(c["features"], sort_keys=True)].append(c)

        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for feature_key, group in by_features.items():
            pipe = FeaturePipeline(FeatureSpec(**json.loads(feature_key)), stopwords=self.stopwords)
            with Timer() as t_feat:
                pipe.fit(train)
                x_tr = pipe.transform(train, self._embeddings(pipe, train))
                x_va = pipe.transform(val, self._embeddings(pipe, val))
            for c in group:
                head_cfg = dict(c["head"])
                kind = head_cfg.pop("kind")
                if head_cfg.get("class_weight") == "none":
                    head_cfg["class_weight"] = None
                with Timer() as t_fit:
                    head = Head(kind, head_cfg, seed=self.cfg.seed).fit(x_tr, y_tr, n_classes)
                pred, proba = self._predict_head(task, head, x_va)
                metrics = self._score(task, y_va, pred, proba)
                value = metrics[task.selection_metric]
                rows.append(
                    {
                        "config_hash": config_hash(c),
                        "config": c,
                        task.selection_metric: value,
                        "n_features": int(x_tr.shape[1]),
                        "feature_seconds": round(t_feat.seconds, 2),
                        "fit_seconds": round(t_fit.seconds, 2),
                    }
                )
                if best is None or self._better(task, value, best["value"]):
                    best = {"config": c, "pipe": pipe, "head": head, "pred": pred,
                            "proba": proba, "metrics": metrics, "value": value}
            logger.info("feature block evaluated", extra={"experiment": spec.id, "configs": len(group)})

        assert best is not None
        out_dir = self.artifact_dir(spec) / "model"
        best["pipe"].save(out_dir)
        best["head"].save(out_dir)
        write_json(out_dir / "config.json", best["config"])
        self._write_predictions(
            self.artifact_dir(spec) / "validation_predictions.csv", val, y_va, best["pred"], best["proba"]
        )
        pd.DataFrame(
            [{k: (json.dumps(v) if k == "config" else v) for k, v in r.items()} for r in rows]
        ).to_csv(self.artifact_dir(spec) / "grid.csv", index=False)
        extra = {}
        if spec.family == "encoder" and best["pipe"].spec.encoder_model:
            extra["encoder"] = self._encoder(best["pipe"].spec.encoder_model).info()
        return {
            "selected_config": best["config"],
            "config_hash": config_hash(best["config"]),
            "n_configs": len(configs),
            "grid": rows,
            "validation_metrics": best["metrics"],
            "seeds": [self.cfg.seed],
            **extra,
        }

    # -- LSTM -----------------------------------------------------------------------
    def _sequence_texts(self, config: dict[str, Any], frame: pd.DataFrame) -> list[str]:
        prep = TextPreprocessor(
            preprocessing_config(config["preprocessing"], config.get("mask_numbers", "none")),
            stopwords=self.stopwords,
        )
        return [sequence_text(prep(t)) for t in frame[self.cfg.data.text_column].astype(str)]

    def _run_lstm(
        self, spec: ExperimentSpec, task: TaskSpec, data: SplitData, *, pilot: bool = False
    ) -> dict[str, Any]:
        from winemood.models import lstm

        if not task.is_regression:
            raise ExperimentError("The LSTM is implemented for score regression (T1) only")
        train, val = data.frames["train"], data.frames["validation"]
        y_tr, y_va = self.targets(task, train), self.targets(task, val)
        configs = spec.grid_configs(self.base_config(spec))
        first_seed = spec.seeds[0]

        if pilot:
            c = configs[0]
            res = lstm.train_lstm(
                self._sequence_texts(c, train), y_tr, self._sequence_texts(c, val), y_va,
                c["lstm"], first_seed, max_epochs=1,
            )
            return {"pilot_epoch_seconds": res.epoch_seconds, "seq_len": res.seq_len,
                    "n_configs": len(configs), "n_seeds": len(spec.seeds)}

        rows, best = [], None
        text_cache: dict[str, tuple[list[str], list[str]]] = {}
        for c in configs:
            key = json.dumps({k: c[k] for k in ("preprocessing", "mask_numbers") if k in c})
            if key not in text_cache:
                text_cache[key] = (self._sequence_texts(c, train), self._sequence_texts(c, val))
            tr_texts, va_texts = text_cache[key]
            res = lstm.train_lstm(tr_texts, y_tr, va_texts, y_va, c["lstm"], first_seed)
            pred = clip_scores(lstm.predict(res.model, va_texts))
            metrics = regression_metrics(y_va, pred)
            rows.append({"config_hash": config_hash(c), "config": c, "mae": metrics["mae"],
                         "best_epoch": res.best_epoch, "seq_len": res.seq_len,
                         "seconds": round(res.seconds, 1),
                         "epoch_seconds": [round(s, 1) for s in res.epoch_seconds]})
            logger.info("lstm config evaluated",
                        extra={"experiment": spec.id, "mae": metrics["mae"], "seconds": round(res.seconds)})
            if best is None or metrics["mae"] < best["mae"]:
                best = {"config": c, "result": res, "pred": pred, "mae": metrics["mae"], "key": key}

        assert best is not None
        tr_texts, va_texts = text_cache[best["key"]]
        seed_runs, preds = [], {}
        for seed in spec.seeds:
            if seed == first_seed:
                res, pred = best["result"], best["pred"]
            else:
                res = lstm.train_lstm(tr_texts, y_tr, va_texts, y_va, best["config"]["lstm"], seed)
                pred = clip_scores(lstm.predict(res.model, va_texts))
            lstm.save(res.model, self.artifact_dir(spec) / f"model_seed{seed}")
            write_json(self.artifact_dir(spec) / f"model_seed{seed}" / "history.json", res.history)
            preds[seed] = pred
            m = regression_metrics(y_va, pred)
            seed_runs.append({"seed": seed, "mae": m["mae"], "rmse": m["rmse"],
                              "best_epoch": res.best_epoch, "seconds": round(res.seconds, 1)})
            logger.info("lstm seed done", extra={"experiment": spec.id, "seed": seed, "mae": m["mae"]})
        write_json(self.artifact_dir(spec) / "config.json", best["config"])
        pred_matrix = np.vstack([preds[s] for s in spec.seeds])
        self._write_predictions(
            self.artifact_dir(spec) / "validation_predictions.csv", val, y_va,
            pred_matrix.mean(axis=0), None, seed_preds={s: preds[s] for s in spec.seeds},
        )
        return {
            "selected_config": best["config"],
            "config_hash": config_hash(best["config"]),
            "n_configs": len(configs),
            "grid": rows,
            "seed_runs": seed_runs,
            "seeds": list(spec.seeds),
            "validation_metrics": seed_summary(y_va, preds),
        }

    # -- predictions I/O ------------------------------------------------------------------
    def _write_predictions(self, path: Path, frame: pd.DataFrame, y, pred, proba=None,
                           seed_preds: dict[int, np.ndarray] | None = None) -> None:
        idc = self.cfg.data.id_column
        table = pd.DataFrame({idc: frame[idc].to_numpy(), "y_true": y, "y_pred": pred})
        if proba is not None:
            for k in range(proba.shape[1]):
                table[f"p{k}"] = proba[:, k]
        for seed, p in (seed_preds or {}).items():
            table[f"pred_seed{seed}"] = p
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    def load_predictions(self, exp_id: str, part: str) -> pd.DataFrame:
        spec = self.plan.get(exp_id)
        return pd.read_csv(self.artifact_dir(spec) / f"{part}_predictions.csv")

    # -- selection -------------------------------------------------------------------------
    def select(self, *, allow_missing: bool = False, n_resamples: int | None = None) -> dict[str, Any]:
        n_resamples = n_resamples or self.cfg.experiments.bootstrap_resamples
        scope = []
        for exp_id in self.plan.test_scope:
            path = self.validation_path(exp_id)
            if not path.is_file():
                if not allow_missing:
                    raise ExperimentError(f"'{exp_id}' has no validation result; run it first")
                scope.append({"experiment": exp_id, "status": "not run"})
                continue
            res = read_json(path)
            scope.append({"experiment": exp_id, "status": "frozen", "config_hash": res["config_hash"],
                          "split": res["split"], "task": res["task"], "seeds": res.get("seeds")})
        final_model, comparisons = self._choose_final_model(n_resamples)
        selection = {
            "frozen": True,
            "frozen_at": now_iso(),
            "rule": (
                "Primary criterion: validation MAE on grouped_v1 (T1). Candidates in order of "
                "complexity: " + ", ".join(T1_CANDIDATES) + ". A more complex candidate replaces the "
                "current choice only if it has lower validation MAE AND the 95% paired cluster-"
                "bootstrap CI of the MAE difference excludes 0."
            ),
            "final_model": final_model,
            "comparisons_on_validation": comparisons,
            "test_scope": scope,
        }
        write_json(self.results / "selection.json", selection)
        return selection

    def _val_stats(self, exp_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        path = self.validation_path(exp_id)
        if not path.is_file():
            return None
        table = self.load_predictions(exp_id, "validation")
        seed_cols = [c for c in table.columns if c.startswith("pred_seed")]
        preds = table[seed_cols].to_numpy().T if seed_cols else table["y_pred"].to_numpy()[None, :]
        return table[self.cfg.data.id_column].to_numpy(), table["y_true"].to_numpy(), preds

    def _choose_final_model(self, n_resamples: int) -> tuple[str | None, list[dict[str, Any]]]:
        groups_all = None
        current, comparisons = None, []
        for exp_id in T1_CANDIDATES:
            loaded = self._val_stats(exp_id)
            if loaded is None:
                comparisons.append({"candidate": exp_id, "status": "not run"})
                continue
            ids, y, preds = loaded
            if groups_all is None:
                spec = self.plan.get(exp_id)
                groups_all = load_split(self.cfg, spec.split).groups
            groups = groups_all.loc[ids].to_numpy()
            if current is None:
                current = (exp_id, ids, regression_stats(y, preds))
                continue
            cur_id, cur_ids, cur_stats = current
            if not np.array_equal(cur_ids, ids):
                raise ExperimentError("validation predictions are not aligned by review_id")
            stats = regression_stats(y, preds)
            diff = paired_difference(cur_stats, stats, groups, REGRESSION_METRICS["mae"],
                                     n_resamples=n_resamples, seed=self.cfg.seed)
            switch = diff.estimate > 0 and diff.excludes_zero()
            comparisons.append({"current": cur_id, "candidate": exp_id,
                                "mae_current_minus_candidate": diff.as_dict(), "switch": switch})
            if switch:
                current = (exp_id, ids, stats)
        return (current[0] if current else None), comparisons

    # -- final test evaluation -------------------------------------------------------------
    def predict_saved(self, spec: ExperimentSpec, frame: pd.DataFrame) -> dict[str, Any]:
        task = self.task(spec)
        directory = self.artifact_dir(spec)
        if spec.family == "lstm":
            from winemood.models import lstm

            config = read_json(directory / "config.json")
            texts = self._sequence_texts(config, frame)
            seeds = read_json(directory / "run.json")["seeds"]
            seed_preds = {s: clip_scores(lstm.predict(lstm.load(directory / f"model_seed{s}"), texts))
                          for s in seeds}
            return {"pred": np.mean(list(seed_preds.values()), axis=0), "proba": None,
                    "seed_preds": seed_preds}
        pipe = FeaturePipeline.load(directory / "model", stopwords=self.stopwords)
        head = Head.load(directory / "model")
        if pipe.spec.kind == "encoder":
            pinned = read_json(self.validation_path(spec.id)).get("encoder", {}).get("revision")
            self._encoder(pipe.spec.encoder_model, revision=pinned)
        x = pipe.transform(frame, self._embeddings(pipe, frame))
        pred, proba = self._predict_head(task, head, x)
        return {"pred": pred, "proba": proba, "seed_preds": None}

    def final(self, *, allow_repeat: bool = False) -> list[dict[str, Any]]:
        selection = read_json(self.results / "selection.json")
        outputs = []
        for entry in selection["test_scope"]:
            if entry["status"] != "frozen":
                logger.warning("skipping experiment that was not run", extra={"experiment": entry["experiment"]})
                continue
            outputs.append(self._final_one(entry, allow_repeat=allow_repeat))
        if selection.get("final_model"):
            outputs.extend(self._final_derived(selection["final_model"], allow_repeat=allow_repeat))
        return outputs

    def _final_one(self, entry: dict[str, Any], *, allow_repeat: bool) -> dict[str, Any]:
        spec = self.plan.get(entry["experiment"])
        task = self.task(spec)
        saved = read_json(self.validation_path(spec.id))
        if saved["config_hash"] != entry["config_hash"]:
            raise ExperimentError(f"{spec.id}: artifacts changed after selection was frozen")
        repeat = bool(allow_repeat)
        guard_test_access(self.results, spec.id, entry["config_hash"], allow_repeat=allow_repeat)
        data = load_split(self.cfg, spec.split, ("test",), allow_test=True)
        log_test_access(self.results, spec.id, entry["config_hash"], spec.split, repeat=repeat)
        test = data.frames["test"]
        y = self.targets(task, test)
        out = self.predict_saved(spec, test)
        seed_metrics = None
        if out["seed_preds"]:
            seed_metrics = [{"seed": s, **regression_metrics(y, p)} for s, p in out["seed_preds"].items()]
            metrics = seed_summary(y, out["seed_preds"])
        else:
            metrics = self._score(task, y, out["pred"], out["proba"])
        self._write_predictions(self.artifact_dir(spec) / "test_predictions.csv", test, y,
                                out["pred"], out["proba"], seed_preds=out["seed_preds"])
        payload = {"experiment": spec.id, "task": spec.task, "split": spec.split,
                   "config_hash": entry["config_hash"], "evaluated_at": now_iso(),
                   "test_metrics": metrics, "seed_metrics": seed_metrics}
        write_json(self.test_path(spec.id), payload)
        logger.info("test evaluation", extra={"experiment": spec.id, task.selection_metric: metrics[task.selection_metric]})
        return payload

    def _final_derived(self, final_model: str, *, allow_repeat: bool) -> list[dict[str, Any]]:
        """E16: the selected T1 model's test predictions, binned (T2) / thresholded (T3)."""
        table = self.load_predictions(final_model, "test")
        points = table["y_true"].to_numpy()
        outputs = []
        for task_name in ("T2", "T3"):
            task = self.plan.tasks[task_name]
            scheme = get_scheme(task.target)
            exp_id = f"E16_{final_model}_as_{task_name}"
            cfg_hash = config_hash({"derived_from": final_model, "task": task_name})
            guard_test_access(self.results, exp_id, cfg_hash, allow_repeat=allow_repeat)
            log_test_access(self.results, exp_id, cfg_hash, "grouped_v1", repeat=allow_repeat,
                            note=f"derived from {final_model} test predictions; no new model")
            y = assign_labels(pd.Series(points), scheme).to_numpy()
            pred = scores_to_classes(table["y_pred"].to_numpy(), scheme)
            metrics = classification_metrics(y, pred, scheme)
            payload = {"experiment": exp_id, "task": task_name, "split": "grouped_v1",
                       "config_hash": cfg_hash, "derived_from": final_model,
                       "evaluated_at": now_iso(), "test_metrics": metrics}
            write_json(self.test_path(exp_id), payload)
            outputs.append(payload)
        return outputs
