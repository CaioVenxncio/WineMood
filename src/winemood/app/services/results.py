"""Read the committed Phase 3 research records for the application.

Every number shown by the Research and Explore pages comes from files under
``results/`` written by the experiment pipeline:

* ``report.json``       - test metrics with cluster-bootstrap CIs, hypotheses,
                          ablations, error analysis, aggregates, access log;
* ``test/*.json``       - per-experiment test metrics (seed details);
* ``validation/*.json`` - training durations and selected configurations;
* ``selection.json``    - the frozen model-selection record;
* ``eda/summary.json``  - dataset aggregates (no review texts).

The application never reads test predictions or test data: it only parses
these already-published aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Display metadata: names, families and protocol notes. No metrics live here.
MODEL_INFO: dict[str, tuple[str, str]] = {
    "E01_dummy_mean": ("Training-mean baseline", "Trivial baseline"),
    "E02_length": ("Review length only", "Trivial baseline"),
    "E03_metadata": ("Metadata only (taster, origin, variety, price)", "Reference (not text)"),
    "E04_tfidf_word": ("Word TF-IDF + Ridge", "Classical NLP · selected"),
    "E08_tfidf_mask_years": ("TF-IDF + Ridge, years masked", "Ablation (H4)"),
    "E10_lstm": ("BiLSTM", "Neural (5 seeds)"),
    "E11_lstm_keep_negations": ("BiLSTM, negations kept", "Ablation (H5)"),
    "E12_encoder": ("Frozen MiniLM encoder + Ridge", "Neural (frozen encoder)"),
    "E40_holdout_tfidf": ("TF-IDF + Ridge", "Unseen tasters"),
    "E41_holdout_lstm": ("BiLSTM", "Unseen tasters"),
    "E42_random_tfidf": ("TF-IDF + Ridge", "Size-matched random split"),
    "E43_random_lstm": ("BiLSTM", "Size-matched random split"),
}
MAIN_COMPARISON = (
    "E01_dummy_mean",
    "E02_length",
    "E03_metadata",
    "E12_encoder",
    "E04_tfidf_word",
    "E10_lstm",
)
SPLIT_LABELS = {
    "grouped_v1": "Main test split (grouped)",
    "taster_holdout_v1": "Taster-holdout test split (unseen critics)",
    "taster_random_v1": "Size-matched random test split",
}


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @classmethod
    def from_dict(cls, d: dict[str, float] | None) -> Interval | None:
        if not d:
            return None
        return cls(float(d["estimate"]), float(d["ci95_low"]), float(d["ci95_high"]))

    def fmt(self, digits: int = 3, signed: bool = False) -> str:
        f = f"{{:{'+' if signed else ''}.{digits}f}}"
        return f"{f.format(self.estimate)} [{f.format(self.low)}, {f.format(self.high)}]"


@dataclass(frozen=True)
class ModelResult:
    experiment: str
    name: str
    family: str
    split: str
    n: int
    mae: Interval
    rmse: Interval | None
    r2: Interval | None
    bias: Interval | None
    median_ae: float | None
    within_2: float | None
    n_seeds: int
    mae_seed_std: float | None
    ensemble_mae: float | None
    train_seconds: float | None

    @property
    def metric_note(self) -> str:
        if self.n_seeds > 1:
            return (
                f"Mean of {self.n_seeds} training seeds (seed SD {self.mae_seed_std:.3f}); "
                f"averaging the seeds' predictions gives MAE {self.ensemble_mae:.3f} (diagnostic)."
            )
        return "Single deterministic model."


@dataclass
class ResearchResults:
    available: bool
    missing: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    models: dict[str, ModelResult] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    eda: dict[str, Any] = field(default_factory=dict)

    # -- convenience views ------------------------------------------------------
    def main_comparison(self) -> list[ModelResult]:
        return [self.models[e] for e in MAIN_COMPARISON if e in self.models]

    @property
    def selected(self) -> ModelResult | None:
        return self.models.get(self.selection.get("final_model", ""))

    @property
    def hypotheses(self) -> dict[str, Any]:
        return self.report.get("hypotheses", {})

    @property
    def ablations(self) -> dict[str, Any]:
        return self.report.get("validation_ablations", {})

    @property
    def error_analysis(self) -> dict[str, Any]:
        return self.report.get("error_analysis", {})

    @property
    def aggregates(self) -> dict[str, Any]:
        return self.report.get("test_prediction_aggregates", {})

    @property
    def classification(self) -> list[dict[str, Any]]:
        return self.report.get("test_classification", [])

    @property
    def test_access_count(self) -> int:
        return len(self.report.get("test_access_log", []))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(results_dir: Path) -> ResearchResults:
    report_path = results_dir / "report.json"
    if not report_path.is_file():
        return ResearchResults(available=False, missing=[str(report_path)])
    report = _read(report_path)
    missing: list[str] = []
    models: dict[str, ModelResult] = {}
    for row in report.get("test_t1", []):
        exp_id = row["experiment"]
        test_path = results_dir / "test" / f"{exp_id}.json"
        test = _read(test_path)["test_metrics"] if test_path.is_file() else {}
        val_path = results_dir / "validation" / f"{exp_id}.json"
        seconds = _read(val_path)["meta"].get("seconds") if val_path.is_file() else None
        if not test_path.is_file():
            missing.append(str(test_path))
        name, family = MODEL_INFO.get(exp_id, (exp_id, ""))
        models[exp_id] = ModelResult(
            experiment=exp_id,
            name=name,
            family=family,
            split=row["split"],
            n=int(row["n"]),
            mae=Interval.from_dict(row["mae"]),
            rmse=Interval.from_dict(row.get("rmse")),
            r2=Interval.from_dict(row.get("r2")),
            bias=Interval.from_dict(row.get("bias")),
            median_ae=row.get("median_ae"),
            within_2=row.get("within_2"),
            n_seeds=int(row.get("n_seeds") or 1),
            mae_seed_std=row.get("mae_seed_std"),
            ensemble_mae=test.get("ensemble_mae"),
            train_seconds=seconds,
        )
    selection_path = results_dir / "selection.json"
    selection = _read(selection_path) if selection_path.is_file() else {}
    if not selection:
        missing.append(str(selection_path))
    eda_path = results_dir / "eda" / "summary.json"
    eda = _read(eda_path) if eda_path.is_file() else {}
    if not eda:
        missing.append(str(eda_path))
    return ResearchResults(True, missing, report, models, selection, eda)
