"""Experiment specifications loaded from ``configs/experiments.toml``.

An experiment has a task (T1/T2/T3), a split, a model family, a base
configuration and an optional hyperparameter grid. Grid keys are dotted paths
into the configuration (e.g. ``features.word.min_df`` or ``head.alpha``).
``base`` names another experiment whose *validation-selected* configuration is
the starting point, so controlled variants change exactly the declared keys.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from winemood.data.labels import RegressionTarget, get_target

MODEL_FAMILIES = ("linear", "lstm", "encoder")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target: str
    kind: str  # "regression" | "classification"
    selection_metric: str
    higher_is_better: bool

    @property
    def scheme(self):
        return get_target(self.target)

    @property
    def is_regression(self) -> bool:
        return isinstance(self.scheme, RegressionTarget)


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    task: str
    family: str
    split: str
    description: str
    config: dict[str, Any]
    grid: dict[str, list[Any]] = field(default_factory=dict)
    base: str | None = None
    seeds: tuple[int, ...] = (42,)
    hypothesis: str | None = None

    def grid_configs(self, base_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        start = merge(base_config or {}, self.config)
        if not self.grid:
            return [start]
        keys = sorted(self.grid)
        out = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            cfg = copy.deepcopy(start)
            for key, value in zip(keys, values, strict=True):
                set_path(cfg, key, value)
            out.append(cfg)
        return out


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge. A sub-dict that sets ``kind`` replaces the base sub-dict
    wholesale, so switching e.g. a ridge head to a logistic head cannot carry
    over stale parameters."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(out.get(key), dict)
            and ("kind" not in value or value.get("kind") == out[key].get("kind"))
        ):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ExperimentPlan:
    tasks: dict[str, TaskSpec]
    experiments: dict[str, ExperimentSpec]
    test_scope: tuple[str, ...]

    def get(self, exp_id: str) -> ExperimentSpec:
        try:
            return self.experiments[exp_id]
        except KeyError:
            raise KeyError(f"Unknown experiment '{exp_id}'") from None


def load_plan(path: Path) -> ExperimentPlan:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    tasks = {
        name: TaskSpec(
            name=name,
            target=t["target"],
            kind=t["kind"],
            selection_metric=t["selection_metric"],
            higher_is_better=bool(t.get("higher_is_better", False)),
        )
        for name, t in raw["tasks"].items()
    }
    experiments: dict[str, ExperimentSpec] = {}
    for e in raw["experiment"]:
        if e["id"] in experiments:
            raise ValueError(f"Duplicate experiment id '{e['id']}'")
        if e["task"] not in tasks:
            raise ValueError(f"Experiment '{e['id']}' uses unknown task '{e['task']}'")
        family = e.get("family", "linear")
        if family not in MODEL_FAMILIES:
            raise ValueError(f"Experiment '{e['id']}' has unknown family '{family}'")
        experiments[e["id"]] = ExperimentSpec(
            id=e["id"],
            task=e["task"],
            family=family,
            split=e.get("split", raw.get("defaults", {}).get("split", "grouped_v1")),
            description=e.get("description", ""),
            config=e.get("config", {}),
            grid=e.get("grid", {}),
            base=e.get("base"),
            seeds=tuple(e.get("seeds", [42])),
            hypothesis=e.get("hypothesis"),
        )
    for spec in experiments.values():
        if spec.base and spec.base not in experiments:
            raise ValueError(f"Experiment '{spec.id}' has unknown base '{spec.base}'")
    scope = tuple(raw.get("test_scope", {}).get("experiments", []))
    unknown = [s for s in scope if s not in experiments]
    if unknown:
        raise ValueError(f"test_scope lists unknown experiments: {unknown}")
    return ExperimentPlan(tasks, experiments, scope)
