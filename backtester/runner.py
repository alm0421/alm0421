"""Run any strategy spec (signal Strategy or allocation Portfolio) and load specs from JSON."""
from __future__ import annotations

import json
from pathlib import Path

from . import engine, portfolio
from .engine import Result
from .portfolio import Portfolio
from .strategy import Strategy

Spec = Strategy | Portfolio


def run(spec: Spec) -> Result:
    if isinstance(spec, Portfolio):
        return portfolio.run(spec)
    return engine.run(spec)


def from_dict(d: dict) -> Spec:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind == "allocation" or "tree" in d:
        return Portfolio.from_dict(d)
    return Strategy.from_dict(d)


def load(path: str | Path) -> Spec:
    return from_dict(json.loads(Path(path).read_text()))


def to_dict(spec: Spec) -> dict:
    return json.loads(spec.to_json())
