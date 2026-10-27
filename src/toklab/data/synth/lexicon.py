"""Seeded enterprise identifier lexicon with disjoint adapt/drift pools.

Identifiers follow the template ``<Org><Domain><Entity><Role>``, e.g.
``AcmeLogisticsShipmentRouter``. Word lists come from
``configs/lexicon.yaml``. Sampling uses Zipf-like weights so a few names
recur often while many remain rare.

``lexicon_adapt`` and ``lexicon_drift`` share **no** terms (asserted).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_YAML = (
    Path(__file__).resolve().parents[4] / "configs" / "lexicon.yaml"
)


def _default_yaml() -> Path:
    candidates = [
        Path.cwd() / "configs" / "lexicon.yaml",
        _DEFAULT_YAML,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _zipf_weights(n: int, s: float = 1.2) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()


def _load_yaml(path: Path) -> dict[str, list[str]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    required = ("orgs", "domains", "entities", "roles")
    out: dict[str, list[str]] = {}
    for key in required:
        if key not in data or not isinstance(data[key], list) or not data[key]:
            raise ValueError(f"{path} must define non-empty list {key!r}")
        words = [str(w) for w in data[key]]
        if len(words) != len(set(words)):
            raise ValueError(f"{path}: duplicate entries in {key!r}")
        out[key] = words
    return out


def _partition(
    words: list[str],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Shuffle then split into two non-empty halves (adapt, drift)."""
    if len(words) < 2:
        raise ValueError("need at least 2 words to form disjoint adapt/drift pools")
    order = list(words)
    rng.shuffle(order)
    mid = len(order) // 2
    adapt, drift = order[:mid], order[mid:]
    if not adapt or not drift:
        raise ValueError("adapt/drift partition produced an empty pool")
    return adapt, drift


@dataclass(frozen=True)
class LexiconPool:
    """One side (adapt or drift) of the enterprise word lists."""

    name: str
    orgs: tuple[str, ...]
    domains: tuple[str, ...]
    entities: tuple[str, ...]
    roles: tuple[str, ...]

    def all_terms(self) -> frozenset[str]:
        return frozenset(
            {*self.orgs, *self.domains, *self.entities, *self.roles}
        )

    def compose(self, org: str, domain: str, entity: str, role: str) -> str:
        return f"{org}{domain}{entity}{role}"


class EnterpriseLexicon:
    """Zipf-sampled ``<Org><Domain><Entity><Role>`` name generator."""

    def __init__(
        self,
        adapt: LexiconPool,
        drift: LexiconPool,
        *,
        seed: int,
        zipf_s: float = 1.2,
    ) -> None:
        overlap = adapt.all_terms() & drift.all_terms()
        if overlap:
            raise AssertionError(
                f"lexicon_adapt and lexicon_drift share terms: {sorted(overlap)}"
            )
        self.lexicon_adapt = adapt
        self.lexicon_drift = drift
        self.seed = seed
        self.zipf_s = zipf_s
        self._rng = np.random.default_rng(seed)
        self._counters = {"adapt": 0, "drift": 0}

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        seed: int = 0,
        zipf_s: float = 1.2,
    ) -> EnterpriseLexicon:
        raw = _load_yaml(Path(path) if path is not None else _default_yaml())
        rng = np.random.default_rng(seed)
        parts: dict[str, tuple[list[str], list[str]]] = {}
        for key in ("orgs", "domains", "entities", "roles"):
            parts[key] = _partition(raw[key], rng)
        adapt = LexiconPool(
            name="adapt",
            orgs=tuple(parts["orgs"][0]),
            domains=tuple(parts["domains"][0]),
            entities=tuple(parts["entities"][0]),
            roles=tuple(parts["roles"][0]),
        )
        drift = LexiconPool(
            name="drift",
            orgs=tuple(parts["orgs"][1]),
            domains=tuple(parts["domains"][1]),
            entities=tuple(parts["entities"][1]),
            roles=tuple(parts["roles"][1]),
        )
        return cls(adapt, drift, seed=seed, zipf_s=zipf_s)

    def _sample_from_pool(self, pool: LexiconPool) -> str:
        def pick(options: tuple[str, ...]) -> str:
            w = _zipf_weights(len(options), self.zipf_s)
            idx = int(self._rng.choice(len(options), p=w))
            return options[idx]

        return pool.compose(
            pick(pool.orgs),
            pick(pool.domains),
            pick(pool.entities),
            pick(pool.roles),
        )

    def next_name(self, which: str = "adapt") -> str:
        """Return the next Zipf-sampled identifier from adapt or drift."""
        if which not in ("adapt", "drift"):
            raise ValueError(f"which must be 'adapt' or 'drift', got {which!r}")
        pool = self.lexicon_adapt if which == "adapt" else self.lexicon_drift
        self._counters[which] += 1
        # Re-seeded substream per draw index keeps sequences stable if we
        # interleave adapt/drift draws in different orders within one process
        # only when using independent generators — here we use one RNG stream.
        return self._sample_from_pool(pool)

    def allocate_names(
        self,
        count: int,
        which: str = "adapt",
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Allocate ``count`` unique names from the given pool."""
        exclude = set(exclude or ())
        names: list[str] = []
        seen = set(exclude)
        # Cap attempts to avoid infinite loops on tiny pools.
        max_attempts = max(1000, count * 200)
        attempts = 0
        while len(names) < count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"could not allocate {count} unique {which} names "
                    f"(got {len(names)}); expand configs/lexicon.yaml"
                )
            candidate = self.next_name(which)
            if candidate in seen:
                # Collision: append a stable numeric suffix.
                suffix = 2
                base = candidate
                while f"{base}V{suffix}" in seen:
                    suffix += 1
                candidate = f"{base}V{suffix}"
            seen.add(candidate)
            names.append(candidate)
        return names

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "zipf_s": self.zipf_s,
            "adapt_terms": sorted(self.lexicon_adapt.all_terms()),
            "drift_terms": sorted(self.lexicon_drift.all_terms()),
        }
