"""Drift held-out identifiers toward a disjoint lexicon_drift pool."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.data.synth.rename import (
    REASON_NO_CANDIDATES,
    collect_renameable_names,
    rename_source,
)

DRIFT_PROPORTIONS = (0, 10, 25, 50)
REASON_PROPORTION_ZERO = "proportion_0"


@dataclass(frozen=True)
class DriftResult:
    source: str
    rename_map: dict[str, str]
    proportion: int
    renamed: bool = False
    reason: str = field(default=REASON_NO_CANDIDATES)


def apply_drift(
    source: str,
    lexicon: EnterpriseLexicon,
    proportion: int,
    *,
    seed: int,
) -> DriftResult:
    """Replace ``proportion``% of definitions (and uses) with lexicon_drift names.

    ``proportion`` must be one of ``{0, 10, 25, 50}``. At 0, the source is
    unchanged. Drift names are drawn only from ``lexicon_drift``. Files that
    cannot be parsed or fail the alpha-equivalence check are left unchanged.
    """
    if proportion not in DRIFT_PROPORTIONS:
        raise ValueError(
            f"proportion must be one of {DRIFT_PROPORTIONS}, got {proportion}"
        )
    if proportion == 0:
        return DriftResult(source, {}, proportion, False, REASON_PROPORTION_ZERO)
    olds = collect_renameable_names(source)
    if not olds:
        return DriftResult(source, {}, proportion)

    rng = np.random.default_rng(seed)
    n_replace = int(round(len(olds) * (proportion / 100.0)))
    n_replace = min(max(n_replace, 0), len(olds))
    if n_replace == 0:
        return DriftResult(source, {}, proportion)

    indices = list(range(len(olds)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n_replace])
    selected = [olds[i] for i in chosen]
    news = lexicon.allocate_names(len(selected), which="drift")
    mapping = dict(zip(selected, news, strict=True))

    # Assert drift names never appear in the adapt term inventory.
    adapt_terms = lexicon.lexicon_adapt.all_terms()
    for new in news:
        if new in adapt_terms:
            raise AssertionError(f"drift name {new!r} equals an adapt lexicon term")

    result = rename_source(source, mapping)
    return DriftResult(
        source=result.source,
        rename_map=result.rename_map,
        proportion=proportion,
        renamed=result.renamed,
        reason=result.reason,
    )
