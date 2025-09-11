# core/hybrid.py
from __future__ import annotations
from typing import Dict, List, Tuple


def _min_max_norm(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {int(i): 1.0 for i, _ in pairs}  # tous égaux → 1.0
    return {int(i): (float(s) - lo) / (hi - lo) for i, s in pairs}


def fuse_scores(
    *,
    collab: List[Tuple[int, float]] | None,
    content: List[Tuple[int, float]] | None,
    alpha: float = 0.6,   # poids du collaboratif
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Fusion linéaire (min-max) des scores collaboratif & contenu.
    alpha=1.0 => 100% collab ; alpha=0.0 => 100% contenu.
    """
    collab = collab or []
    content = content or []

    c_norm = _min_max_norm(collab)
    t_norm = _min_max_norm(content)

    keys = set(c_norm.keys()) | set(t_norm.keys())
    fused: List[Tuple[int, float]] = []
    for k in keys:
        s = alpha * c_norm.get(k, 0.0) + (1.0 - alpha) * t_norm.get(k, 0.0)
        fused.append((int(k), float(s)))

    fused.sort(key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]
