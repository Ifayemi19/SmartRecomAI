from __future__ import annotations
import os, pathlib
import pandas as pd
from typing import Dict, List, Optional

GOODBOOKS_FILES = ["books.csv", "ratings.csv", "book_tags.csv", "tags.csv", "to_read.csv"]

def _candidates() -> List[pathlib.Path]:
    home = pathlib.Path.home()
    bases = [
        pathlib.Path(os.environ.get("GOODBOOKS_DIR", "")) if os.environ.get("GOODBOOKS_DIR") else None,
        pathlib.Path("data/raw/goodbooks-10k"),
        pathlib.Path("data/raw"),
        pathlib.Path("data"),
        home / "Downloads",
        pathlib.Path("."),
    ]
    return [p for p in bases if p and p.exists()]

def find_file(name: str) -> Optional[pathlib.Path]:
    for base in _candidates():
        p1 = base / name
        p2 = base / "goodbooks-10k" / name
        if p1.exists(): return p1
        if p2.exists(): return p2
    return None

def load_goodbooks() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, Optional[pd.DataFrame]] = {}
    missing = []
    for f in GOODBOOKS_FILES:
        fp = find_file(f)
        if not fp:
            missing.append(f)
            dfs[f] = None  # type: ignore
        else:
            dfs[f] = pd.read_csv(fp)
    if missing:
        raise FileNotFoundError(
            "Introuvable: " + ", ".join(missing) +
            "\nPlace les CSV dans data/raw/ (ou export GOODBOOKS_DIR=/chemin/)"
        )
    return {
        "books": dfs["books.csv"],         # type: ignore
        "ratings": dfs["ratings.csv"],     # type: ignore
        "book_tags": dfs["book_tags.csv"], # type: ignore
        "tags": dfs["tags.csv"],           # type: ignore
        "to_read": dfs["to_read.csv"],     # type: ignore
    }
