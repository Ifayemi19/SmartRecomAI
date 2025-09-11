# core/content_infer.py
from __future__ import annotations
import os, json, re, difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from data_loader import find_file  # pour backfill

@dataclass
class _Meta:
    n_items: int
    n_users: int
    iid2ix: Dict[int, int]
    ix2iid: Dict[int, int]

def _clean_token(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", "", s.lower())

class ContentEngine:
    """
    Deux modes de contenu :
      1) TF-IDF (tes artefacts déjà entraînés) → 'propre'
      2) 'Soup' CountVectorizer (title+authors+avg_rating) → esprit notebook, kNN cosine
         + fuzzy match par titre.
    """

    def __init__(self, artifacts_dir: str | os.PathLike = "artifacts"):
        self.art_dir = Path(artifacts_dir)
        if not self.art_dir.exists():
            raise FileNotFoundError(f"Artifacts dir '{self.art_dir}' introuvable")

        # --- meta + catalogue (avec backfill possible)
        self.meta = self._load_meta(self.art_dir / "model_meta.json")
        self.catalog = self._load_catalog(self.art_dir / "books_catalog.csv")
        self._rebuild_catalog_index()

        # --- TF-IDF (déjà dispo dans tes artefacts)
        self.vectorizer: TfidfVectorizer = joblib.load(self.art_dir / "content_tfidf.joblib")
        self.X_items_tfidf: csr_matrix = load_npz(self.art_dir / "item_content_tfidf.npz").tocsr()
        self.knn_tfidf: NearestNeighbors = joblib.load(self.art_dir / "content_knn.joblib")

        # --- optionnel: user_item pour centroïde utilisateur
        try:
            self.user_item: Optional[csr_matrix] = load_npz(self.art_dir / "user_item_csr.npz").tocsr()
        except Exception:
            self.user_item = None

        # --- 'Soup' on-demand (lazy)
        self.count_vec: Optional[CountVectorizer] = None
        self.X_items_count: Optional[csr_matrix] = None
        self.knn_count: Optional[NearestNeighbors] = None
        self.title_index: Optional[Dict[str, int]] = None  # token titre -> book_id

        self._MIN_OVERFETCH = 50
        self._OVERFETCH_FACTOR = 5
        self._SAVE_BACKFILL = True

    # ---------- meta & catalogue ----------
    def _load_meta(self, fp: Path) -> _Meta:
        d = json.loads(fp.read_text(encoding="utf-8"))
        iid2ix = {int(k): int(v) for k, v in d["iid2ix"].items()}
        ix2iid = {int(k): int(v) for k, v in d["ix2iid"].items()}
        return _Meta(n_items=int(d["n_items"]), n_users=int(d["n_users"]), iid2ix=iid2ix, ix2iid=ix2iid)

    def _load_catalog(self, fp: Path) -> pd.DataFrame:
        if not fp.exists():
            raise FileNotFoundError(f"Catalogue '{fp}' manquant")
        df = pd.read_csv(fp)
        if "book_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "book_id"})
        need = ["book_id", "title", "authors", "language_code", "average_rating", "ratings_count"]
        for c in need:
            if c not in df.columns:
                df[c] = np.nan if c in ("average_rating",) else ""
        df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["book_id"]).astype({"book_id": int})
        return df.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    def _rebuild_catalog_index(self) -> None:
        self._catalog_idx: Dict[int, pd.Series] = {int(r.book_id): r for _, r in self.catalog.iterrows()}

    # ---------- backfill depuis goodbooks (si titre/auteur manquent) ----------
    def _try_load_raw_books(self) -> Optional[pd.DataFrame]:
        p = find_file("books.csv")
        if not p: return None
        raw = pd.read_csv(p)
        if "book_id" not in raw.columns and "id" in raw.columns:
            raw = raw.rename(columns={"id": "book_id"})
        need = ["book_id", "original_title", "title", "authors", "language_code", "average_rating", "ratings_count"]
        for c in need:
            if c not in raw.columns:
                raw[c] = np.nan if c in ("average_rating",) else ""
        raw["book_id"] = pd.to_numeric(raw["book_id"], errors="coerce").astype("Int64")
        raw = raw.dropna(subset=["book_id"]).astype({"book_id": int})
        return raw.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    def _backfill_catalog(self, missing_ids: List[int]) -> None:
        if not missing_ids: return
        raw = self._try_load_raw_books()
        if raw is None or raw.empty: return
        add = raw.loc[raw["book_id"].isin(missing_ids), ["book_id","title","authors","language_code","average_rating","ratings_count"]]
        if add.empty: return
        self.catalog = pd.concat([self.catalog, add], ignore_index=True)
        self.catalog = self.catalog.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)
        self._rebuild_catalog_index()
        if self._SAVE_BACKFILL:
            try: self.catalog.to_csv(self.art_dir / "books_catalog.csv", index=False)
            except Exception: pass

    # ---------- titre → id (fuzzy) ----------
    def _ensure_title_index(self):
        if self.title_index is not None:
            return
        keys: Dict[str, int] = {}
        # utilise catalog + (si dispo) raw pour couvrir 'original_title'
        raw = self._try_load_raw_books()
        def add_title(bid: int, t: str):
            tok = _clean_token(t)
            if tok and tok not in keys:
                keys[tok] = int(bid)
        for df in [self.catalog, raw] if raw is not None else [self.catalog]:
            if df is None: continue
            for _, r in df.iterrows():
                bid = int(r["book_id"])
                t = r.get("original_title", None) if "original_title" in df.columns else None
                if t and isinstance(t, str) and t.strip():
                    add_title(bid, t)
                t2 = r.get("title", None)
                if t2 and isinstance(t2, str) and t2.strip():
                    add_title(bid, t2)
        self.title_index = keys

    def _find_book_id_by_title(self, title: str) -> Optional[int]:
        self._ensure_title_index()
        if not self.title_index:
            return None
        tok = _clean_token(title)
        if tok in self.title_index:
            return self.title_index[tok]
        # fuzzy via difflib
        choices = list(self.title_index.keys())
        match = difflib.get_close_matches(tok, choices, n=1, cutoff=0.6)
        if match:
            return self.title_index[match[0]]
        return None

    # ---------- 'soup' CountVectorizer (title+authors+avg_rating) ----------
    def _ensure_count_soup(self):
        if self.count_vec is not None:
            return
        # récupérer aussi original_title si disponible
        raw = self._try_load_raw_books()
        df = self.catalog.copy()
        if raw is not None and "original_title" in raw.columns:
            df = df.merge(raw[["book_id","original_title"]], on="book_id", how="left")
        # ordonner selon mapping iid pour alignement rangées
        df = df[df["book_id"].isin(self.meta.iid2ix)].copy()
        df["ix"] = df["book_id"].map(self.meta.iid2ix)
        df = df.sort_values("ix").reset_index(drop=True)

        # features
        title = df["original_title"].fillna(df["title"]).fillna("")
        authors = df["authors"].fillna("")
        avg = df["average_rating"].fillna("").astype(str)
        # clean comme ton notebook (lower + remove spaces)
        title = title.map(_clean_token)
        authors = authors.map(_clean_token)
        avg = avg.map(_clean_token)

        soup = (title + " " + authors + " " + avg).str.strip()

        self.count_vec = CountVectorizer(stop_words="english")
        self.X_items_count = self.count_vec.fit_transform(soup.tolist())
        self.knn_count = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn_count.fit(self.X_items_count)

    # ---------- helpers ----------
    def _ix_to_id(self, ix: int) -> int:
        return int(self.meta.ix2iid[int(ix)])

    def _enrich(self, ids_scores: List[Tuple[int, float]]) -> List[Dict]:
        # backfill si besoin
        missing = [int(b) for b, _ in ids_scores if int(b) not in self._catalog_idx]
        if missing: self._backfill_catalog(missing)
        out: List[Dict] = []
        for book_id, score in ids_scores:
            row = self._catalog_idx.get(int(book_id))
            if row is None:
                out.append({"book_id": int(book_id), "title": "(titre indisponible)", "authors": "",
                            "language_code": "", "average_rating": None, "ratings_count": 0,
                            "score": float(score)})
            else:
                out.append({"book_id": int(book_id), "title": row.title, "authors": row.authors,
                            "language_code": row.language_code,
                            "average_rating": float(row.average_rating) if pd.notna(row.average_rating) else None,
                            "ratings_count": int(row.ratings_count) if pd.notna(row.ratings_count) else 0,
                            "score": float(score)})
        return out

    # ---------- API : content-based façon notebook (STRICT) ----------
    def recommend_by_title(
        self,
        title: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Top-k similaires à un titre (soup CountVectorizer), seed potentiellement inclus."""
        self._ensure_count_soup()
        bid = self._find_book_id_by_title(title)
        if bid is None or int(bid) not in self.meta.iid2ix:
            return []
        seed_ix = int(self.meta.iid2ix[int(bid)])
        want = min(top_k, self.X_items_count.shape[0])  # type: ignore
        dists, neighs = self.knn_count.kneighbors(self.X_items_count[seed_ix], n_neighbors=want, return_distance=True)  # type: ignore
        pairs = [(self._ix_to_id(int(ix)), 1.0 - float(dists[0, j])) for j, ix in enumerate(neighs[0])]
        return self._enrich(pairs)[:want]

    def recommend_by_text_count(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Recherche textuelle dans la soup (CountVectorizer) — équivalent à ton `count.fit_transform(...); cosine`."""
        if not (query or "").strip():
            return []
        self._ensure_count_soup()
        q = _clean_token(query)
        q_vec = self.count_vec.transform([q])  # type: ignore
        want = min(top_k, self.X_items_count.shape[0])  # type: ignore
        dists, neighs = self.knn_count.kneighbors(q_vec, n_neighbors=want, return_distance=True)  # type: ignore
        pairs = [(self._ix_to_id(int(ix)), 1.0 - float(dists[0, j])) for j, ix in enumerate(neighs[0])]
        return self._enrich(pairs)[:want]

    # ---------- API : chemin 'propre' TF-IDF (conservé) ----------
    def recommend_by_text(self, query: str, top_k: int = 10, *, strict_unfiltered: bool = True) -> List[Dict]:
        if not (query or "").strip(): return []
        q_vec = self.vectorizer.transform([query])
        want = min(top_k, self.X_items_tfidf.shape[0])
        dists, neighs = self.knn_tfidf.kneighbors(q_vec, n_neighbors=want, return_distance=True)
        pairs = [(self._ix_to_id(int(ix)), 1.0 - float(dists[0, j])) for j, ix in enumerate(neighs[0])]
        return self._enrich(pairs)[:want]

    def recommend_by_book(self, book_id: int, top_k: int = 10, *, strict_unfiltered: bool = True) -> List[Dict]:
        if int(book_id) not in self.meta.iid2ix: return []
        seed_ix = int(self.meta.iid2ix[int(book_id)])
        want = min(top_k, self.X_items_tfidf.shape[0])
        dists, neighs = self.knn_tfidf.kneighbors(self.X_items_tfidf[seed_ix], n_neighbors=want, return_distance=True)
        pairs = [(self._ix_to_id(int(ix)), 1.0 - float(dists[0, j])) for j, ix in enumerate(neighs[0])]
        return self._enrich(pairs)[:want]
