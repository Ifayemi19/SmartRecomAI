# core/collab_infer.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors

from data_loader import find_file  # pour backfill titres si besoin

@dataclass
class _Meta:
    n_items: int
    n_users: int
    uid2ix: Dict[int, int]
    iid2ix: Dict[int, int]
    ix2uid: Dict[int, int]
    ix2iid: Dict[int, int]

class CollabEngine:
    """
    Item-item collaborative filtering (cosine, brute).
    Ajoute les filtres:
      - popularity_thres: nb minimal d'utilisateurs ayant noté un livre
      - ratings_thres:    nb minimal de notes par utilisateur
    """

    def __init__(self, artifacts_dir: str | os.PathLike = "artifacts"):
        self.art_dir = Path(artifacts_dir)
        if not self.art_dir.exists():
            raise FileNotFoundError(f"Artifacts dir '{self.art_dir}' introuvable")
        self.user_item: csr_matrix = load_npz(self.art_dir / "user_item_csr.npz").tocsr()
        self.item_user: csr_matrix = self.user_item.T.tocsr()
        self.knn_default: NearestNeighbors = joblib.load(self.art_dir / "item_knn.joblib")
        self.meta = self._load_meta(self.art_dir / "model_meta.json")

        # catalogue + backfill
        self.catalog = self._load_catalog(self.art_dir / "books_catalog.csv")
        self._catalog_idx = {int(r.book_id): r for _, r in self.catalog.iterrows()}
        self._SAVE_BACKFILL = True

        # cache des KNN filtrés
        self._knn_cache: Dict[Tuple[int,int], Tuple[NearestNeighbors, np.ndarray, np.ndarray]] = {}

    # ---------- meta & catalog ----------
    def _load_meta(self, fp: Path) -> _Meta:
        d = json.loads(fp.read_text(encoding="utf-8"))
        return _Meta(
            n_items=int(d["n_items"]), n_users=int(d["n_users"]),
            uid2ix={int(k): int(v) for k,v in d["uid2ix"].items()},
            iid2ix={int(k): int(v) for k,v in d["iid2ix"].items()},
            ix2uid={int(k): int(v) for k,v in d["ix2uid"].items()},
            ix2iid={int(k): int(v) for k,v in d["ix2iid"].items()},
        )

    def _load_catalog(self, fp: Path) -> pd.DataFrame:
        if not fp.exists():
            raise FileNotFoundError(f"Catalogue '{fp}' manquant")
        df = pd.read_csv(fp)
        if "book_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id":"book_id"})
        need = ["book_id","title","authors","language_code","average_rating","ratings_count"]
        for c in need:
            if c not in df.columns:
                df[c] = np.nan if c in ("average_rating",) else ""
        df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["book_id"]).astype({"book_id": int})
        return df.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    # ---------- backfill titres si manquants ----------
    def _try_load_raw_books(self) -> Optional[pd.DataFrame]:
        p = find_file("books.csv")
        if not p: return None
        raw = pd.read_csv(p)
        if "book_id" not in raw.columns and "id" in raw.columns:
            raw = raw.rename(columns={"id": "book_id"})
        need = ["book_id","title","authors","language_code","average_rating","ratings_count"]
        for c in need:
            if c not in raw.columns:
                raw[c] = np.nan if c in ("average_rating",) else ""
        raw["book_id"] = pd.to_numeric(raw["book_id"], errors="coerce").astype("Int64")
        raw = raw.dropna(subset=["book_id"]).astype({"book_id": int})
        return raw[need].drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    def _backfill_catalog(self, missing_ids: List[int]) -> None:
        if not missing_ids: return
        raw = self._try_load_raw_books()
        if raw is None or raw.empty: return
        add = raw.loc[raw["book_id"].isin(missing_ids)]
        if add.empty: return
        self.catalog = pd.concat([self.catalog, add], ignore_index=True)
        self.catalog = self.catalog.drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)
        self._catalog_idx = {int(r.book_id): r for _, r in self.catalog.iterrows()}
        if self._SAVE_BACKFILL:
            try: self.catalog.to_csv(self.art_dir / "books_catalog.csv", index=False)
            except Exception: pass

    # ---------- utils ----------
    def _ix_to_id(self, ix: int) -> int:
        return int(self.meta.ix2iid[int(ix)])

    def _ensure_knn_filtered(self, popularity_thres: int, ratings_thres: int
                             ) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
        key = (popularity_thres, ratings_thres)
        if key in self._knn_cache:
            return self._knn_cache[key]

        # masques
        item_pop = np.diff(self.item_user.indptr)  # nnz par item
        mask_items = item_pop >= popularity_thres
        user_pop = np.diff(self.user_item.indptr)  # nnz par user
        mask_users = user_pop >= ratings_thres

        # filtre user×item puis transpose
        ui = self.user_item[mask_users][:, mask_items].tocsr()
        iu = ui.T.tocsr()

        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(iu)
        # mapping: anciens indices -> nouveaux (pour sélectionner les items aimés)
        kept_item_ix = np.where(mask_items)[0]
        old2new = -np.ones(self.item_user.shape[0], dtype=int)
        old2new[kept_item_ix] = np.arange(kept_item_ix.size)

        self._knn_cache[key] = (knn, old2new, kept_item_ix)
        return self._knn_cache[key]

    # ---------- public ----------
    def recommend_for_user(
        self,
        user_id: int,
        k_top: int = 10,
        *,
        exclude_seen: bool = True,
        topn_from_each_liked: int = 50,
        agg: str = "sum",               # "sum" | "max"
        normalize_by_likes: bool = False,
        popularity_thres: Optional[int] = None,
        ratings_thres: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Recommandations collaboratives item-item.
        Si popularity_thres/ratings_thres sont fournis → applique les filtres du notebook.
        """
        if int(user_id) not in self.meta.uid2ix:
            return []
        user_ix = int(self.meta.uid2ix[int(user_id)])
        liked_items = self.user_item[user_ix].indices
        if liked_items.size == 0:
            return []

        if popularity_thres and ratings_thres:
            knn, old2new, kept_ix = self._ensure_knn_filtered(popularity_thres, ratings_thres)
            liked_new = [old2new[i] for i in liked_items if old2new[i] >= 0]
            if not liked_new:
                return []
            # matrice d'entraînement correspondante
            _, _, kept_items_order = self._knn_cache[(popularity_thres, ratings_thres)]
            # on utilise directement la matrice knn.fit(iu), donc pour interroger on prend iu[liked_new]
            # Pour ça, on a besoin de la matrice iu filtrée, mais on ne l'a pas gardée:
            # astuce: on reconstitue depuis knn._fit_X (sklearn stocke les features)
            iu_filt: csr_matrix = knn._fit_X  # type: ignore
            dists, neighs = knn.kneighbors(iu_filt[liked_new], n_neighbors=max(50, topn_from_each_liked), return_distance=True)
            seen = set(liked_new) if exclude_seen else set()
            scores: Dict[int, float] = {}
            for row in range(neighs.shape[0]):
                for j, n_item_new in enumerate(neighs[row]):
                    n_item_new = int(n_item_new)
                    if n_item_new in seen: continue
                    sim = 1.0 - float(dists[row, j])
                    if agg == "max":
                        scores[n_item_new] = max(scores.get(n_item_new, 0.0), sim)
                    else:
                        scores[n_item_new] = scores.get(n_item_new, 0.0) + sim
            if normalize_by_likes and liked_new:
                for k in list(scores.keys()):
                    scores[k] /= float(len(liked_new))
            # convertit new->old->book_id
            kept_items_arr = kept_ix  # old indices conservés, ordre d'entraînement
            out_pairs: List[Tuple[int, float]] = []
            for new_ix, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_top]:
                old_ix = kept_items_arr[int(new_ix)]
                out_pairs.append((self._ix_to_id(int(old_ix)), float(s)))
            return out_pairs

        # chemin par défaut (sans filtres)
        knn = self.knn_default
        dists, neighs = knn.kneighbors(self.item_user[liked_items], n_neighbors=max(50, topn_from_each_liked), return_distance=True)
        seen = set(liked_items.tolist()) if exclude_seen else set()
        scores: Dict[int, float] = {}
        for row in range(neighs.shape[0]):
            for j, n_item_ix in enumerate(neighs[row]):
                n_item_ix = int(n_item_ix)
                if n_item_ix in seen: continue
                sim = 1.0 - float(dists[row, j])
                if agg == "max":
                    scores[n_item_ix] = max(scores.get(n_item_ix, 0.0), sim)
                else:
                    scores[n_item_ix] = scores.get(n_item_ix, 0.0) + sim
        if normalize_by_likes and liked_items.size > 0:
            for k in list(scores.keys()):
                scores[k] /= float(liked_items.size)
        ranked_ix = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_top]
        return [(self._ix_to_id(ix), float(score)) for ix, score in ranked_ix]

    def enrich(self, ids_scores: List[Tuple[int, float]]) -> List[Dict]:
        missing = [int(b) for b,_ in ids_scores if int(b) not in self._catalog_idx]
        if missing: self._backfill_catalog(missing)
        out: List[Dict] = []
        for book_id, score in ids_scores:
            row = self._catalog_idx.get(int(book_id))
            if row is None:
                out.append({"book_id": int(book_id), "title": "(titre indisponible)", "authors": "",
                            "language_code":"", "average_rating": None, "ratings_count": 0, "score": float(score)})
            else:
                out.append({"book_id": int(book_id), "title": row.title, "authors": row.authors,
                            "language_code": row.language_code,
                            "average_rating": float(row.average_rating) if pd.notna(row.average_rating) else None,
                            "ratings_count": int(row.ratings_count) if pd.notna(row.ratings_count) else 0,
                            "score": float(score)})
        return out
