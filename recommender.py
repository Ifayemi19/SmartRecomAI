# recommender.py (amélioré)
from __future__ import annotations

import re
from typing import Dict, Tuple, List, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# =========  PARTIE COLLABORATIVE (item-item KNN)  ===========
# ============================================================

def build_mappings(ratings: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Construit les mappings ids métier -> indices pour users et items.
    """
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["book_id"].unique())
    uid2ix = {int(uid): int(i) for i, uid in enumerate(user_ids)}
    iid2ix = {int(iid): int(i) for i, iid in enumerate(item_ids)}
    return uid2ix, iid2ix


def build_csr(
    ratings: pd.DataFrame,
    uid2ix: Dict[int, int],
    iid2ix: Dict[int, int],
    min_rating: int = 3,
    *,
    weighted: bool = False,            # NEW: pondère par la note (comme dans ton notebook)
    l2_normalize_rows: bool = False,   # NEW: normalisation L2 des profils utilisateurs
) -> csr_matrix:
    """
    Convertit les ratings en matrice user×item.
    - Par défaut: binaire (implicit feedback) pour notes >= min_rating.
    - Si weighted=True: poids = rating (>=min_rating).
    - Option l2_normalize_rows: normalise chaque ligne (utile si weighted).
    """
    r = ratings.loc[ratings["rating"] >= min_rating, ["user_id", "book_id", "rating"]].copy()
    rows = r["user_id"].map(uid2ix).to_numpy()
    cols = r["book_id"].map(iid2ix).to_numpy()
    data = (r["rating"].astype("float32").to_numpy() if weighted else np.ones(len(rows), dtype=np.float32))
    n_users, n_items = len(uid2ix), len(iid2ix)
    X = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    if l2_normalize_rows:
        # normalise chaque profil utilisateur (évite qu'un gros rateur domine)
        norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        X = X.multiply(1.0 / norms[:, None])
    return X


def fit_item_knn(user_item: csr_matrix, n_neighbors: int = 50, metric: str = "cosine") -> NearestNeighbors:
    """
    Entraîne un k-NN item-item (cosinus par défaut) sur la matrice item×user.
    """
    item_user = user_item.T
    knn = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(item_user)
    return knn


def aggregate_user_recs(
    user_ix: int,
    user_item: csr_matrix,
    knn: NearestNeighbors,
    k_top: int = 10,
    exclude_seen: bool = True,
    *,
    topn_from_each_liked: int = 50,
    agg: str = "sum",                  # "sum" | "max"
    normalize_by_likes: bool = False,  # NEW: moyenne au lieu de somme
) -> List[tuple[int, float]]:
    """
    Recommandations collaboratives pour un utilisateur (indices item),
    en agrégeant les similarités des voisins des items déjà likés.
    """
    liked_items = user_item[user_ix].indices
    if liked_items.size == 0:
        return []

    scores: Dict[int, float] = {}
    seen = set(liked_items.tolist()) if exclude_seen else set()

    dists, neighs = knn.kneighbors(user_item.T[liked_items], return_distance=True, n_neighbors=topn_from_each_liked)
    for row in range(neighs.shape[0]):
        for j, n_item in enumerate(neighs[row]):
            n_item = int(n_item)
            if n_item in seen:
                continue
            sim = 1.0 - float(dists[row, j])  # cos sim
            if agg == "max":
                scores[n_item] = max(scores.get(n_item, 0.0), sim)
            else:
                scores[n_item] = scores.get(n_item, 0.0) + sim

    # Option: moyenne par nb de likes (évite de favoriser les gros lecteurs)
    if normalize_by_likes and liked_items.size > 0:
        for k in list(scores.keys()):
            scores[k] /= float(liked_items.size)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_top]
    return ranked


# ============================================================
# ==========  PARTIE CONTENU (TF-IDF + k-NN cosinus)  ========
# ============================================================

def _clean_words(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.lower()
    s = re.sub(r"[^\w\s\.\']+", " ", s)  # ponctuation -> espace (garde . ' pour initiales)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _authors_clean_words(s: str) -> str:
    s = str(s) if s is not None else ""
    toks = []
    for tok in re.split(r"[|,]", s):
        t = _clean_words(tok)
        if t:
            toks.append(t)
    # dédoublonner en gardant l’ordre
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


def _rating_bucket_token(r) -> str:
    try:
        v = float(r)
    except Exception:
        return ""
    if v >= 4.5:
        return "rating_top"
    if v >= 4.0:
        return "rating_high"
    if v >= 3.5:
        return "rating_mid"
    return "rating_low"


def build_item_content_matrix_tfidf(
    items: pd.DataFrame,
    iid2ix: Dict[int, int],
    *,
    title_col: str = "title",           # supporte aussi "original_title"
    authors_col: str = "authors",
    rating_col: str = "average_rating",
    extra_text_cols: Iterable[str] = (),   # ex: ("genres","description")
    title_weight: float = 2.0,             # pondération naïve par répétition
    authors_weight: float = 1.0,
    extra_weight: float = 1.0,
    max_features: int = 50000,
    min_df: int | float = 2,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: Optional[str | List[str]] = "english",
    sublinear_tf: bool = True,
    return_soup: bool = False,             # NEW: retourne la soupe pour debug
) -> Tuple[csr_matrix, TfidfVectorizer, pd.DataFrame] | Tuple[csr_matrix, TfidfVectorizer, pd.DataFrame, pd.Series]:
    """
    Construit une matrice TF-IDF item×termes à partir d’une “soupe” de texte :
      title (×title_weight) + authors (×authors_weight) + rating_bucket + extra_text_cols (×extra_weight).
    Les lignes sont réordonnées pour correspondre à iid2ix.
    """
    df = items[items["book_id"].isin(iid2ix)].copy()
    df["ix"] = df["book_id"].map(iid2ix)
    df = df.sort_values("ix").reset_index(drop=True)

    # fallback titre
    if title_col not in df.columns or df[title_col].isna().all():
        if "original_title" in df.columns:
            title_col = "original_title"
        else:
            df["title"] = ""
            title_col = "title"

    title = df.get(title_col, "").fillna("").astype(str).map(_clean_words)
    authors = df.get(authors_col, "").fillna("").astype(str).map(_authors_clean_words)
    rating_tok = df.get(rating_col, np.nan).apply(_rating_bucket_token)

    # Extra colonnes (genres, description…)
    extra = pd.Series([""] * len(df))
    for c in extra_text_cols:
        if c in df.columns:
            extra = (extra + " " + df[c].fillna("").astype(str).map(_clean_words)).str.strip()

    # Pondération naïve par répétition (répète le texte w fois)
    def _repeat(s: pd.Series, w: float) -> pd.Series:
        if w <= 1.0:
            return s
        rep = max(1, int(round(w)))
        return s.apply(lambda t: " ".join([t] * rep) if t else "")

    title = _repeat(title, title_weight)
    authors = _repeat(authors, authors_weight)
    extra = _repeat(extra, extra_weight)

    soup = (title + " " + authors + " " + rating_tok + " " + extra).str.strip()

    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=sublinear_tf,
        norm="l2",  # parfait pour cosinus
    )
    X_items = tfidf.fit_transform(soup.tolist())
    if return_soup:
        return X_items, tfidf, df, soup
    return X_items, tfidf, df


def fit_item_content_knn(X_items: csr_matrix, n_neighbors: int = 50, metric: str = "cosine") -> NearestNeighbors:
    """
    Entraîne un k-NN sur les embeddings TF-IDF des items.
    """
    knn = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(X_items)
    return knn


def recommend_similar_items_content(
    seed_item_ix: int,
    X_items: csr_matrix,
    knn: NearestNeighbors,
    k_top: int = 10,
    include_seed: bool = False,
    *,
    overfetch_factor: int = 5,   # NEW: évite les listes vides quand k_top petit
    min_overfetch: int = 50,     # NEW
) -> List[tuple[int, float]]:
    """
    Renvoie les items les plus proches d’un item (indices), score = cos_sim.
    Overfetch + slice pour éviter le cas où l’item seed est exclu et vide la liste.
    """
    fetch_k = max(min_overfetch, k_top * overfetch_factor)
    dists, neighs = knn.kneighbors(X_items[seed_item_ix], return_distance=True, n_neighbors=fetch_k)
    out: List[tuple[int, float]] = []
    for j, n_item in enumerate(neighs[0]):
        n_item = int(n_item)
        if not include_seed and n_item == seed_item_ix:
            continue
        sim = 1.0 - float(dists[0, j])
        out.append((n_item, sim))
    return out[:k_top]


def aggregate_user_recs_content(
    user_ix: int,
    user_item: csr_matrix,
    X_items: csr_matrix,
    knn: NearestNeighbors,
    k_top: int = 10,
    exclude_seen: bool = True,
    *,
    topn_from_each_liked: int = 50,
    agg: str = "sum",              # "sum" | "max"
    overfetch_factor: int = 3,     # NEW: robustesse
    min_overfetch: int = 50,       # NEW
    normalize_by_likes: bool = False,  # NEW
) -> List[tuple[int, float]]:
    """
    Recos 'par contenu' pour un utilisateur (indices item) :
      - on récupère les voisins (contenu) des items likés
      - on agrège leurs similarités (sum ou max)
    """
    liked_items = user_item[user_ix].indices
    if liked_items.size == 0:
        return []

    scores: Dict[int, float] = {}
    seen = set(liked_items.tolist()) if exclude_seen else set()

    fetch_k = max(min_overfetch, topn_from_each_liked * overfetch_factor)
    dists, neighs = knn.kneighbors(X_items[liked_items], return_distance=True, n_neighbors=fetch_k)
    for row in range(neighs.shape[0]):
        for j, n_item in enumerate(neighs[row]):
            n_item = int(n_item)
            if n_item in seen:
                continue
            sim = 1.0 - float(dists[row, j])
            if agg == "max":
                scores[n_item] = max(scores.get(n_item, 0.0), sim)
            else:
                scores[n_item] = scores.get(n_item, 0.0) + sim

    if normalize_by_likes and liked_items.size > 0:
        for k in list(scores.keys()):
            scores[k] /= float(liked_items.size)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_top]
    return ranked


# ============================================================
# =====  CONTENU : CENTROÏDE UTILISATEUR (nouvelle API)  =====
# ============================================================

def _l2_normalize(vec: csr_matrix) -> csr_matrix:
    if not issparse(vec):
        raise ValueError("Attendu vecteur sparse CSR.")
    norm = np.sqrt(vec.multiply(vec).sum())
    if norm == 0:
        return vec
    return vec.multiply(1.0 / float(norm))


def recommend_content_for_user_centroid(
    user_ix: int,
    user_item: csr_matrix,
    X_items: csr_matrix,
    knn: NearestNeighbors,
    k_top: int = 10,
    exclude_seen: bool = True,
    *,
    overfetch_factor: int = 5,
    min_overfetch: int = 80,
) -> List[tuple[int, float]]:
    """
    Recommandations contenu par 'centroïde utilisateur':
      - calcule un embedding = moyenne (L2-normalisée) des items likés
      - trouve les voisins plus proches de ce centroïde
    """
    liked = user_item[user_ix].indices
    if liked.size == 0:
        return []

    # centroïde (moyenne simple des TF-IDF des items likés)
    centroid = X_items[liked].mean(axis=0)
    centroid = csr_matrix(centroid)
    centroid = _l2_normalize(centroid)

    fetch_k = max(min_overfetch, k_top * overfetch_factor)
    dists, neighs = knn.kneighbors(centroid, return_distance=True, n_neighbors=fetch_k)

    seen = set(liked.tolist()) if exclude_seen else set()
    out: List[tuple[int, float]] = []
    for j, n_item in enumerate(neighs[0]):
        n_item = int(n_item)
        if n_item in seen:
            continue
        sim = 1.0 - float(dists[0, j])
        out.append((n_item, sim))
    return out[:k_top]


# ============================================================
# ===============  UTILITAIRES IDS <-> INDICES  =============
# ============================================================

def ix_to_ids(
    recs_ix_and_score: List[tuple[int, float]],
    ix2iid: Dict[int, int],
) -> List[tuple[int, float]]:
    """
    Convertit [(item_ix, score)] -> [(book_id, score)]
    """
    return [(int(ix2iid[ix]), float(score)) for ix, score in recs_ix_and_score]
