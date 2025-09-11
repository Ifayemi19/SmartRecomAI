# train_model.py
from __future__ import annotations
import argparse
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.exceptions import NotFittedError

# --- tes modules locaux ---
from data_loader import load_goodbooks
from recommender import (
    build_mappings,
    build_csr,
    fit_item_knn,
    build_item_content_matrix_tfidf,
    fit_item_content_knn,
)

# Dossier des artefacts (à côté de ce script)
ART_DIR = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Stopwords FR (liste courte, améliorable au besoin)
FRENCH_SW = [
    "les","des","une","un","et","de","du","la","le","au","aux","en","pour","par","avec","sur",
    "dans","ce","cet","cette","ces","plus","moins","ou","où","est","sont","onto","auxquels","auxquelles",
    "dont","quand","quoi","qui","que","ne","pas","ni","mais","car","donc","or","si","comme","vers"
]

def _pick_title_col(books: pd.DataFrame) -> str:
    """Prend 'title' si non vide, sinon 'original_title'."""
    if "title" in books.columns and books["title"].notna().sum() > 0:
        return "title"
    if "original_title" in books.columns and books["original_title"].notna().sum() > 0:
        return "original_title"
    # Dernier recours: invente une colonne vide pour éviter crash
    books["title"] = ""
    return "title"

def _ensure_catalog_columns(books: pd.DataFrame) -> pd.DataFrame:
    """Prépare un catalogue propre pour l'enrichissement et l'API."""
    out = books.copy()
    # Colonnes attendues
    if "title" not in out.columns and "original_title" in out.columns:
        out["title"] = out["original_title"]
    if "authors" not in out.columns:
        out["authors"] = ""
    if "language_code" not in out.columns:
        out["language_code"] = "EN"
    if "average_rating" in out.columns:
        out["average_rating"] = pd.to_numeric(out["average_rating"], errors="coerce")
    else:
        out["average_rating"] = np.nan
    if "ratings_count" in out.columns:
        out["ratings_count"] = pd.to_numeric(out["ratings_count"], errors="coerce").fillna(0).astype(int)
    else:
        out["ratings_count"] = 0
    keep = ["book_id","title","authors","language_code","average_rating","ratings_count"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()

def _parse_tuple(s: str) -> Tuple[int, int]:
    """Parse '1,2' -> (1,2)"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("format attendu: '1,2'")
    a, b = int(parts[0]), int(parts[1])
    if a > b:
        raise argparse.ArgumentTypeError("ngram_range invalide: a <= b requis")
    return (a, b)

def _parse_list(s: str) -> Tuple[str, ...]:
    """Parse 'a,b,c' -> ('a','b','c')"""
    if not s.strip():
        return tuple()
    return tuple([p.strip() for p in s.split(",") if p.strip()])

def _stopwords_choice(opt: str | None):
    """Mappe un choix en stop_words sklearn ou liste custom."""
    if not opt or opt.lower() in ("none", "null"):
        return None
    if opt.lower() in ("en", "english"):
        return "english"
    if opt.lower() in ("fr", "french"):
        return FRENCH_SW
    # défaut raisonnable
    return "english"

def train_and_serialize(
    n_neighbors: int = 50,
    min_rating: int = 3,
    *,
    content_min_df: int | float = 2,
    content_ngram: Tuple[int, int] = (1, 2),
    content_max_features: int = 50000,
    content_title_weight: float = 2.0,
    content_authors_weight: float = 1.0,
    content_extra_cols: Tuple[str, ...] = tuple(),  # ex: ("description","genres")
    content_stopwords: Optional[str] = "english",
) -> Dict[str, Any]:
    """
    Entraîne et sérialise:
      - Collaboratif (item-item KNN) sur matrice user×item (binaire, rating>=min_rating)
      - Contenu (TF-IDF + KNN) sur 'title/authors/(rating bucket)/extras'
    """
    print("📥 Chargement GoodBooks…")
    data = load_goodbooks()
    books, ratings = data["books"], data["ratings"]

    # --------- Collaboratif (item-item) ---------
    print("🧮 Construction mappings et matrice user×item…")
    uid2ix, iid2ix = build_mappings(ratings)
    csr = build_csr(ratings, uid2ix, iid2ix, min_rating=min_rating)

    print(f"🔧 Entraînement k-NN (collab) avec n_neighbors={n_neighbors}…")
    knn_collab = fit_item_knn(csr, n_neighbors=n_neighbors)

    print("💾 Sauvegarde artefacts collaboratif…")
    joblib.dump(knn_collab, ART_DIR / "item_knn.joblib")
    save_npz(ART_DIR / "user_item_csr.npz", csr)

    # --------- Catalogue commun ---------
    print("🗂️ Préparation catalogue…")
    books_catalog = _ensure_catalog_columns(books)
    books_catalog.to_csv(ART_DIR / "books_catalog.csv", index=False)

    # --------- Contenu (TF-IDF + KNN) ---------
    print("✍️  Construction matrice TF-IDF items (contenu)…")
    title_col = _pick_title_col(books)
    stop_words = _stopwords_choice(content_stopwords)

    try:
        X_items, tfidf, items_sorted = build_item_content_matrix_tfidf(
            books, iid2ix,
            title_col=title_col,
            authors_col="authors",
            rating_col="average_rating",
            extra_text_cols=content_extra_cols,
            title_weight=content_title_weight,
            authors_weight=content_authors_weight,
            max_features=content_max_features,
            min_df=content_min_df,
            ngram_range=content_ngram,
            stop_words=stop_words,
            sublinear_tf=True
        )
    except ValueError as e:
        # Cas classique: "After pruning, no terms remain." -> on abaisse min_df
        if "no terms remain" in str(e).lower():
            print("⚠️  Aucun terme après pruning. Refit avec min_df=1…")
            X_items, tfidf, items_sorted = build_item_content_matrix_tfidf(
                books, iid2ix,
                title_col=title_col,
                authors_col="authors",
                rating_col="average_rating",
                extra_text_cols=content_extra_cols,
                title_weight=content_title_weight,
                authors_weight=content_authors_weight,
                max_features=content_max_features,
                min_df=1,
                ngram_range=content_ngram,
                stop_words=stop_words,
                sublinear_tf=True
            )
        else:
            raise

    print(f"🔧 Entraînement k-NN (contenu) avec n_neighbors={n_neighbors}…")
    knn_content = fit_item_content_knn(X_items, n_neighbors=n_neighbors)

    print("💾 Sauvegarde artefacts contenu…")
    save_npz(ART_DIR / "item_content_tfidf.npz", X_items)
    joblib.dump(tfidf, ART_DIR / "content_tfidf.joblib")
    joblib.dump(knn_content, ART_DIR / "content_knn.joblib")

    # --------- Méta ---------
    print("📝 Écriture des métadonnées…")
    meta = {
        "n_neighbors": n_neighbors,
        "min_rating": min_rating,
        "uid2ix": {int(k): int(v) for k, v in uid2ix.items()},
        "iid2ix": {int(k): int(v) for k, v in iid2ix.items()},
        "ix2uid": {int(v): int(k) for k, v in uid2ix.items()},
        "ix2iid": {int(v): int(k) for k, v in iid2ix.items()},
        "n_users": int(len(uid2ix)),
        "n_items": int(len(iid2ix)),
        "content": {
            "title_col": title_col,
            "authors_col": "authors",
            "rating_col": "average_rating",
            "extra_text_cols": list(content_extra_cols),
            "min_df": content_min_df,
            "ngram_range": list(content_ngram),
            "max_features": content_max_features,
            "title_weight": content_title_weight,
            "authors_weight": content_authors_weight,
            "stop_words": (
                "english" if stop_words == "english"
                else "none" if stop_words is None
                else "custom"
            ),
        },
    }
    (ART_DIR / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    print("✅ Entraîné & sérialisé →", ART_DIR.resolve())
    print(f"   • users: {meta['n_users']}  • items: {meta['n_items']}")
    return meta

def main():
    p = argparse.ArgumentParser(description="Entraîner et sérialiser les artefacts collab + contenu (GoodBooks-10k).")
    p.add_argument("--n-neighbors", type=int, default=50, help="Nombre de voisins k-NN (collab + contenu).")
    p.add_argument("--min-rating", type=int, default=3, help="Seuil de rating pour considérer un 'like' (collab).")
    p.add_argument("--content-min-df", type=float, default=2, help="min_df TF-IDF (peut être float pour ratio).")
    p.add_argument("--content-ngram", type=_parse_tuple, default=(1,2), help="ngram_range ex: '1,2'.")
    p.add_argument("--content-max-features", type=int, default=50000, help="nb max de termes TF-IDF.")
    p.add_argument("--content-title-weight", type=float, default=2.0, help="poids du titre (répétition naïve).")
    p.add_argument("--content-authors-weight", type=float, default=1.0, help="poids des auteurs (répétition naïve).")
    p.add_argument("--content-extra-cols", type=_parse_list, default=tuple(), help="colonnes texte supplémentaires séparées par des virgules (ex: description,genres).")
    p.add_argument("--content-stopwords", type=str, default="english", help="english|fr|none (par défaut: english).")

    args = p.parse_args()

    try:
        train_and_serialize(
            n_neighbors=args.n_neighbors,
            min_rating=args.min_rating,
            content_min_df=args.content_min_df,
            content_ngram=args.content_ngram,
            content_max_features=args.content_max_features,
            content_title_weight=args.content_title_weight,
            content_authors_weight=args.content_authors_weight,
            content_extra_cols=args.content_extra_cols,
            content_stopwords=args.content_stopwords,
        )
    except Exception as e:
        print("❌ Erreur durant l'entraînement:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
