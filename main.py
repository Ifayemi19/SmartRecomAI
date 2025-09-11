# api/main.py
from __future__ import annotations
import os, math, re, difflib
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation

# sklearn pour "soup" (content-based par titre)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dossier des artefacts
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

# Moteurs internes (tes fichiers core)
from core.content_infer import ContentEngine
from core.collab_infer import CollabEngine
from core.hybrid import fuse_scores

# Localisation des CSV
from data_loader import find_file

TAGS_METADATA = [
    {"name": "health", "description": "État de l’API et chargement des artefacts."},
    {"name": "content", "description": "Recommandations basées contenu (TF-IDF ou 'soup')."},
    {"name": "items", "description": "Détails d’un livre & similaires (ID ou titre)."},
    {"name": "collab", "description": "Recommandations collaboratives item-item."},
    {"name": "hybrid", "description": "Fusion collaboratif + contenu (scores min-max)."},
]

app = FastAPI(
    title="SmartRecomAI API",
    description="Endpoints pour tester les recommandations (contenu, collaboratif, hybride).",
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Singletons moteurs ----------
_CONTENT: Optional[ContentEngine] = None
_COLLAB: Optional[CollabEngine] = None
_BOOKS_RAW: Optional[pd.DataFrame] = None  # cache books.csv

def content_engine() -> ContentEngine:
    global _CONTENT
    if _CONTENT is None:
        _CONTENT = ContentEngine(artifacts_dir=ARTIFACTS_DIR)
    return _CONTENT

def collab_engine() -> CollabEngine:
    global _COLLAB
    if _COLLAB is None:
        _COLLAB = CollabEngine(artifacts_dir=ARTIFACTS_DIR)
    return _COLLAB

# ---------- books.csv helpers ----------
BOOKS_REQUIRED_COLS = [
    "id","book_id","best_book_id","work_id","books_count","isbn","isbn13","authors",
    "original_publication_year","original_title","title","language_code",
    "average_rating","ratings_count","work_ratings_count","work_text_reviews_count",
    "ratings_1","ratings_2","ratings_3","ratings_4","ratings_5","image_url","small_image_url"
]

def _load_books_raw() -> pd.DataFrame:
    """Charge goodbooks-10k/books.csv et normalise les types essentiels."""
    global _BOOKS_RAW
    if _BOOKS_RAW is not None:
        return _BOOKS_RAW
    p = find_file("books.csv")
    if not p:
        raise FileNotFoundError("books.csv introuvable. Place-le dans data/raw/goodbooks-10k/ (ou GOODBOOKS_DIR).")
    df = pd.read_csv(p, dtype={"isbn": str, "isbn13": str})
    for c in BOOKS_REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ["id","book_id","best_book_id","work_id","books_count","ratings_count",
              "work_ratings_count","work_text_reviews_count","ratings_1","ratings_2",
              "ratings_3","ratings_4","ratings_5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["isbn"] = df["isbn"].astype(str).fillna("").replace({"<NA>": ""})
    df["isbn13"] = df["isbn13"].astype(str).fillna("").replace({"<NA>": ""})
    _BOOKS_RAW = df
    return _BOOKS_RAW

def _norm_isbn_str(s: str) -> Optional[str]:
    """Convertit une chaîne (parfois scientifique) en 10/13 chiffres, tronque à 13 si besoin."""
    if not s or s.strip() == "":
        return None
    s = s.strip()
    digits = "".join(ch for ch in s if ch.isdigit() or ch.upper() == "X")
    if not digits:
        try:
            n = Decimal(s)
            n_int = n.quantize(Decimal("1"))
            digits = str(n_int)
        except (InvalidOperation, ValueError):
            return None
    if len(digits) > 13:
        digits = digits[:13]
    return digits

# ---------- Safe helpers (éviter Series ambigus) ----------
def _safe_str(x: Any) -> str:
    """Convertit n'importe quelle valeur pandas/numpy en str sûre (jamais Series/NA)."""
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        if isinstance(x, (np.integer,)):
            return str(int(x))
        if isinstance(x, (np.floating,)):
            v = float(x)
            return "" if (not math.isfinite(v)) else str(v)
        if isinstance(x, (np.bool_, bool, int, float)):
            return "" if (isinstance(x, float) and not math.isfinite(x)) else str(x)
        if isinstance(x, (pd.Series, pd.Index, np.ndarray)):
            try:
                return str(x.tolist())
            except Exception:
                return str(x)
        return str(x)
    except Exception:
        return ""

def _row_to_book_dict(row: pd.Series) -> Dict[str, Any]:
    """Uniformise les champs retournés pour un livre (issus de books.csv)."""
    def _i(v):
        try:
            return int(v) if pd.notna(v) else None
        except Exception:
            return None
    def _f(v):
        try:
            val = float(v)
            return val if math.isfinite(val) else None
        except Exception:
            return None
    return {
        "id": _i(row.get("id")),
        "book_id": _i(row.get("book_id")),
        "best_book_id": _i(row.get("best_book_id")),
        "work_id": _i(row.get("work_id")),
        "books_count": _i(row.get("books_count")),
        "isbn": _norm_isbn_str(str(row.get("isbn")) if row.get("isbn") is not None else ""),
        "isbn13": _norm_isbn_str(str(row.get("isbn13")) if row.get("isbn13") is not None else ""),
        "authors": _safe_str(row.get("authors")),
        "original_publication_year": _f(row.get("original_publication_year")),
        "original_title": _safe_str(row.get("original_title")),
        "title": _safe_str(row.get("title")),
        "language_code": _safe_str(row.get("language_code")),
        "average_rating": _f(row.get("average_rating")),
        "ratings_count": _i(row.get("ratings_count")),
        "work_ratings_count": _i(row.get("work_ratings_count")),
        "work_text_reviews_count": _i(row.get("work_text_reviews_count")),
        "ratings_1": _i(row.get("ratings_1")),
        "ratings_2": _i(row.get("ratings_2")),
        "ratings_3": _i(row.get("ratings_3")),
        "ratings_4": _i(row.get("ratings_4")),
        "ratings_5": _i(row.get("ratings_5")),
        "image_url": _safe_str(row.get("image_url")),
        "small_image_url": _safe_str(row.get("small_image_url")),
    }

def _get_book_info(book_id: int) -> Optional[Dict[str, Any]]:
    """
    Cherche d'abord sur books.csv['id'] == book_id (index dataset 1..10000),
    puis fallback sur books.csv['book_id'] == book_id (Goodreads ID).
    """
    df = _load_books_raw()
    bid = int(book_id)
    hit = df.loc[df["id"] == bid]
    if not hit.empty:
        return _row_to_book_dict(hit.iloc[0])
    hit = df.loc[df["book_id"] == bid]
    if not hit.empty:
        return _row_to_book_dict(hit.iloc[0])
    return None

def _merge_basic_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Complète title/authors/lang/rating depuis books.csv sans booléens ambigus."""
    rr = dict(r)

    if "score" in rr:
        try:
            s = float(rr["score"])
            rr["score"] = s if math.isfinite(s) else 0.0
        except Exception:
            rr["score"] = 0.0

    title_str = _safe_str(rr.get("title"))
    if title_str and title_str != "(titre indisponible)":
        rr["title"] = title_str
        return rr

    bid = rr.get("book_id")
    info = _get_book_info(int(bid)) if isinstance(bid, (int, np.integer)) else None
    if not info:
        rr["title"] = title_str or rr.get("title") or ""
        rr["authors"] = _safe_str(rr.get("authors"))
        rr["language_code"] = _safe_str(rr.get("language_code"))
        return rr

    title2 = _safe_str(info.get("title")) or _safe_str(info.get("original_title"))
    rr["title"] = title2 or title_str or "(titre indisponible)"
    rr["authors"] = _safe_str(info.get("authors")) or _safe_str(rr.get("authors"))
    rr["language_code"] = _safe_str(info.get("language_code")) or _safe_str(rr.get("language_code"))

    try:
        ar = info.get("average_rating")
        rr["average_rating"] = float(ar) if (ar is not None and math.isfinite(float(ar))) else rr.get("average_rating")
    except Exception:
        pass
    try:
        rc = info.get("ratings_count")
        rr["ratings_count"] = int(rc) if rc is not None else rr.get("ratings_count", 0)
    except Exception:
        rr["ratings_count"] = rr.get("ratings_count", 0)

    return rr

# ---------- JSON sanitisation robuste ----------
def _json_safe(obj: Any) -> Any:
    """Convertit récursivement l'objet en structures JSON-serializables."""
    # pandas structures
    if isinstance(obj, pd.DataFrame):
        return [_json_safe(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {k: _json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, (pd.Index,)):
        return [_json_safe(v) for v in obj.tolist()]

    # numpy
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # conteneurs Python
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # scalaires Python
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def _fix_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        rr = dict(r)
        if "score" in rr:
            try:
                s = float(rr["score"]); rr["score"] = s if math.isfinite(s) else 0.0
            except Exception:
                rr["score"] = 0.0
        out.append(rr)
    return out

def _attach_details_safe(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        rr = dict(r)
        bid = rr.get("book_id")
        try:
            rr["book"] = _get_book_info(int(bid)) if isinstance(bid, (int, np.integer)) else None
        except Exception:
            rr["book"] = None
        out.append(rr)
    return out

# ---------- Normalisation sûre (plus de bool implicite) ----------
def _norm_for_match(s: Any) -> str:
    """Normalise pour le matching sans jamais évaluer la vérité d'un objet pandas."""
    if s is None:
        s = ""
    else:
        try:
            if pd.isna(s):
                s = ""
        except Exception:
            pass
    s = str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9x]+", "", s)
    return s

# ---------- Soup cache (CountVectorizer sur title+authors+rating) ----------
_SOUP_READY = False
_SOUP_DF: Optional[pd.DataFrame] = None
_SOUP_VECT: Optional[CountVectorizer] = None
_SOUP_MAT = None  # scipy sparse matrix
_SOUP_ALL_KEYS: List[str] = []       # pour difflib

def _build_soup_cache() -> None:
    global _SOUP_READY, _SOUP_DF, _SOUP_VECT, _SOUP_MAT, _SOUP_ALL_KEYS
    if _SOUP_READY:
        return

    df = _load_books_raw().copy()
    work = df[["id","title","original_title","authors","language_code","average_rating","ratings_count"]].copy()

    # cast sûr -> str
    for c in ["title","original_title","authors","average_rating"]:
        work[c] = work[c].fillna("").astype(str)

    # soup (comme dans ton notebook)
    work["soup"] = (
        work["original_title"].str.lower().str.replace(" ", "", regex=False) + " " +
        work["authors"].str.lower().str.replace(" ", "", regex=False) + " " +
        work["average_rating"].str.lower().str.replace(" ", "", regex=False)
    )

    # normalisation vectorisée (PAS d'appel de _norm_for_match sur une Series)
    t1 = work["title"].map(_norm_for_match)
    t2 = work["original_title"].map(_norm_for_match)
    work["norm_title"] = (t1 + " " + t2)

    # clés pour fuzzy
    keys = set()
    for a, b in zip(t1, t2):
        if a:
            keys.add(a)
        if b:
            keys.add(b)
    _SOUP_ALL_KEYS = sorted([k for k in keys if k])

    vect = CountVectorizer(stop_words="english")
    mat = vect.fit_transform(work["soup"])

    _SOUP_DF, _SOUP_VECT, _SOUP_MAT = work.reset_index(drop=True), vect, mat
    _SOUP_READY = True

def _find_row_by_title(query: str) -> Optional[int]:
    """Renvoie l'index de ligne (dans _SOUP_DF) le plus proche du titre donné."""
    _build_soup_cache()
    assert _SOUP_DF is not None
    qn = _norm_for_match(query)
    if not qn:
        return None

    # 1) match exact
    exact = _SOUP_DF["norm_title"] == qn
    hit = np.where(exact.values)[0]
    if hit.size > 0:
        return int(hit[0])

    # 2) fuzzy via difflib sur ALL_KEYS
    if _SOUP_ALL_KEYS:
        close = difflib.get_close_matches(qn, _SOUP_ALL_KEYS, n=1, cutoff=0.6)
        if close:
            key = close[0]
            for i, nt in enumerate(_SOUP_DF["norm_title"]):
                if key and key in nt:
                    return i
    return None

def _similar_by_row(row_ix: int, top_k: int, include_seed: bool) -> List[Dict[str, Any]]:
    """Cosine sur la matrice 'soup' → liste de dicts (typage natif JSON, jamais pd.NA/Series)."""
    _build_soup_cache()
    assert _SOUP_DF is not None and _SOUP_MAT is not None

    sims = cosine_similarity(_SOUP_MAT[row_ix], _SOUP_MAT).ravel()
    order = np.argsort(-sims)  # décroissant

    results: List[Dict[str, Any]] = []
    wanted = top_k + (0 if include_seed else 1)

    for idx in order:
        if not include_seed and int(idx) == int(row_ix):
            continue

        row = _SOUP_DF.iloc[int(idx)]

        title1 = _safe_str(row.get("title"))
        title2 = _safe_str(row.get("original_title"))
        title_out = title1 if title1 else (title2 if title2 else "(titre indisponible)")

        authors_out = _safe_str(row.get("authors"))
        lang_out = _safe_str(row.get("language_code"))

        avg_out = None
        try:
            v = float(row.get("average_rating"))
            avg_out = v if math.isfinite(v) else None
        except Exception:
            avg_out = None

        cnt_out = 0
        try:
            v = row.get("ratings_count")
            cnt_out = int(v) if pd.notna(v) else 0
        except Exception:
            cnt_out = 0

        bid_out = None
        try:
            v = row.get("id")
            bid_out = int(v) if pd.notna(v) else None
        except Exception:
            bid_out = None

        score_out = float(sims[int(idx)])

        results.append({
            "book_id": bid_out,
            "title": title_out,
            "authors": authors_out,
            "language_code": lang_out,
            "average_rating": avg_out,
            "ratings_count": cnt_out,
            "score": score_out
        })

        if len(results) >= wanted:
            break

    if not include_seed and results and len(results) > 0:
        results = results[:top_k]
    else:
        results = results[:top_k]

    return results

# ---------- Wrappers tolérants (signatures différentes possibles) ----------
def _content_recommend_by_book(ce: ContentEngine, book_id: int, top_k: int):
    try:
        return ce.recommend_by_book(book_id, top_k=top_k, strict_unfiltered=True)
    except TypeError:
        return ce.recommend_by_book(book_id, top_k=top_k)

def _content_recommend_by_text(ce: ContentEngine, query: str, top_k: int):
    try:
        return ce.recommend_by_text(query, top_k=top_k, strict_unfiltered=True)
    except TypeError:
        return ce.recommend_by_text(query, top_k=top_k)

def _content_user_centroid(ce: ContentEngine, user_id: int, top_k: int, exclude_seen: bool = True):
    try:
        return ce.recommend_for_user_centroid(user_id=user_id, top_k=top_k, exclude_seen=exclude_seen)
    except Exception:
        return []

def _collab_recommend(eng: CollabEngine, **kwargs):
    try:
        return eng.recommend_for_user(**kwargs)
    except TypeError:
        simple = {k: kwargs[k] for k in ("user_id", "k_top", "exclude_seen") if k in kwargs}
        return eng.recommend_for_user(**simple)

# ======================================================================
#                            HEALTH
# ======================================================================

@app.get("/health", tags=["health"])
def health() -> Dict[str, Any]:
    try:
        _ = content_engine()
        _ = collab_engine()
        _ = _load_books_raw()
        _build_soup_cache()
        return _json_safe({"status": "ok", "artifacts": ARTIFACTS_DIR, "books_csv": True})
    except Exception as e:
        return _json_safe({"status": "error", "detail": str(e)})

# ======================================================================
#                            ITEMS (DETAILS)
# ======================================================================

@app.get("/items/{book_id}", tags=["items"])
def get_item_details(book_id: int) -> Dict[str, Any]:
    info = _get_book_info(book_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Livre {book_id} introuvable dans books.csv")
    return _json_safe({"book": info})

@app.get("/items", tags=["items"])
def get_items_batch(ids: str = Query(..., description="Liste d'IDs, ex: 42,12,7")) -> Dict[str, Any]:
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    except Exception:
        raise HTTPException(status_code=400, detail="Paramètre 'ids' invalide")
    results, missing = [], []
    for bid in id_list:
        info = _get_book_info(bid)
        if info: results.append(info)
        else:    missing.append(bid)
    return _json_safe({"count": len(results), "books": results, "missing": missing})

# ======================================================================
#              ITEMS (SIMILAIRES PAR TITRE) — STATIQUE D'ABORD
# ======================================================================

@app.get("/items/by_title/similar", tags=["items"])
def similar_items_by_title(
    title: str = Query(..., description="Titre approx. autorisée (ex: 'The Hobbit')"),
    top_k: int = Query(10, ge=1, le=200),
    details: bool = Query(True, description="Inclure la fiche complète"),
    include_seed: bool = Query(False, description="Inclure le livre source"),
):
    try:
        row_ix = _find_row_by_title(title)
        if row_ix is None:
            raise HTTPException(status_code=404, detail=f"Aucun match/similaire trouvé pour '{title}'")
        results = _similar_by_row(row_ix=row_ix, top_k=top_k, include_seed=include_seed)
        results = _fix_scores([_merge_basic_fields(r) for r in results])
        if details:
            results = _attach_details_safe(results)
        payload = {
            "reason": f"similaire (soup/title) à '{title}' ({'avec' if include_seed else 'sans'} seed)",
            "results": results
        }
        return _json_safe(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alias sans conflit (optionnel)
@app.get("/titles/similar", tags=["items"])
def similar_by_title_alias(
    title: str = Query(..., description="Titre approx. autorisée"),
    top_k: int = Query(10, ge=1, le=200),
    details: bool = Query(True),
    include_seed: bool = Query(False),
):
    return similar_items_by_title(title=title, top_k=top_k, details=details, include_seed=include_seed)

# ======================================================================
#              ITEMS (SIMILAIRES PAR ID) — DYNAMIQUE ENSUITE
# ======================================================================

@app.get("/items/{book_id}/similar", tags=["items"])
def similar_items_by_id(
    book_id: int,
    top_k: int = Query(10, ge=1, le=200),
    details: bool = Query(True, description="Inclure la fiche complète"),
    include_seed: bool = Query(False, description="Inclure le livre source"),
):
    try:
        ce = content_engine()
        want = top_k if include_seed else min(top_k + 1, 1000)
        raw = _content_recommend_by_book(ce, book_id=book_id, top_k=want)
        if not raw:
            raise HTTPException(status_code=404, detail=f"Aucun similaire pour book_id={book_id}")
        if not include_seed:
            raw = [r for r in raw if int(r.get("book_id", -1)) != int(book_id)]
        results = raw[:top_k]
        results = _fix_scores([_merge_basic_fields(r) for r in results])
        if details:
            results = _attach_details_safe(results)
        return _json_safe({
            "reason": f"similaire (tfidf) à {book_id} ({'avec' if include_seed else 'sans'} seed)",
            "results": results
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
#                            CONTENU
# ======================================================================

@app.get("/search", tags=["content"])
def search_content(
    query: str = Query(..., description="Requête texte (ex: 'space opera prince')"),
    top_k: int = Query(10, ge=1, le=200),
    mode: str = Query("tfidf", enum=["tfidf", "soup"], description="Moteur contenu: 'tfidf' ou 'soup'."),
    details: bool = Query(True, description="Inclure la fiche complète"),
):
    try:
        if mode == "soup":
            row_ix = _find_row_by_title(query)
            if row_ix is not None:
                results = _similar_by_row(row_ix=row_ix, top_k=top_k, include_seed=False)
                reason = f"contenu (soup via titre) : '{query}'"
            else:
                results = []
                reason = f"contenu (soup) : '{query}'"
        else:
            ce = content_engine()
            results = _content_recommend_by_text(ce, query=query, top_k=top_k)
            reason = f"contenu (tfidf) : '{query}'"

        results = _fix_scores([_merge_basic_fields(r) for r in results])
        if details:
            results = _attach_details_safe(results)
        return _json_safe({"reason": reason, "results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
#                            COLLABORATIF
# ======================================================================

@app.get("/users/{user_id}/collab", tags=["collab"])
def collab_user(
    user_id: int,
    top_k: int = Query(10, ge=1, le=200),
    exclude_seen: bool = Query(True, description="Exclure les livres déjà notés"),
    popularity_thres: Optional[int] = Query(None, description="Nb min de lecteurs par livre (ex: 60)."),
    ratings_thres: Optional[int] = Query(None, description="Nb min de notes par utilisateur (ex: 50)."),
    agg: str = Query("sum", enum=["sum", "max"], description="Agrégation des similarités."),
    normalize_by_likes: bool = Query(False, description="Diviser le score par #items aimés"),
    details: bool = Query(True, description="Inclure la fiche complète"),
):
    try:
        eng = collab_engine()
        ids_scores = _collab_recommend(
            eng,
            user_id=user_id,
            k_top=top_k * 3,
            exclude_seen=exclude_seen,
            topn_from_each_liked=50,
            agg=agg,
            normalize_by_likes=normalize_by_likes,
            popularity_thres=popularity_thres,
            ratings_thres=ratings_thres,
        )
        enriched = eng.enrich(ids_scores)[:top_k]
        if not enriched:
            raise HTTPException(status_code=404, detail=f"Aucune reco collab pour user_id={user_id}")
        enriched = _fix_scores([_merge_basic_fields(r) for r in enriched])
        if details:
            enriched = _attach_details_safe(enriched)
        return _json_safe({"reason": f"collaboratif pour user {user_id}", "results": enriched})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
#                              HYBRIDE
# ======================================================================

@app.get("/users/{user_id}/hybrid", tags=["hybrid"])
def hybrid_user(
    user_id: int,
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="Poids du collaboratif (0=contenu, 1=collab)"),
    top_k: int = Query(10, ge=1, le=200),
    details: bool = Query(True, description="Inclure la fiche complète"),
):
    try:
        collab = collab_engine()
        content = content_engine()
        collab_ids_scores: List[Tuple[int, float]] = _collab_recommend(
            collab, user_id=user_id, k_top=top_k * 10, exclude_seen=True
        )
        if not collab_ids_scores:
            raise HTTPException(status_code=404, detail=f"Aucune base collab pour user_id={user_id}")

        content_ids_scores: List[Tuple[int, float]] = []
        content_recs = _content_user_centroid(content, user_id=user_id, top_k=top_k * 10, exclude_seen=True)
        if content_recs:
            content_ids_scores = [(int(r["book_id"]), float(r.get("score", 0.0))) for r in content_recs]
        else:
            content_ids_scores = [(bid, 0.0) for (bid, _) in collab_ids_scores]

        fused = fuse_scores(
            collab=collab_ids_scores,
            content=content_ids_scores,
            alpha=alpha,
            top_k=top_k * 3,
        )
        enriched = collab.enrich(fused)[:top_k]
        enriched = _fix_scores([_merge_basic_fields(r) for r in enriched])
        if details:
            enriched = _attach_details_safe(enriched)
        return _json_safe({"reason": f"hybride α={alpha} pour user {user_id}", "results": enriched})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
#                       RECHERCHE DE TITRES (diagnostic)
# ======================================================================

@app.get("/titles/search", tags=["items"])
def titles_search(
    q: str = Query(..., description="Titre à chercher (approx autorisée)"),
    top_n: int = Query(10, ge=1, le=50)
) -> Dict[str, Any]:
    _build_soup_cache()
    assert _SOUP_DF is not None
    qn = _norm_for_match(q)
    if not qn:
        raise HTTPException(status_code=400, detail="Paramètre q vide")
    rows: List[Tuple[int, float]] = []
    for i, row in _SOUP_DF.iterrows():
        cands = [
            _norm_for_match(_safe_str(row.get("title"))),
            _norm_for_match(_safe_str(row.get("original_title")))
        ]
        best = 0.0
        for c in cands:
            if not c:
                continue
            best = max(best, difflib.SequenceMatcher(None, qn, c).ratio())
        if best > 0:
            rows.append((int(i), float(best)))
    rows.sort(key=lambda x: x[1], reverse=True)
    take = rows[:top_n]
    out: List[Dict[str, Any]] = []
    for i, s in take:
        r = _SOUP_DF.iloc[i]
        avg_val = None
        try:
            vv = float(r.get("average_rating"))
            avg_val = vv if math.isfinite(vv) else None
        except Exception:
            avg_val = None
        out.append({
            "row_ix": int(i),
            "id": int(r["id"]) if pd.notna(r["id"]) else None,
            "title": _safe_str(r.get("title")),
            "original_title": _safe_str(r.get("original_title")),
            "authors": _safe_str(r.get("authors")),
            "average_rating": avg_val,
            "score": float(s)
        })
    return _json_safe({"query": q, "matches": out})
