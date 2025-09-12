# api/main.py
from __future__ import annotations

try:
    import sys
    import pysqlite3  # via pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.service import (
    api_health,
    get_item_details_fn,
    get_items_batch_fn,
    similar_items_by_title_fn,
    similar_items_by_id_fn,
    search_content_fn,
    collab_user_fn,
    hybrid_user_fn,
    titles_search_fn,
)

app = FastAPI(
    title="SmartRecomAI API",
    version="2.0.0",
    description="Endpoints pour recommandations (contenu, collaboratif, hybride) + utilitaires titres.",
)

# CORS permissif (ajuste si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Health --------------------------------- #
@app.get("/health")
def health():
    """Vérifie le chargement des artefacts et du cache interne."""
    try:
        return api_health()
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ------------------------------ Items --------------------------------- #
@app.get("/items/{book_id}")
def get_item_details(book_id: int):
    """Fiche détaillée d’un livre."""
    try:
        return get_item_details_fn(book_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/items")
def get_items_batch(ids: str = Query(..., description="Liste d'IDs, ex: 42,12,7")):
    """Fiches détaillées pour plusieurs IDs (séparés par des virgules)."""
    try:
        id_list: List[int] = [int(x) for x in ids.split(",") if x.strip().isdigit()]
        return get_items_batch_fn(id_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------------------------- Similarités ------------------------------- #
@app.get("/items/by_title/similar")
def similar_items_by_title(
    title: str = Query(..., description="Titre approx (fuzzy)"),
    top_k: int = Query(10, ge=1, le=200),
    details: bool = True,
    include_seed: bool = False,
):
    """Similaires à partir d’un titre (moteur 'soup' + fuzzy)."""
    try:
        return similar_items_by_title_fn(title, top_k, details, include_seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items/{book_id}/similar")
def similar_items_by_id(
    book_id: int,
    top_k: int = Query(10, ge=1, le=200),
    details: bool = True,
    include_seed: bool = False,
):
    """Similaires à partir d’un ID (moteur contenu/tfidf)."""
    try:
        return similar_items_by_id_fn(book_id, top_k, details, include_seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------- Contenu (RAG) ---------------------------- #
@app.get("/search")
def search_content(
    query: str = Query(..., description="Requête texte libre"),
    top_k: int = Query(10, ge=1, le=200),
    mode: str = Query("tfidf", enum=["tfidf", "soup"], description="Moteur contenu"),
    details: bool = True,
):
    """Recherche basée contenu (tfidf) ou 'soup'."""
    try:
        return search_content_fn(query, top_k, mode, details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------- Collaboratif ------------------------------ #
@app.get("/users/{user_id}/collab")
def collab_user(
    user_id: int,
    top_k: int = Query(10, ge=1, le=200),
    exclude_seen: bool = Query(True, description="Exclure les livres déjà notés"),
    popularity_thres: Optional[int] = Query(None, description="Nb min de lecteurs par livre"),
    ratings_thres: Optional[int] = Query(None, description="Nb min de notes par utilisateur"),
    agg: str = Query("sum", enum=["sum", "max"], description="Agrégation des similarités"),
    normalize_by_likes: bool = Query(False, description="Diviser le score par #items aimés"),
    details: bool = True,
):
    """Recommandations collaboratives pour un utilisateur."""
    try:
        return collab_user_fn(
            user_id=user_id,
            top_k=top_k,
            exclude_seen=exclude_seen,
            popularity_thres=popularity_thres,
            ratings_thres=ratings_thres,
            agg=agg,
            normalize_by_likes=normalize_by_likes,
            details=details,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------ Hybride -------------------------------- #
@app.get("/users/{user_id}/hybrid")
def hybrid_user(
    user_id: int,
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="Poids du collaboratif (0=contenu, 1=collab)"),
    top_k: int = Query(10, ge=1, le=200),
    details: bool = True,
):
    """Fusion collaboratif + contenu (min-max) avec pondération alpha."""
    try:
        return hybrid_user_fn(user_id, alpha, top_k, details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------- Diagnostic titres -------------------------- #
@app.get("/titles/search")
def titles_search(
    q: str = Query(..., description="Titre approx (fuzzy)"),
    top_n: int = Query(10, ge=1, le=50),
):
    """Recherche fuzzy de titres (diagnostic)."""
    try:
        return titles_search_fn(q, top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Exécution directe : uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
