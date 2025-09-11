from __future__ import annotations
import json
from typing import Type
from pydantic import BaseModel, Field

# BaseTool CrewAI (selon ta version)
try:
    from crewai.tools import BaseTool  # CrewAI récent
except Exception:
    from crewai_tools import BaseTool  # fallback legacy

# On consomme les FONCTIONS pures de ton service layer
from api import service

# =============== Schemas Pydantic ===============

class HealthArgs(BaseModel):
    pass

class ItemDetailsArgs(BaseModel):
    book_id: int = Field(..., description="ID du livre (books.csv id ou book_id fallback)")

class ItemsBatchArgs(BaseModel):
    ids: str = Field(..., description="IDs séparés par des virgules, ex: '42,12,7'")

class SimilarByIdArgs(BaseModel):
    book_id: int = Field(..., description="ID du livre graine")
    top_k: int = Field(10, ge=1, le=200)
    include_seed: bool = False
    details: bool = True

class SimilarByTitleArgs(BaseModel):
    title: str = Field(..., description="Titre approx (fuzzy)")
    top_k: int = Field(10, ge=1, le=200)
    include_seed: bool = False
    details: bool = True

class SearchContentArgs(BaseModel):
    query: str = Field(..., description="Requête libre")
    top_k: int = Field(10, ge=1, le=200)
    mode: str = Field("tfidf", pattern="^(tfidf|soup)$")
    details: bool = True

class CollabUserArgs(BaseModel):
    user_id: int
    top_k: int = Field(10, ge=1, le=200)
    exclude_seen: bool = True
    popularity_thres: int | None = None
    ratings_thres: int | None = None
    agg: str = Field("sum", pattern="^(sum|max)$")
    normalize_by_likes: bool = False
    details: bool = True

class HybridUserArgs(BaseModel):
    user_id: int
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1, le=200)
    details: bool = True

class TitlesSearchArgs(BaseModel):
    q: str = Field(..., description="Titre approx (fuzzy)")
    top_n: int = Field(10, ge=1, le=50)

# =============== Tools ===============

class HealthTool(BaseTool):
    name: str = "api_health"
    description: str = "Vérifie le chargement des artefacts, books.csv et le cache 'soup'."
    args_schema: Type[BaseModel] = HealthArgs

    def _run(self, **kwargs) -> str:
        res = service.api_health()
        return json.dumps(res, ensure_ascii=False)

class ItemDetailsTool(BaseTool):
    name: str = "item_details"
    description: str = "Retourne la fiche détaillée d’un livre à partir de son ID."
    args_schema: Type[BaseModel] = ItemDetailsArgs

    def _run(self, **kwargs) -> str:
        a = ItemDetailsArgs(**kwargs)
        try:
            res = service.get_item_details_fn(a.book_id)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        return json.dumps(res, ensure_ascii=False)

class ItemsBatchTool(BaseTool):
    name: str = "items_batch"
    description: str = "Retourne plusieurs fiches détails (IDs séparés par ',')."
    args_schema: Type[BaseModel] = ItemsBatchArgs

    def _run(self, **kwargs) -> str:
        a = ItemsBatchArgs(**kwargs)
        try:
            id_list = [int(x) for x in a.ids.split(",") if x.strip().isdigit()]
            res = service.get_items_batch_fn(id_list)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        return json.dumps(res, ensure_ascii=False)

class SimilarByIdTool(BaseTool):
    name: str = "similar_items_by_id"
    description: str = "Recommande des livres similaires (contenu/tfidf) à partir d’un book_id."
    args_schema: Type[BaseModel] = SimilarByIdArgs

    def _run(self, **kwargs) -> str:
        a = SimilarByIdArgs(**kwargs)
        res = service.similar_items_by_id_fn(
            book_id=a.book_id,
            top_k=a.top_k,
            details=a.details,
            include_seed=a.include_seed,
        )
        return json.dumps(res, ensure_ascii=False)

class SimilarByTitleTool(BaseTool):
    name: str = "similar_items_by_title"
    description: str = "Recommande des livres similaires à partir d’un titre (soup+fuzzy)."
    args_schema: Type[BaseModel] = SimilarByTitleArgs

    def _run(self, **kwargs) -> str:
        a = SimilarByTitleArgs(**kwargs)
        res = service.similar_items_by_title_fn(
            title=a.title,
            top_k=a.top_k,
            details=a.details,
            include_seed=a.include_seed,
        )
        return json.dumps(res, ensure_ascii=False)

class SearchContentTool(BaseTool):
    name: str = "search_content"
    description: str = "Recherche de contenu (tfidf ou soup) à partir d’une requête libre."
    args_schema: Type[BaseModel] = SearchContentArgs

    def _run(self, **kwargs) -> str:
        a = SearchContentArgs(**kwargs)
        res = service.search_content_fn(
            query=a.query, top_k=a.top_k, mode=a.mode, details=a.details
        )
        return json.dumps(res, ensure_ascii=False)

class CollabUserTool(BaseTool):
    name: str = "collab_user"
    description: str = "Recommandations collaboratives pour un user_id."
    args_schema: Type[BaseModel] = CollabUserArgs

    def _run(self, **kwargs) -> str:
        a = CollabUserArgs(**kwargs)
        res = service.collab_user_fn(
            user_id=a.user_id,
            top_k=a.top_k,
            exclude_seen=a.exclude_seen,
            popularity_thres=a.popularity_thres,
            ratings_thres=a.ratings_thres,
            agg=a.agg,
            normalize_by_likes=a.normalize_by_likes,
            details=a.details,
        )
        return json.dumps(res, ensure_ascii=False)

class HybridUserTool(BaseTool):
    name: str = "hybrid_user"
    description: str = "Recommandations hybrides (alpha=poids collab) pour un user_id."
    args_schema: Type[BaseModel] = HybridUserArgs

    def _run(self, **kwargs) -> str:
        a = HybridUserArgs(**kwargs)
        res = service.hybrid_user_fn(
            user_id=a.user_id, alpha=a.alpha, top_k=a.top_k, details=a.details
        )
        return json.dumps(res, ensure_ascii=False)

class TitlesSearchTool(BaseTool):
    name: str = "titles_search"
    description: str = "Diagnostic: recherche fuzzy de titres dans books.csv (top_n)."
    args_schema: Type[BaseModel] = TitlesSearchArgs

    def _run(self, **kwargs) -> str:
        a = TitlesSearchArgs(**kwargs)
        try:
            res = service.titles_search_fn(q=a.q, top_n=a.top_n)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        return json.dumps(res, ensure_ascii=False)

# ---------- Tool “généralistes” via Groq (pas d’API livres) ----------

class GeneralGenerationArgs(BaseModel):
    prompt: str = Field(..., description="Question généraliste")

class GenerationTool(BaseTool):
    name: str = "generation_tool"
    description: str = (
        "Répond aux questions généralistes directement avec ChatGroq (aucun accès à l’API livres). "
        "Par défaut, propose 3 livres si la question explicite une reco mais sans contrainte d’API."
    )
    args_schema: Type[BaseModel] = GeneralGenerationArgs

    def _run(self, **kwargs) -> str:
        a = GeneralGenerationArgs(**kwargs)
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.2,
            )
            # Politique : proposer 3 livres par défaut (sauf instruction contraire)
            system_hint = (
                "Tu es un assistant de recommandations de livres. "
                "Sauf indication contraire explicite de l'utilisateur, propose 3 livres pertinents. "
                "Quand tu ne dépends pas de la base de données locale, réponds de façon généraliste."
            )
            resp = llm.invoke(f"{system_hint}\n\nQuestion: {a.prompt}")
            content = getattr(resp, "content", str(resp))
            return json.dumps({"ok": True, "answer": content}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
