# agents/agents.py
from __future__ import annotations
from crewai import Agent
from agents.llm import get_llm
from agents.tools import (
    SimilarByIdTool, SimilarByTitleTool, SearchContentTool,
    CollabUserTool, HybridUserTool, ItemDetailsTool
)

llm = get_llm()

agent_title = Agent(
    role="Reco par titre",
    goal="Proposer 3–7 livres similaires à un titre donné (un seul appel outil).",
    backstory="Précis et concis.",
    tools=[SimilarByTitleTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

agent_search = Agent(
    role="Recherche contenu",
    goal="Trouver des livres via requête texte (tfidf/soup) (un seul appel outil).",
    backstory="Exploite la sémantique.",
    tools=[SearchContentTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

agent_sim_by_id = Agent(
    role="Similaires par ID",
    goal="Suggérer 3–7 livres proches d’un book_id (un seul appel outil).",
    backstory="Point de départ: l'ID.",
    tools=[SimilarByIdTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

agent_item_details = Agent(
    role="Fiche livre",
    goal="Restituer la fiche d’un livre par ID (un seul appel outil).",
    backstory="Métadonnées claires.",
    tools=[ItemDetailsTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

agent_collab = Agent(
    role="Reco collaborative",
    goal="Proposer 3–7 livres via historique (user_id) (un seul appel outil).",
    backstory="Item-item maîtrisé.",
    tools=[CollabUserTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

agent_hybrid = Agent(
    role="Reco hybride",
    goal="Mixer contenu + collaboratif (alpha) pour 3–7 titres (un seul appel outil).",
    backstory="Équilibrage des signaux.",
    tools=[HybridUserTool()],
    llm=llm, memory=False, verbose=False, allow_delegation=False
)

__all__ = [
    "agent_title", "agent_search", "agent_sim_by_id", "agent_item_details",
    "agent_collab", "agent_hybrid"
]
