# agents/crew.py — CrewAI (Router + Retriever) sans YAML, mémoire passée par Streamlit
from __future__ import annotations
import os
from crewai import Agent, Crew, Process, Task, LLM
import streamlit as st
# Tools basées sur api/service.py (pas d'appels HTTP)
from agents.tools import (
    SimilarByIdTool,
    SimilarByTitleTool,
    SearchContentTool,
    CollabUserTool,
    HybridUserTool,
    TitlesSearchTool,
    GenerationTool,
)

# Clé Groq (depuis .env ou environnement)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def _groq_llm(temp: float = 0.1) -> LLM:
    return LLM(model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",temperature=temp,api_key=GROQ_API_KEY)

class BookRecoCrew:
    """Assemble le crew Router -> Retriever, avec routing par mot-clé."""

    def __init__(self) -> None:
        llm_router = _groq_llm(temp=0.0)
        llm_retr = _groq_llm(temp=0.2)

        # -------- Agents --------
        self.router_agent = Agent(
            role="Router",
            goal="Choisir UN mot-clé de routage pour la question.",
            backstory=(
                "Tu es un routeur. Analyse la question et rends EXACTEMENT UN mot-clé parmi :\n"
                "- by_title (similaires via un titre)\n"
                "- by_id (similaires via un ID)\n"
                "- content (recherche contenu tfidf/soup)\n"
                "- collab (recommandations collaboratives pour user_id)\n"
                "- hybrid (recommandations hybrides pour user_id)\n"
                "- titles_search (diagnostic fuzzy de titres)\n"
                "- general (réponse généraliste via LLM, propose 3 livres par défaut)\n"
                "Répond UNIQUEMENT par ce mot-clé, sans texte additionnel."
            ),
            llm=llm_router,
            verbose=False,
        )

        self.retriever_agent = Agent(
            role="Retriever",
            goal="Répondre en appelant l’outil approprié selon le mot-clé du routeur.",
            backstory=(
                "Tu sais extraire les paramètres (book_id, title, user_id, top_k, mode...) depuis la question. "
                "Tu appelles le bon tool, puis tu formates une réponse claire. "
                "Par défaut, propose 3 livres, sauf si l’utilisateur demande un autre nombre."
            ),
            llm=llm_retr,
            verbose=False,
        )

        # -------- Tasks --------
        self.router_task = Task(
            description=(
                "Question de l'utilisateur : \"{question}\"\n"
                "Contexte mémoire (brut) : \"{memory_summary}\"\n\n"
                "RENVOIE EXACTEMENT UN MOT parmi : by_title | by_id | content | collab | hybrid | titles_search | general.\n"
                "Aucune explication, aucun autre texte."
            ),
            expected_output="Un seul mot-clé : by_title | by_id | content | collab | hybrid | titles_search | general",
            agent=self.router_agent,
        )

        self.retriever_task = Task(
            description=(
                "Question: \"{question}\"\n"
                "Contexte mémoire (brut) : \"{memory_summary}\"\n\n"
                "Décision du routeur (un mot-clé) : la sortie de la tâche précédente.\n\n"
                "Selon le mot-clé :\n"
                "- by_title  -> utilise similar_items_by_title (args: title extrait, top_k=3 par défaut, details=True)\n"
                "- by_id     -> utilise similar_items_by_id (args: book_id extrait, top_k=3 par défaut, details=True)\n"
                "- content   -> utilise search_content (args: query extrait, top_k=3 par défaut, mode= 'tfidf' ou 'soup' si l'utilisateur le dit)\n"
                "- collab    -> utilise collab_user (args: user_id extrait, top_k=3 par défaut)\n"
                "- hybrid    -> utilise hybrid_user (args: user_id extrait, top_k=3 par défaut, alpha=0.6 si non précisé)\n"
                "- titles_search -> utilise titles_search (args: q extrait, top_n=3)\n"
                "- general   -> utilise generation_tool avec la question brute; par défaut propose 3 livres.\n\n"
                "IMPORTANT :\n"
                "1) Extrais proprement les arguments depuis le texte de la question (ex: 'id=42', 'user_id=7', 'top=5', 'mode=soup').\n"
                "2) Si un argument obligatoire manque, demande UNE clarification courte.\n"
                "3) Affiche une réponse claire et concise, avec 3 suggestions par défaut si pertinent."
            ),
            expected_output=(
                "Une réponse utilisateur lisible (listes numérotées pour les livres), avec les titres/auteurs et une courte raison."
            ),
            agent=self.retriever_agent,
            tools=[
                SimilarByIdTool(),
                SimilarByTitleTool(),
                SearchContentTool(),
                CollabUserTool(),
                HybridUserTool(),
                TitlesSearchTool(),
                GenerationTool(),
            ],
        )

        # -------- Crew --------
        self.crew = Crew(
            agents=[self.router_agent, self.retriever_agent],
            tasks=[self.router_task, self.retriever_task],
            process=Process.sequential,
            verbose=False,
        )

    def process_with_streamlit_memory(self, user_message: str, memory_summary: str) -> str:
        """Enrichit les inputs et exécute le crew (mémoire fournie par st.session_state)."""
        inputs = {
            "question": user_message.strip(),
            "memory_summary": (memory_summary or "").strip(),
        }
        result = self.crew.kickoff(inputs=inputs)
        return str(result)
