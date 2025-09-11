# agents/router.py — point d’entrée appelé par Streamlit
from __future__ import annotations
from agents.crew import BookRecoCrew

def run_turn(user_message: str, memory_summary: str) -> str:
    crew = BookRecoCrew()
    return crew.process_with_streamlit_memory(user_message, memory_summary)
