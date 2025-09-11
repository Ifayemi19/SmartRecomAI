# streamlit_app.py — UI ultra-minimal (Groq-only, SANS health)
from __future__ import annotations
import os, traceback
import streamlit as st
from dotenv import load_dotenv

# Charge .env (pour GROQ_API_KEY utilisé côté Crew)
load_dotenv()

# ---- Multi-agents ----
try:
    from agents.router import run_turn as router_run_turn  # type: ignore
    CREW_READY = True
except Exception as e:
    CREW_READY = False
    CREW_IMPORT_ERROR = e

def run_agents_turn(message: str, memory_summary: str) -> str:
    if not CREW_READY:
        return """[Multi-agents indisponible] Vérifie la structure du projet et les dépendances :
- agents/__init__.py
- agents/crew.py
- agents/router.py
- agents/tools.py
- api/__init__.py
- api/service.py

Erreur: {}""".format(CREW_IMPORT_ERROR)
    try:
        return str(router_run_turn(message, memory_summary))
    except Exception:
        return "[Erreur] Le router multi-agents a échoué:\n" + traceback.format_exc()

# ---- Page ----
st.set_page_config(page_title="SmartRecomAI (Groq)", page_icon="📚", layout="wide")

st.markdown("### 📚 SmartRecomAI")
st.caption("Discutez de vos goûts, demandez des titres similaires, ou utilisez votre user_id.")

# ---- État de session ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = ""

# ---- Historique ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Entrée utilisateur ----
user_input = st.chat_input("Écrivez votre message…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion des agents…"):
            answer = run_agents_turn(user_input, st.session_state.memory_summary)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Mémoire (résumé ultra simple stocké dans st.session_state)
    st.session_state.memory_summary = (st.session_state.memory_summary + " " + user_input + " " + answer)[-4000:]

st.markdown(
    "<hr style='border: none; border-top: 1px solid #eee; margin: 1rem 0;' />",
    unsafe_allow_html=True
)
st.caption("Astuce: utilisez un *book id* (ex: id=42), un *titre*, ou votre *user_id* pour guider le routeur.")
