# agents/llm.py — lit config/agents.yaml pour choisir le modèle
import os, yaml
from langchain_groq import ChatGroq
import streamlit as st

_BASE = os.path.dirname(os.path.dirname(__file__))

def _load_agents_yaml():
    p = os.path.join(_BASE, "config", "agents.yaml")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _model_for(agent_key: str) -> str:
    cfg = _load_agents_yaml().get(agent_key, {})
    llm = (cfg.get("llm") or "").strip()
    if llm.startswith("groq/"):
        return llm.split("/", 1)[1]
    return "meta-llama/llama-4-maverick-17b-128e-instruct")

def get_llm(agent_key: str):
    #api_key = os.environ.get("GROQ_API_KEY")
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquant dans l'environnement.")
    return ChatGroq(model=_model_for(agent_key), temperature=0.2)
