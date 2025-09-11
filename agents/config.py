# agents/config.py
import os, yaml
from typing import Any, Dict

_BASE = os.path.dirname(os.path.dirname(__file__))

print(f"Chargement config depuis {os.path.join(_BASE, 'config')}")

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_agents_yaml() -> Dict[str, Any]:
    return _load_yaml(os.path.join(_BASE, "config", "agents.yaml"))

def load_tasks_yaml() -> Dict[str, Any]:
    return _load_yaml(os.path.join(_BASE, "config", "tasks.yaml"))

# -------- Helpers spécifiques
def router_prompt(question: str) -> str:
    t = load_tasks_yaml().get("router_task", {})
    return (t.get("description") or "").replace("{question}", question)

def router_expected() -> str:
    t = load_tasks_yaml().get("router_task", {})
    return t.get("expected_output") or "title|text|id|details|collab|hybrid|generate"

def retriever_default_count() -> int:
    """
    Lis un éventuel default dans tasks.yaml:
    retriever_task:
      defaults:
        count: 3
    Sinon fallback 3.
    """
    t = load_tasks_yaml().get("retriever_task", {})
    return int(((t.get("defaults") or {}).get("count")) or 3)

def generate_system_prompt() -> str:
    """
    Prompt système pour le mode 'generate'. Tu peux aussi l'ajouter dans tasks.yaml si tu veux.
    """
    return ("Tu es un assistant francophone concis. Réponds clairement. "
            "À moins que l'utilisateur demande un autre nombre, propose au plus 3 éléments.")
