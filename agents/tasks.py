# agents/tasks.py
from __future__ import annotations
from crewai import Task

# Task: routage -> sort un JSON strict
def route_task(agent) -> Task:
    return Task(
        description=(
            "Analyse l'INPUT et décide d'une action.\n"
            "Renvoie UNIQUEMENT un JSON.\n"
            "INPUT:\n"
            "{input}"
        ),
        expected_output="Un JSON valide (objet) décrivant l'action (voir système).",
        agent=agent,
        async_execution=False,
    )

# Task: exécution -> lit le JSON et agit
def execute_task(agent) -> Task:
    return Task(
        description=(
            "On te fournit la requête utilisateur et la sortie JSON du routeur.\n"
            "Agis en conséquence: smalltalk ou appel d'outil.\n"
            "INPUT UTILISATEUR:\n"
            "{input}\n\n"
            "ROUTE JSON (du routeur):\n"
            "{context}"
        ),
        expected_output=(
            "- Si smalltalk: un bref message.\n"
            "- Sinon: le JSON renvoyé par l'endpoint (intégral), sans paraphrase."
        ),
        agent=agent,
        async_execution=False,
    )
