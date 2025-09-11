# agents/crew_setup.py
from __future__ import annotations
import json
from typing import Dict, Any
from crewai import Crew, Process

from agents.agents import make_router_agent, make_executor_agent
from agents.tasks import route_task, execute_task
from agents.tools import (
    SimilarByIdTool, SimilarByTitleTool, SearchContentTool,
    CollabUserTool, HybridUserTool, ItemDetailsTool
)

# ✅ instancie les tools (vrais BaseTool)
TOOLS = [
    SimilarByIdTool(),
    SimilarByTitleTool(),
    SearchContentTool(),
    CollabUserTool(),
    HybridUserTool(),
    ItemDetailsTool(),
]

def run_crew(user_text: str) -> Dict[str, Any]:
    # 1) Router
    router_agent = make_router_agent()
    r_task = route_task(router_agent)
    r_crew = Crew(
        agents=[router_agent],
        tasks=[r_task],
        process=Process.sequential,
        verbose=False,
    )
    route_out = r_crew.kickoff(inputs={"input": user_text})
    route_raw = str(route_out or "")

    # parse JSON robuste
    try:
        pos = route_raw.rfind("{")
        route_payload = json.loads(route_raw[pos:] if pos != -1 else route_raw)
    except Exception:
        route_payload = {"action": "smalltalk", "message": "Salut ! Comment puis-je t’aider ?"}

    # 2) Executor (avec tools à la création)
    exec_agent = make_executor_agent(TOOLS)
    e_task = execute_task(exec_agent)
    e_crew = Crew(
        agents=[exec_agent],
        tasks=[e_task],
        process=Process.sequential,
        verbose=False,
    )
    ctx = json.dumps(route_payload, ensure_ascii=False)
    final = e_crew.kickoff(inputs={"input": user_text, "context": ctx})
    raw = str(final or "")
    return {"raw": raw, "route": route_payload}
