"""
AACPCrewAgent
A CrewAI-native agent that receives and responds to AACP packets.
CrewAI agent roles map directly to AACP domains.
"""

import os
import re
import json
import time

MODEL_COSTS = {
    "gpt-4.1-mini": 0.40,
    "gpt-4.1":      2.00,
    "gpt-4o":       5.00,
    "gpt-4o-mini":  0.15,
}

AACP_BACKSTORY = """You natively understand AACP v1.1 pipe-delimited coordination packets.
Format: TASK|DOM|return:AGENT|p:PRIORITY|aacp:1.1|key:value...
When given an AACP packet interpret it directly and respond with
valid JSON only. No markdown fences. No preamble."""


class AACPCrewAgent:
    """
    Wrapper that creates a CrewAI Agent configured to understand AACP packets.
    The agent role maps directly to an AACP domain.
    """

    def __init__(self, role, goal, backstory, domain,
                 model="gpt-4.1-mini", api_key=None):
        self.role    = role
        self.domain  = domain
        self.model   = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._cpm    = MODEL_COSTS.get(model, 1.0)
        self.name    = f"{domain}-AGENT"

        # Lazy import crewai so packet_bus tests work without it installed
        from crewai import Agent, LLM
        self._llm = LLM(
            model=f"openai/{model}",
            api_key=self.api_key,
            temperature=0,
            max_tokens=2000,
        )
        self.crew_agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory + "\n\n" + AACP_BACKSTORY,
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    def receive(self, packet, data=None):
        from crewai import Task, Crew
        parts = [f"Coordination packet:\n{packet}"]
        if data:
            parts.append(f"\nData:\n{json.dumps(data, indent=2)}")
        parts.append("\nRespond with valid JSON only. No markdown fences.")

        task = Task(
            description="\n".join(parts),
            expected_output="Valid JSON object",
            agent=self.crew_agent,
        )
        start = time.time()
        try:
            crew   = Crew(agents=[self.crew_agent], tasks=[task], verbose=False)
            raw    = crew.kickoff().raw
            latency_ms = (time.time() - start) * 1000
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            result   = json.loads(raw.strip())
            tokens_est = len(json.dumps(data or {}).split()) * 2
            cost_usd   = tokens_est / 1_000_000 * self._cpm
            return {
                "result": result, "tokens_in": tokens_est,
                "tokens_out": len(raw.split()) * 2,
                "latency_ms": round(latency_ms, 0),
                "cost_usd": round(cost_usd, 6), "error": None,
            }
        except json.JSONDecodeError as e:
            return {"result": None, "tokens_in": 0, "tokens_out": 0,
                    "latency_ms": round((time.time()-start)*1000, 0),
                    "cost_usd": 0.0, "error": f"JSON parse error: {e}"}
        except Exception as e:
            return {"result": None, "tokens_in": 0, "tokens_out": 0,
                    "latency_ms": round((time.time()-start)*1000, 0),
                    "cost_usd": 0.0, "error": str(e)}


class AuditAgent:
    """Deterministic audit agent. No LLM. No CrewAI. $0.00."""
    name   = "AUDIT-AGENT"
    domain = "LOG"

    def receive(self, packet, data=None):
        return {
            "result": {"logged": True, "ts": time.time()},
            "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 1, "cost_usd": 0.0, "error": None,
        }
