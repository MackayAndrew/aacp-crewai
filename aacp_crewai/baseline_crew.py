"""
BaselineCrew
Standard CrewAI orchestration WITHOUT AACP.
Natural language task descriptions per hop.
Used by comparison.py to show what AACP replaces.
"""

import os
import re
import json
import time
from pathlib import Path

MODEL_COSTS = {
    "gpt-4.1-mini": 0.40,
    "gpt-4.1":      2.00,
    "gpt-4o":       5.00,
    "gpt-4o-mini":  0.15,
}


class BaselineCrewResult:
    def __init__(self, workflow, model):
        self.workflow   = workflow
        self.model      = model
        self.agent_hops = []
        self.outputs    = {}
        self.success    = True

    @property
    def agent_cost(self):
        return round(sum(h.get("cost_usd", 0) for h in self.agent_hops), 6)

    @property
    def total_cost(self):
        return self.agent_cost

    @property
    def agent_tokens(self):
        return sum(h.get("tokens_in", 0) for h in self.agent_hops)


class BaselineCrew:
    """Standard CrewAI WITHOUT AACP -- natural language task descriptions."""

    def __init__(self, model="gpt-4.1-mini", api_key=None,
                 data_dir="data", output_dir="output", verbose=True):
        self.model      = model
        self.api_key    = api_key or os.environ.get("OPENAI_API_KEY")
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose
        self._cpm       = MODEL_COSTS.get(model, 1.0)
        self.result     = BaselineCrewResult("payroll", model)
        self._llm       = None

    def _get_llm(self):
        if not self._llm:
            from crewai import LLM
            self._llm = LLM(
                model=f"openai/{self.model}",
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=2000,
            )
        return self._llm

    def _make_agent(self, role, goal, backstory):
        from crewai import Agent
        return Agent(
            role=role, goal=goal, backstory=backstory,
            llm=self._get_llm(), verbose=False, allow_delegation=False,
        )

    def _run_task(self, agent, description, data=None):
        from crewai import Task, Crew
        full = description
        if data:
            full += f"\n\nData:\n{json.dumps(data, indent=2)}"
        full += "\n\nRespond with valid JSON only. No markdown fences."

        task  = Task(description=full, expected_output="Valid JSON", agent=agent)
        start = time.time()
        crew  = Crew(agents=[agent], tasks=[task], verbose=False)
        out   = crew.kickoff()
        latency = (time.time() - start) * 1000

        tokens_est = len(full.split()) * 2
        cost_usd   = tokens_est / 1_000_000 * self._cpm

        self.result.agent_hops.append({
            "description": description[:80],
            "tokens_in":   tokens_est,
            "cost_usd":    round(cost_usd, 6),
            "latency_ms":  round(latency, 0),
        })

        if self.verbose:
            print(f'  [NL task] "{description[:65]}"  ${cost_usd:.4f}')

        try:
            raw = out.raw
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw.strip())
        except Exception:
            return {"raw": out.raw[:200]}

    def run_payroll(self, period="2026-03"):
        import csv
        emp_data = list(csv.DictReader(
            open(self.data_dir / f"employees_{period}.csv")))
        bud_data = list(csv.DictReader(
            open(self.data_dir / f"budgets_{period}.csv")))

        hr_agent  = self._make_agent(
            "HR Payroll Specialist",
            "Process payroll data accurately",
            "Expert in UK payroll, PAYE and pension calculations.",
        )
        fin_agent = self._make_agent(
            "Finance Controller",
            "Manage budget data and financial calculations",
            "Expert in cost centre budgets and financial reporting.",
        )

        print(f"\n  --- Payroll workflow (natural language tasks) ---")

        r1 = self._run_task(hr_agent,
            f"Retrieve all active employee salary records for period {period}. "
            f"Include employee ID, name, department, cost centre, base salary, "
            f"delta, gross pay and pension rate. Return as JSON.",
            {"employees": emp_data})

        r2 = self._run_task(fin_agent,
            f"Retrieve cost centre budget data for {period}. Calculate YTD "
            f"utilisation percentage and flag any exceeding 85% of approved "
            f"annual budget. Return as JSON.",
            {"budgets": bud_data})

        r3 = self._run_task(hr_agent,
            f"Calculate the full payroll for {period}. Apply PAYE at 20%, "
            f"pension deductions per rate, compute net pay. Flag cost centres "
            f"breaching 90% of budget. Return as JSON.",
            {"employees": r1, "budgets": r2})

        r4 = self._run_task(hr_agent,
            f"Generate an executive payroll summary report for {period}. "
            f"Include key figures, anomalies, and recommended actions. "
            f"Return as JSON.",
            {"payroll": r3})

        from .agent import AuditAgent
        AuditAgent().receive("LOG audit", {"period": period})

        self.result.outputs = {
            "employees": r1, "budgets": r2, "payroll": r3, "report": r4
        }
        return self.result
