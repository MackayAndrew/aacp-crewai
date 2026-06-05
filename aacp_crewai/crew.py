"""
AACPCrew
AACP-coordinated CrewAI orchestrator.

CrewAI maps to AACP more naturally than any other framework:
  Agent role  → AACP DOM  (HR, FIN, IT, SALES, CS)
  Task        → AACP TASK (FETCH, PROC, MERGE, etc.)
  Crew        → AACP Orchestrator

Usage:
    from aacp_crewai import AACPCrew
    crew = AACPCrew(model="gpt-4.1-mini")
    result = crew.run_workflow("payroll", period="2026-03")
    print(result.summary())
"""

import os
from pathlib import Path
from .packet_bus import AACPPacketBus
from .agent      import AACPCrewAgent, AuditAgent


def _make_agents(model, api_key):
    """
    Create specialist agents with roles that map to AACP domains.
    This is the natural fit -- CrewAI roles ARE AACP domains.
    """
    kw = {"model": model, "api_key": api_key}

    hr = AACPCrewAgent(
        role="HR Payroll and People Operations Specialist",
        goal="Process employee data, calculate payroll, manage onboarding",
        backstory="Senior HR specialist with deep expertise in UK payroll, "
                  "PAYE calculations, pension contributions and people operations.",
        domain="HR", **kw,
    )
    finance = AACPCrewAgent(
        role="Finance Controller and Management Accountant",
        goal="Process financial data, manage budgets, reconcile accounts",
        backstory="Experienced finance controller specialising in management accounting, "
                  "cost centre budgets, bank reconciliation and month-end close. "
                  "Always pre-computes numeric values -- never writes arithmetic in JSON.",
        domain="FIN", **kw,
    )
    it = AACPCrewAgent(
        role="IT Systems Administrator and Identity Access Manager",
        goal="Provision and deprovision user accounts, manage licences and access",
        backstory="Senior IT administrator specialising in Active Directory, "
                  "Entra ID, licence management and identity access management.",
        domain="IT", **kw,
    )
    sales = AACPCrewAgent(
        role="Sales Operations and CRM Analyst",
        goal="Qualify leads, route opportunities, track pipeline",
        backstory="Sales operations specialist with expertise in BANT scoring, "
                  "CRM management and pipeline analysis.",
        domain="SALES", **kw,
    )
    cs = AACPCrewAgent(
        role="Customer Service Resolution Specialist",
        goal="Triage complaints, determine resolution strategies, manage goodwill",
        backstory="Senior CS specialist with expertise in complaint resolution, "
                  "LTV assessment and retention strategy.",
        domain="CS", **kw,
    )
    return {
        "hr": hr, "finance": finance, "it": it,
        "sales": sales, "cs": cs, "audit": AuditAgent(),
    }


class AACPCrew:
    """AACP-coordinated CrewAI multi-agent workflows."""

    def __init__(self, model="gpt-4.1-mini", api_key=None,
                 data_dir="data", output_dir="output", verbose=True):
        self.model      = model
        self.api_key    = api_key or os.environ.get("OPENAI_API_KEY")
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose
        self.output_dir.mkdir(exist_ok=True)
        self._agents    = _make_agents(self.model, self.api_key)

    def run_workflow(self, workflow, **kwargs):
        print(f"\n{'='*60}")
        print(f"  AACP-CrewAI: {workflow.upper()}")
        print(f"  Model: {self.model}")
        print(f"{'='*60}")

        bus = AACPPacketBus(
            workflow=workflow, model=self.model,
            audit_log=str(self.output_dir / "audit_crewai.jsonl"),
            verbose=self.verbose,
        )

        if workflow == "payroll":
            self._run_payroll(bus, **kwargs)
        elif workflow == "it_provisioning":
            self._run_it_provisioning(bus, **kwargs)
        elif workflow == "sales_qualification":
            self._run_sales_qualification(bus, **kwargs)
        else:
            raise ValueError(f"Unknown workflow: {workflow}")

        t = bus.result
        print(f"\n{'-'*60}")
        print(f"  COMPLETE — {workflow}")
        print(f"  Hops:  {len(t.hops)}")
        print(f"  Cost:  ${t.total_cost:.4f}")
        print(f"  Time:  {t.total_latency_ms/1000:.1f}s")
        print(f"{'-'*60}")
        return bus.result

    def _run_payroll(self, bus, period="2026-03"):
        import csv
        import json as _json
        from aacp.encoders.workflows.payroll import PayrollEncoder
        enc = PayrollEncoder()

        emp_file = self.data_dir / f"employees_{period}.csv"
        bud_file = self.data_dir / f"budgets_{period}.csv"
        rul_file = self.data_dir / "payroll_rules.json"
        emp_data = list(csv.DictReader(open(emp_file))) if emp_file.exists() else []
        bud_data = list(csv.DictReader(open(bud_file))) if bud_file.exists() else []
        rules    = _json.load(open(rul_file)) if rul_file.exists() else {}

        r1 = bus.dispatch("ORCHESTRATOR", self._agents["hr"],
            enc.fetch_employees(period).packet,
            {"employees": emp_data, "period": period},
            lambda x: f"{x.get('total_employees',0)} employees")
        if not r1: return

        r2 = bus.dispatch("ORCHESTRATOR", self._agents["finance"],
            enc.fetch_budgets(period).packet,
            {"budgets": bud_data, "period": period},
            lambda x: f"{x.get('flagged_count',0)} flagged")
        if not r2: return

        r3 = bus.dispatch("ORCHESTRATOR", self._agents["hr"],
            enc.merge_and_calculate(period).packet,
            {"employees": r1, "budgets": r2, "rules": rules},
            lambda x: f"gross £{x.get('totals',{}).get('gross',0):,}")
        if not r3: return

        r4 = bus.dispatch("ORCHESTRATOR", self._agents["hr"],
            enc.generate_report(period, period).packet,
            {"payroll_summary": r3, "period": period},
            lambda x: str(x.get("executive_summary",""))[:60])

        bus.dispatch("ORCHESTRATOR", self._agents["audit"],
            enc.log_run(period).packet,
            {"period": period}, lambda x: "Logged")

        bus.result.outputs = {
            "employees": r1, "budgets": r2, "payroll": r3, "report": r4
        }

    def _run_it_provisioning(self, bus, username="j.smith",
                              dept="Engineering", licences=None):
        from aacp.encoders.workflows.jml import JMLEncoder
        enc      = JMLEncoder()
        licences = licences or ["M365", "Slack", "VPN"]

        r1 = bus.dispatch("ORCHESTRATOR", self._agents["hr"],
            enc.fetch_new_hire(f"E-{username}").packet,
            {"username": username, "dept": dept},
            lambda x: str(x.get("employee", {}).get("name", username)))

        r2 = bus.dispatch("ORCHESTRATOR", self._agents["it"],
            enc.create_account(username, dept).packet,
            {"username": username, "dept": dept},
            lambda x: f"email: {x.get('email','?')}")
        if not r2: return

        r3 = bus.dispatch("ORCHESTRATOR", self._agents["it"],
            enc.assign_licences(username, licences).packet,
            {"username": username, "licences": licences},
            lambda x: f"licences: {x.get('licences_assigned', [])}")

        r4 = bus.dispatch("ORCHESTRATOR", self._agents["it"],
            enc.configure_access(username).packet,
            {"username": username},
            lambda x: f"systems: {x.get('systems_granted', [])}")

        r5 = bus.dispatch("ORCHESTRATOR", self._agents["it"],
            enc.send_welcome(username).packet,
            {"username": username}, lambda x: "Welcome sent")

        bus.dispatch("ORCHESTRATOR", self._agents["audit"],
            enc.log_provisioning(username).packet,
            {"username": username}, lambda x: "Logged")

        bus.result.outputs = {"account": r2, "licences": r3, "access": r4}

    def _run_sales_qualification(self, bus, lead_id="L-001", lead=None):
        from aacp.encoders.workflows.sales import SalesEncoder
        enc  = SalesEncoder()
        lead = lead or {
            "id": lead_id, "company": "Demo Corp",
            "budget_gbp": 75000, "timeline_months": 3,
            "need_score": 8, "authority_score": 9, "engaged": True,
        }

        r1 = bus.dispatch("ORCHESTRATOR", self._agents["sales"],
            enc.fetch_lead(lead_id).packet,
            {"lead": lead},
            lambda x: str(x.get("lead", {}).get("company", "?")))
        if not r1: return

        r2 = bus.dispatch("ORCHESTRATOR", self._agents["sales"],
            enc.score_lead(lead_id).packet,
            {"lead": r1, "lead_id": lead_id},
            lambda x: f"Score: {x.get('total_score',0)}/100")
        if not r2: return

        r3 = bus.dispatch("ORCHESTRATOR", self._agents["sales"],
            enc.route_lead(lead_id).packet,
            {"score_result": r2},
            lambda x: f"Stage: {x.get('stage','?')}")

        bus.dispatch("ORCHESTRATOR", self._agents["audit"],
            enc.log_qualification(lead_id).packet,
            {"lead_id": lead_id}, lambda x: "Logged")

        bus.dispatch("ORCHESTRATOR", self._agents["sales"],
            enc.notify_rep(lead_id).packet,
            {"lead_id": lead_id, "routing": r3},
            lambda x: "Rep notified")

        bus.result.outputs = {"lead": r1, "score": r2, "routing": r3}
