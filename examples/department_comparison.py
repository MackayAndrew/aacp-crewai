"""
examples/department_comparison.py
===================================
Full department day comparison: WITHOUT AACP vs WITH AACP.

Same scope as lab v3 and aacp-langchain department comparison:
  1. JML Onboarding       3 hires  x 6 hops = 18
  2. Payroll                         5 hops =  5
  3. Sales Qualification  3 leads  x 5 hops = 15
  4. CS Resolution        3 tickets x 5 hops = 15
  5. Month-End Close                 6 hops =  6
  ─────────────────────────────────────────────
  Total coordination hops:                   59

Run:
    python3 examples/department_comparison.py --mock
    python3 examples/department_comparison.py --data ../aacp-lab/data
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared mock data and workflow ─────────────────────────────────────────

from aacp_crewai.workflows.department_day import (
    run_department_day,
    write_mock_data,
    MOCK_EMPLOYEES, MOCK_BUDGETS, MOCK_NEW_HIRES,
    MOCK_LEADS, MOCK_TICKETS, MOCK_RULES,
)
from aacp_crewai.packet_bus import AACPPacketBus
from aacp_crewai.agent      import AACPCrewAgent, AuditAgent
from aacp_crewai.agents     import HRAgent, FinanceAgent, ITAgent, SalesAgent, CSAgent


# ── Baseline: WITHOUT AACP ────────────────────────────────────────────────

class BaselineDepartmentDay:
    """
    Same 5 workflows but orchestrator writes natural language
    task descriptions for every coordination hop.
    """

    COORD_PROMPTS = {
        "jml_fetch":    "Retrieve the new hire employee record for {arg}. Include role, department, required licences and systems access.",
        "jml_account":  "Create an Active Directory and Entra ID account for {arg}. Set up email address, group memberships and temporary password.",
        "jml_licences": "Assign the required software licences to {arg}: {extra}. Confirm which licences were assigned and note any failures.",
        "jml_access":   "Configure full system access profile for {arg} including VPN, SharePoint and all role-appropriate systems.",
        "jml_welcome":  "Send a welcome email to {arg} with their credentials, first day instructions and IT contact details.",
        "jml_log":      "Write the provisioning completion audit record for {arg} to the IT compliance trail.",
        "pay_emp":      "Retrieve all active employee salary records for period {arg} including base salary, cost centre and pension rate. Return as JSON.",
        "pay_budget":   "Retrieve cost centre budget allocations for {arg}. Calculate YTD utilisation and flag any approaching or breaching their annual budget.",
        "pay_merge":    "Using the employee and budget data provided, calculate the full payroll for {arg}. Apply PAYE at 20%, pension deductions, and flag cost centres breaching 90 percent of budget.",
        "pay_report":   "Generate an executive payroll summary report for {arg} highlighting anomalies, budget breaches and recommended actions.",
        "pay_log":      "Write the payroll run completion audit record for period {arg} to the HR compliance trail.",
        "sal_fetch":    "Retrieve the full lead profile for {arg} from the CRM including budget, timeline, need score and decision-making authority.",
        "sal_score":    "Score lead {arg} against BANT qualification criteria. Assign scores for Budget, Authority, Need and Timeline and determine if qualified.",
        "sal_route":    "Based on the BANT score for {arg}, route this lead to the appropriate sales rep or nurture sequence with recommended next steps.",
        "sal_log":      "Write the lead qualification outcome for {arg} to the CRM audit trail.",
        "sal_notify":   "Send a notification to the assigned sales representative about qualified lead {arg} with context and recommended next steps.",
        "cs_fetch":     "Retrieve the customer profile for {arg} including lifetime value, loyalty years and complaint history.",
        "cs_triage":    "Triage the complaint in ticket {arg}. Categorise by intent, score sentiment and priority, and determine if escalation is required.",
        "cs_resolve":   "Generate a resolution strategy for ticket {arg} considering the customer LTV and loyalty. Include tone guidance and goodwill recommendation if appropriate.",
        "cs_send":      "Send the resolution response to the customer for ticket {arg} via their preferred channel.",
        "cs_log":       "Write the ticket resolution outcome for {arg} to the customer service audit trail.",
        "me_balance":   "Retrieve the trial balance and open items from the GL for period {arg} for month-end close processing.",
        "me_recon":     "Perform bank reconciliation for period {arg}. Match GL transactions against bank statements and report unmatched items.",
        "me_accruals":  "Calculate and post period accruals for {arg} to the GL in accordance with accrual policy. Report all journal entries made.",
        "me_variance":  "Run variance analysis comparing {arg} actuals against prior period and budget. Flag material variances for CFO attention.",
        "me_accounts":  "Generate the management accounts pack for {arg} for CFO and Finance Director review.",
        "me_certify":   "Write the month-end close certification record for {arg} to the finance audit trail.",
    }

    def __init__(self, model, api_key, data_dir, output_dir, verbose=False):
        self.model      = model
        self.api_key    = api_key
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose
        self.coord_hops = []
        self.agent_hops = []
        self._llm       = None

        from aacp_crewai.agent import MODEL_COSTS
        self._cpm = MODEL_COSTS.get(model, 1.0)

    def _get_llm(self):
        if not self._llm:
            from crewai import LLM
            self._llm = LLM(
                model=f"openai/{self.model}",
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=200,
            )
        return self._llm

    def _make_agent(self, role, goal, backstory):
        from crewai import Agent
        return Agent(
            role=role, goal=goal, backstory=backstory,
            llm=self._get_llm(), verbose=False, allow_delegation=False,
        )

    def _coord(self, key, arg="", extra=""):
        """Write one NL coordination message via LLM. One call per hop."""
        import re, time
        from crewai import Task, Crew
        from crewai import Agent

        prompt = self.COORD_PROMPTS[key].format(arg=arg, extra=extra)
        agent  = self._make_agent(
            "Workflow Orchestrator",
            "Write clear coordination instructions to specialist agents",
            "You coordinate multi-agent workflows by writing clear task instructions.",
        )
        task  = Task(description=prompt, expected_output="Coordination instruction", agent=agent)
        start = time.time()
        crew  = Crew(agents=[agent], tasks=[task], verbose=False)
        out   = crew.kickoff()
        latency = (time.time() - start) * 1000

        msg = out.raw
        tokens_est = len(prompt.split()) * 2
        cost_usd   = tokens_est / 1_000_000 * self._cpm

        self.coord_hops.append({
            "key": key, "message": msg[:80],
            "tokens_in": tokens_est, "cost_usd": round(cost_usd, 6),
            "latency_ms": round(latency, 0),
        })

        if self.verbose:
            print(f"  [NL coord] {key}: \"{msg[:65]}\"  ${cost_usd:.4f}")

        return msg

    def _agent_call(self, agent_obj, nl_msg, data=None):
        """Send NL message to agent and track cost."""
        r = agent_obj.receive(nl_msg, data or {})
        self.agent_hops.append(r)
        return r.get("result")

    @property
    def coordination_cost(self):
        return round(sum(h["cost_usd"] for h in self.coord_hops), 6)

    @property
    def agent_cost(self):
        return round(sum(h.get("cost_usd", 0) for h in self.agent_hops), 6)

    @property
    def total_cost(self):
        return round(self.coordination_cost + self.agent_cost, 6)

    @property
    def coordination_tokens(self):
        return sum(h["tokens_in"] for h in self.coord_hops)

    def run(self, period="2026-03"):
        import csv as _csv

        kw = {"model": self.model, "api_key": self.api_key}
        hr      = HRAgent(**kw)
        finance = FinanceAgent(**kw)
        it      = ITAgent(**kw)
        sales   = SalesAgent(**kw)
        cs      = CSAgent(**kw)
        audit   = AuditAgent()

        emp_data  = list(_csv.DictReader(open(self.data_dir / "employees_2026-03.csv")))
        bud_data  = list(_csv.DictReader(open(self.data_dir / "budgets_2026-03.csv")))
        hire_data = list(_csv.DictReader(open(self.data_dir / "new_hires_2026-03.csv")))
        lead_data = list(_csv.DictReader(open(self.data_dir / "leads_2026-03.csv")))
        tick_data = list(_csv.DictReader(open(self.data_dir / "tickets_2026-03.csv")))

        print(f"\n  --- Workflow 1/5: JML Onboarding (3 hires x 6 hops = 18) ---")
        for hire in hire_data:
            uid = hire["username"]
            msg = self._coord("jml_fetch", uid)
            self._agent_call(hr, msg, {"employee": hire})
            msg = self._coord("jml_account", uid)
            self._agent_call(it, msg, {"username": uid, "dept": hire["dept"]})
            msg = self._coord("jml_licences", uid, hire.get("licences", "M365"))
            self._agent_call(it, msg, {"username": uid})
            msg = self._coord("jml_access", uid)
            self._agent_call(it, msg, {"username": uid})
            msg = self._coord("jml_welcome", uid)
            self._agent_call(it, msg, {"username": uid})
            msg = self._coord("jml_log", uid)
            audit.receive(msg, {"username": uid})

        print(f"\n  --- Workflow 2/5: Payroll (5 hops) ---")
        msg = self._coord("pay_emp", period)
        r1  = self._agent_call(hr, msg, {"employees": emp_data})
        msg = self._coord("pay_budget", period)
        r2  = self._agent_call(finance, msg, {"budgets": bud_data})
        msg = self._coord("pay_merge", period)
        r3  = self._agent_call(hr, msg, {"employees": r1, "budgets": r2})
        msg = self._coord("pay_report", period)
        r4  = self._agent_call(hr, msg, {"payroll": r3})
        msg = self._coord("pay_log", period)
        audit.receive(msg, {"period": period})

        print(f"\n  --- Workflow 3/5: Sales Qualification (3 leads x 5 hops = 15) ---")
        for lead in lead_data:
            lid = lead["id"]
            msg = self._coord("sal_fetch", lid)
            r1  = self._agent_call(sales, msg, {"lead": lead})
            msg = self._coord("sal_score", lid)
            r2  = self._agent_call(sales, msg, {"lead": r1})
            msg = self._coord("sal_route", lid)
            r3  = self._agent_call(sales, msg, {"score": r2})
            msg = self._coord("sal_log", lid)
            audit.receive(msg, {"lead_id": lid})
            msg = self._coord("sal_notify", lid)
            self._agent_call(sales, msg, {"routing": r3})

        print(f"\n  --- Workflow 4/5: CS Resolution (3 tickets x 5 hops = 15) ---")
        for ticket in tick_data:
            tid = ticket["id"]
            msg = self._coord("cs_fetch", ticket["customer_id"])
            r1  = self._agent_call(cs, msg, {"ticket": ticket})
            msg = self._coord("cs_triage", tid)
            r2  = self._agent_call(cs, msg, {"ticket": ticket, "customer": r1})
            msg = self._coord("cs_resolve", tid)
            r3  = self._agent_call(cs, msg, {"triage": r2})
            msg = self._coord("cs_send", tid)
            self._agent_call(cs, msg, {"resolution": r3})
            msg = self._coord("cs_log", tid)
            audit.receive(msg, {"ticket_id": tid})

        print(f"\n  --- Workflow 5/5: Month-End Close (6 hops) ---")
        for key, arg in [
            ("me_balance", period), ("me_recon", period),
            ("me_accruals", period), ("me_variance", period),
            ("me_accounts", period), ("me_certify", period),
        ]:
            msg = self._coord(key, arg)
            if "certify" in key:
                audit.receive(msg, {"period": period})
            else:
                self._agent_call(finance, msg, {"period": period})


# ── AACP version ──────────────────────────────────────────────────────────

def run_aacp_department_day(model, api_key, data_dir, output_dir, period, verbose):
    kw = {"model": model, "api_key": api_key}
    agents = {
        "hr":      HRAgent(**kw),
        "finance": FinanceAgent(**kw),
        "it":      ITAgent(**kw),
        "sales":   SalesAgent(**kw),
        "cs":      CSAgent(**kw),
        "audit":   AuditAgent(),
    }
    bus = AACPPacketBus(
        workflow="department_day", model=model,
        audit_log=f"{output_dir}/audit_aacp_dept.jsonl",
        verbose=verbose,
    )
    run_department_day(bus, agents, Path(data_dir), period)
    return bus.result


# ── Comparison table ──────────────────────────────────────────────────────

def print_comparison(b, a, model):
    w = 64
    print(f"\n{'='*w}")
    print(f"  CrewAI DEPARTMENT DAY COMPARISON")
    print(f"  Model: {model}  |  Period: 2026-03")
    print(f"  5 workflows  |  59 coordination hops")
    print(f"{'='*w}")
    print(f"  {'Metric':<38} {'NO AACP':>12} {'AACP':>10}")
    print(f"  {'-'*62}")

    def row(label, bv, av, hi=False):
        mark = " <-" if hi else ""
        print(f"  {label:<38} {str(bv):>12} {str(av):>10}{mark}")

    row("Coordination approach",       "NL task desc",  "AACP packet")
    row("Coordination LLM calls",       len(b.coord_hops), 0,          hi=True)
    row("Coordination cost (USD)",
        f"${b.coordination_cost:.4f}", "$0.0000",                      hi=True)
    row("Coordination tokens (approx)", b.coordination_tokens,
        sum(max(1, len(h.packet) // 4) for h in a.hops))
    print(f"  {'-'*62}")
    row("Agent LLM calls",
        len(b.agent_hops), len(a.hops))
    row("Agent cost (USD)",
        f"${b.agent_cost:.4f}", f"${a.total_cost:.4f}")
    print(f"  {'-'*62}")
    row("Total cost (USD)",
        f"${b.total_cost:.4f}", f"${a.total_cost:.4f}",               hi=True)
    saving = b.total_cost - a.total_cost
    pct    = (saving / b.total_cost * 100) if b.total_cost > 0 else 0
    row("Total saving", "",
        f"${saving:.4f} ({pct:.0f}%)",                                 hi=True)
    print(f"  {'-'*62}")
    row("Coordination deterministic",  "NO",   "YES", hi=True)
    row("Schema validated",            "NO",   "YES", hi=True)
    row("Audit trail structured",      "NO",   "YES", hi=True)
    print(f"  {'='*w}\n")

    print(f"  SUMMARY")
    print(f"  Coordination LLM calls eliminated: {len(b.coord_hops)}")
    print(f"  Coordination cost saved:            ${b.coordination_cost:.4f}")
    print(f"  Total cost reduction:               {pct:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-4.1-mini")
    parser.add_argument("--data",    default="data")
    parser.add_argument("--mock",    action="store_true")
    parser.add_argument("--output",  default="output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first"); sys.exit(1)

    data_dir = Path(args.data)
    if args.mock:
        write_mock_data(data_dir)

    if not (data_dir / "employees_2026-03.csv").exists():
        print(f"Data files not found in {data_dir}. Run with --mock.")
        sys.exit(1)

    Path(args.output).mkdir(exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  aacp-crewai Department Day Comparison")
    print(f"  5 workflows  |  59 coordination hops each")
    print(f"  Model: {args.model}")
    print(f"{'='*64}")

    print(f"\nRUN 1: WITHOUT AACP (natural language task descriptions)")
    baseline = BaselineDepartmentDay(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        data_dir=str(data_dir),
        output_dir=args.output,
        verbose=args.verbose,
    )
    baseline.run(period="2026-03")

    print(f"\nRUN 2: WITH AACP (rule-based packet coordination, $0.00 encoding)")
    aacp_result = run_aacp_department_day(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        data_dir=str(data_dir),
        output_dir=args.output,
        period="2026-03",
        verbose=args.verbose,
    )

    print_comparison(baseline, aacp_result, args.model)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "framework":          "crewai",
        "model":              args.model,
        "workflows":          5,
        "coordination_hops":  59,
        "baseline": {
            "coordination_llm_calls": len(baseline.coord_hops),
            "coordination_cost_usd":  baseline.coordination_cost,
            "coordination_tokens":    baseline.coordination_tokens,
            "agent_cost_usd":         baseline.agent_cost,
            "total_cost_usd":         baseline.total_cost,
            "deterministic":          False,
            "validated":              False,
        },
        "with_aacp": {
            "coordination_llm_calls": 0,
            "coordination_cost_usd":  0.0,
            "agent_cost_usd":         aacp_result.total_cost,
            "total_cost_usd":         aacp_result.total_cost,
            "deterministic":          True,
            "validated":              True,
        },
    }
    p = f"{args.output}/crewai_dept_comparison_{ts}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {p}\n")


if __name__ == "__main__":
    main()
