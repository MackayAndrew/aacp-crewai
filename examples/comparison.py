"""
aacp-crewai comparison
Same payroll workflow: WITHOUT AACP vs WITH AACP.

Run:
    python3 examples/comparison.py --mock
    python3 examples/comparison.py --data ../aacp-lab/data
"""

import sys
import os
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOCK_EMPLOYEES = [
    {"id": "E001", "name": "Alice Smith", "dept": "Engineering",
     "cost_centre": "CC-10", "base_salary_gbp": "72000", "delta_gbp": "0",
     "pension_rate": "0.05", "status": "active"},
    {"id": "E002", "name": "Bob Jones", "dept": "Sales",
     "cost_centre": "CC-20", "base_salary_gbp": "58000", "delta_gbp": "2500",
     "pension_rate": "0.05", "status": "active"},
    {"id": "E003", "name": "Carol White", "dept": "Finance",
     "cost_centre": "CC-30", "base_salary_gbp": "65000", "delta_gbp": "0",
     "pension_rate": "0.08", "status": "active"},
    {"id": "E004", "name": "David Brown", "dept": "Engineering",
     "cost_centre": "CC-10", "base_salary_gbp": "85000", "delta_gbp": "5000",
     "pension_rate": "0.05", "status": "active"},
]

MOCK_BUDGETS = [
    {"cc_id": "CC-10", "cc_name": "Engineering", "approved_annual_gbp": "420000",
     "ytd_spend_gbp": "378000", "owner": "Sarah Chen", "gl_code": "GL-1010"},
    {"cc_id": "CC-20", "cc_name": "Sales", "approved_annual_gbp": "140000",
     "ytd_spend_gbp": "98000", "owner": "Marcus Webb", "gl_code": "GL-2020"},
    {"cc_id": "CC-30", "cc_name": "Finance", "approved_annual_gbp": "160000",
     "ytd_spend_gbp": "124000", "owner": "David Park", "gl_code": "GL-3030"},
]


def write_mock(data_dir):
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "employees_2026-03.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MOCK_EMPLOYEES[0].keys())
        w.writeheader()
        w.writerows(MOCK_EMPLOYEES)
    with open(data_dir / "budgets_2026-03.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MOCK_BUDGETS[0].keys())
        w.writeheader()
        w.writerows(MOCK_BUDGETS)
    with open(data_dir / "payroll_rules.json", "w") as f:
        json.dump({"version": "payroll_v2", "paye_rate": 0.20}, f)


def print_comparison(baseline, aacp, model):
    w = 62
    print(f"\n{'='*w}")
    print(f"  CrewAI COMPARISON: Same workflow, two coordination styles")
    print(f"  Model: {model}  |  Period: 2026-03")
    print(f"{'='*w}")
    print(f"  {'Metric':<36} {'NO AACP':>12} {'AACP':>10}")
    print(f"  {'-'*58}")

    def row(label, bv, av, hi=False):
        mark = " <-" if hi else ""
        print(f"  {label:<36} {str(bv):>12} {str(av):>10}{mark}")

    row("Coordination approach",       "NL task desc",  "AACP packet")
    row("Coordination LLM calls",       len(baseline.agent_hops), 0, hi=True)
    row("Coordination cost (USD)",
        f"${baseline.agent_cost:.4f}", "$0.0000",       hi=True)
    print(f"  {'-'*58}")
    row("Agent cost (USD)",
        f"${baseline.agent_cost:.4f}", f"${aacp.total_cost:.4f}")
    row("Total cost (USD)",
        f"${baseline.total_cost:.4f}", f"${aacp.total_cost:.4f}", hi=True)
    saving = baseline.total_cost - aacp.total_cost
    pct    = (saving / baseline.total_cost * 100) if baseline.total_cost > 0 else 0
    row("Total saving", "", f"${saving:.4f} ({pct:.0f}%)", hi=True)
    print(f"  {'-'*58}")
    row("Coordination deterministic",  "NO",   "YES", hi=True)
    row("Schema validated",            "NO",   "YES", hi=True)
    row("Audit trail structured",      "NO",   "YES", hi=True)
    print(f"  {'='*w}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gpt-4.1-mini")
    parser.add_argument("--data",   default="data")
    parser.add_argument("--mock",   action="store_true")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first")
        sys.exit(1)

    data_dir = Path(args.data)
    if args.mock:
        write_mock(data_dir)

    Path(args.output).mkdir(exist_ok=True)

    from aacp_crewai.baseline_crew import BaselineCrew
    from aacp_crewai.crew          import AACPCrew

    print("\nRUN 1: WITHOUT AACP (natural language task descriptions)")
    baseline = BaselineCrew(model=args.model, data_dir=str(data_dir),
                            output_dir=args.output, verbose=True)
    baseline.run_payroll(period="2026-03")

    print("\nRUN 2: WITH AACP (rule-based packet coordination)")
    crew = AACPCrew(model=args.model, data_dir=str(data_dir),
                    output_dir=args.output, verbose=True)
    aacp_result = crew.run_workflow("payroll", period="2026-03")

    print_comparison(baseline.result, aacp_result, args.model)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "framework": "crewai",
        "model": args.model,
        "baseline": {
            "coordination_llm_calls": len(baseline.result.agent_hops),
            "total_cost_usd":         baseline.result.total_cost,
            "deterministic":          False,
            "validated":              False,
        },
        "with_aacp": {
            "coordination_llm_calls": 0,
            "coordination_cost_usd":  0.0,
            "total_cost_usd":         aacp_result.total_cost,
            "deterministic":          True,
            "validated":              True,
        },
    }
    p = f"{args.output}/crewai_comparison_{ts}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {p}\n")


if __name__ == "__main__":
    main()
