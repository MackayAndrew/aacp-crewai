# aacp-crewai

**AACP coordination layer for CrewAI multi-agent workflows.**

CrewAI maps more naturally to AACP than any other framework because
agent roles map directly to AACP domains and tasks map to AACP task types.

## Install

```bash
pip install aacp-crewai
```

## Quick start

```python
from aacp_crewai.crew import AACPCrew

crew = AACPCrew(model="gpt-4.1-mini")
result = crew.run_workflow("payroll", period="2026-03")
print(result.summary())
```

## Measured results

Payroll workflow comparison. gpt-4.1-mini. June 2026.

```
                        WITHOUT AACP    WITH AACP
Coordination LLM calls:        4            0
Coordination cost:          $0.0005      $0.0000
Total cost:                 $0.0005      $0.0003
Total saving:                             39%
Coordination deterministic:    NO          YES
Schema validated:              NO          YES
```

## The natural fit

```
CrewAI concept     AACP concept
──────────────     ────────────
Agent role         DOM  (HR, FIN, IT, SALES, CS)
Agent goal         Workflow objective
Task description   AACP packet content
Crew kickoff       Orchestrator run
```

## Without AACP vs With AACP

```
Without aacp-crewai (standard CrewAI):
  Orchestrator
    ↓ "Retrieve all active employee salary records for March 2026.
       Include employee ID, name, department, cost centre, base salary,
       any changes this month, and pension rate. Return as JSON."
  HR Agent  ← verbose, varies every run

With aacp-crewai:
  Orchestrator
    ↓ FETCH|HR|return:ORCHESTRATOR|p:1|aacp:1.1|res:emp_salary|period:2026-03
  HR Agent  ✓ validates. $0.00 encoding. Identical every run.
```

## Comparison demo

```bash
python3 examples/comparison.py --mock
```

## Workflows

```python
# Payroll (5 hops)
crew.run_workflow("payroll", period="2026-03")

# IT provisioning / JML (6 hops)
crew.run_workflow("it_provisioning", username="j.smith", dept="Engineering")

# Sales qualification (5 hops)
crew.run_workflow("sales_qualification", lead_id="L-001")
```

## Requirements

- Python 3.10+
- `OPENAI_API_KEY`
- `pip install aacp-crewai`

## Links

- Protocol spec: https://aacp.dev
- Python SDK: https://github.com/MackayAndrew/aacp
- LangChain integration: https://github.com/MackayAndrew/aacp-langchain
- Community rules (241): https://github.com/MackayAndrew/aacp-community-rules
- IETF Draft: https://datatracker.ietf.org/doc/draft-mackay-aacp/

## Licence

MIT
