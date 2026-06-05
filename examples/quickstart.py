"""
aacp-crewai quickstart

Run:
    export OPENAI_API_KEY=sk-...
    python3 examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aacp_crewai.crew import AACPCrew


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first")
        sys.exit(1)

    print("\naacp-crewai quickstart")
    print("=" * 50)
    print("CrewAI agent roles map directly to AACP domains.")
    print("AACP packets coordinate the crew. $0.00 encoding.\n")

    crew = AACPCrew(model="gpt-4.1-mini")
    result = crew.run_workflow(
        "it_provisioning",
        username="j.smith",
        dept="Engineering",
        licences=["M365", "Slack"],
    )

    print("\n" + result.summary())
    print(f"\nEncoding cost: $0.00 (rule-based encoder)")
    print(f"Agent cost:    ${result.total_cost:.4f}")


if __name__ == "__main__":
    main()
