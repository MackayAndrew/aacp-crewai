from ..agent import AACPCrewAgent

class CSAgent(AACPCrewAgent):
    def __init__(self, model="gpt-4.1-mini", api_key=None):
        super().__init__(
            role="Customer Service Resolution Specialist",
            goal="Triage complaints, determine resolution strategies, manage goodwill",
            backstory="""Senior CS specialist with expertise in complaint resolution,
LTV assessment and retention strategy.
Balances customer empathy with commercial judgment when making goodwill decisions.""",
            domain="CS", model=model, api_key=api_key,
        )
