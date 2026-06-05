from ..agent import AACPCrewAgent

class SalesAgent(AACPCrewAgent):
    def __init__(self, model="gpt-4.1-mini", api_key=None):
        super().__init__(
            role="Sales Operations and CRM Analyst",
            goal="Qualify leads, route opportunities, track pipeline",
            backstory="""Sales operations specialist with expertise in BANT scoring,
CRM management and pipeline analysis.
Makes data-driven qualification decisions based on budget, authority, need and timeline.""",
            domain="SALES", model=model, api_key=api_key,
        )
