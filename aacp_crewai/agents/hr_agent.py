from ..agent import AACPCrewAgent

class HRAgent(AACPCrewAgent):
    def __init__(self, model="gpt-4.1-mini", api_key=None):
        super().__init__(
            role="HR Payroll and People Operations Specialist",
            goal="Process employee data, calculate payroll, manage onboarding",
            backstory="""Senior HR specialist with deep expertise in UK payroll,
PAYE calculations, pension contributions and people operations.
Process structured employee data and return precise payroll calculations.""",
            domain="HR", model=model, api_key=api_key,
        )
