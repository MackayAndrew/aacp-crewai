from ..agent import AACPCrewAgent

class FinanceAgent(AACPCrewAgent):
    def __init__(self, model="gpt-4.1-mini", api_key=None):
        super().__init__(
            role="Finance Controller and Management Accountant",
            goal="Process financial data, manage budgets, reconcile accounts",
            backstory="""Experienced finance controller specialising in management accounting,
cost centre budgets, bank reconciliation and month-end close.
Always pre-computes numeric values and never writes arithmetic expressions in JSON.""",
            domain="FIN", model=model, api_key=api_key,
        )
