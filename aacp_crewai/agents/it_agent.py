from ..agent import AACPCrewAgent

class ITAgent(AACPCrewAgent):
    def __init__(self, model="gpt-4.1-mini", api_key=None):
        super().__init__(
            role="IT Systems Administrator and Identity Access Manager",
            goal="Provision and deprovision user accounts, manage licences and access",
            backstory="""Senior IT administrator specialising in Active Directory,
Entra ID, licence management and identity access management.
Follows principle of least privilege and documents all provisioning actions.""",
            domain="IT", model=model, api_key=api_key,
        )
