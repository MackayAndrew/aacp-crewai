"""aacp-crewai -- AACP coordination layer for CrewAI."""

from .packet_bus        import AACPPacketBus, WorkflowResult
from .agent             import AuditAgent
from .agents            import HRAgent, FinanceAgent, ITAgent, SalesAgent, CSAgent

__version__ = "0.1.0"
__all__ = ["AACPPacketBus", "WorkflowResult", "AuditAgent",
           "HRAgent", "FinanceAgent", "ITAgent", "SalesAgent", "CSAgent"]
