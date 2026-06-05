"""Tests for AACPCrew packet bus -- no API calls needed."""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aacp_crewai.packet_bus import AACPPacketBus
from aacp_crewai.agent      import AuditAgent

passed = failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {label}"); passed += 1
    else:
        print(f"  ✗ FAIL: {label}"); failed += 1

class MockAgent:
    name = "MOCK-AGENT"
    def receive(self, packet, data=None):
        return {"result": {"ok": True}, "tokens_in": 50,
                "tokens_out": 20, "latency_ms": 100,
                "cost_usd": 0.00003, "error": None}

print("\n" + "="*50)
print("  aacp-crewai PacketBus Tests")
print("="*50)

tmpdir = tempfile.mkdtemp()
try:
    bus = AACPPacketBus("test","gpt-4.1-mini",
                        audit_log=f"{tmpdir}/audit.jsonl", verbose=False)
    agent = MockAgent()

    r = bus.dispatch("ORCHESTRATOR", agent,
        "FETCH|HR|return:HR-Agent|p:1|aacp:1.1|res:emp_salary|period:2026-03")
    check("valid packet dispatches", r is not None)
    check("hop recorded",            len(bus.result.hops) == 1)
    check("hop is valid",            bus.result.hops[0].valid)
    check("cost accumulated",        bus.result.total_cost > 0)

    audit = AuditAgent()
    bus2  = AACPPacketBus("test2","gpt-4.1-mini",
                          audit_log=f"{tmpdir}/audit2.jsonl", verbose=False)
    r2 = bus2.dispatch("ORCHESTRATOR", audit,
        "LOG|HR|return:AUD-Agent|p:2|aacp:1.1|status:complete")
    check("audit agent result",   r2 is not None)
    check("audit agent $0.00",    bus2.result.total_cost == 0.0)

    print(f"\n  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  PacketBus tests passed. No API calls needed.\n")
finally:
    shutil.rmtree(tmpdir)
