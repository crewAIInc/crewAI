from ube_foundation import TrustEngine, PqcKeyPair

class NTIGuardrailMiddleware:
    """
    Neutral Trust Infrastructure (NTI) Guardrail for CrewAI agents.
    Enforces post-quantum zero-trust capability bounds and cryptographic signatures.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.engine = TrustEngine()
        self.pqc_key = PqcKeyPair.generate()

    def grant_capability(self, capability: str):
        self.engine.grant(self.agent_id, capability)

    def verify_tool_execution(self, tool_name: str, tool_input: dict) -> bool:
        import json
        req = {
            "id": f"req-{abs(hash(str(tool_input)))}",
            "actor": self.agent_id,
            "capability": tool_name,
            "action": tool_name,
            "input": tool_input,
            "signature": None,
            "pqc_signature": None,
            "public_key": None,
            "pqc_public_key": None,
            "token": None,
            "identity_claim": None
        }
        message = json.dumps(req, sort_keys=True).encode('utf-8')
        req["pqc_signature"] = self.pqc_key.sign(message)
        req["pqc_public_key"] = self.pqc_key.public_key_hex()
        
        decision = json.loads(self.engine.evaluate(json.dumps(req)))
        return decision.get("decision") == "Allow"
