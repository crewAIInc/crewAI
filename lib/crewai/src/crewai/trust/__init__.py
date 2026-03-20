"""Trust verification integrations for CrewAI agents.

This module provides optional integration with the Joy trust network
for verifying agent trustworthiness before delegation.

Example:
    from crewai.trust import JoyVerifier

    verifier = JoyVerifier()
    result = await verifier.verify_agent("ag_xxx")

    if result.is_trusted:
        # Safe to delegate
        pass
"""

from crewai.trust.joy_verifier import JoyVerifier, VerificationResult, TrustVerificationError

__all__ = ["JoyVerifier", "VerificationResult", "TrustVerificationError"]
