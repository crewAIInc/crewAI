"""
HDP (Human Delegation Provenance) integration example for CrewAI.

This example demonstrates how to attach cryptographic delegation provenance
to a CrewAI crew using the hdp-crewai middleware package. Every task
executed on behalf of the original human is recorded in a tamper-evident
audit chain, verifiable offline with a single Ed25519 public key.

Related issue: https://github.com/crewAIInc/crewAI/issues/5102
Package:       pip install hdp-crewai
Spec:          https://datatracker.ietf.org/doc/draft-helixar-hdp-agentic-delegation/

Design considerations addressed:
  1. Scope enforcement    — block / log tools outside authorized_tools
  2. Delegation depth     — max_hops prevents unbounded delegation chains
  3. Token size / perf    — Ed25519 = 64 bytes/hop; fully non-blocking
  4. Verification         — verify_chain() validates offline with the public key
  5. Memory integration   — token is persisted alongside CrewAI task outputs
"""

import os

from crewai import Agent, Crew, Task
from crewai.tools import BaseTool

# pip install hdp-crewai
from hdp_crewai import HdpMiddleware, HdpPrincipal, ScopePolicy, verify_chain


# ---------------------------------------------------------------------------
# 1. Generate or load your Ed25519 signing key
#    In production: load from a secrets manager / HSM.
# ---------------------------------------------------------------------------

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

private_key = Ed25519PrivateKey.generate()
signing_key_bytes: bytes = private_key.private_bytes_raw()
public_key = private_key.public_key()


# ---------------------------------------------------------------------------
# 2. Define the HDP scope — what is the human authorising?
# ---------------------------------------------------------------------------

scope = ScopePolicy(
    intent="Analyse Q1 sales data and produce an executive summary",
    data_classification="confidential",
    network_egress=False,       # no outbound calls permitted
    persistence=False,
    authorized_tools=[          # only these tools are in-scope
        "FileReadTool",
        "CSVAnalysisTool",
        "SummaryWriterTool",
    ],
    max_hops=5,                 # at most 5 agent hops
)


# ---------------------------------------------------------------------------
# 3. Create the middleware
# ---------------------------------------------------------------------------

middleware = HdpMiddleware(
    signing_key=signing_key_bytes,
    session_id="q1-sales-review-2026",
    principal=HdpPrincipal(
        id="analyst@company.com",
        id_type="email",
        display_name="Sales Analyst",
    ),
    scope=scope,
    key_id="company-signing-key-v1",
    strict=False,   # set True to raise HDPScopeViolationError on unauthorized tools
    persist_token=True,  # write token to crewAI storage for retroactive auditing
)


# ---------------------------------------------------------------------------
# 4. Build the crew and wire the middleware
# ---------------------------------------------------------------------------

data_agent = Agent(
    role="Data Analyst",
    goal="Extract key metrics from Q1 sales data",
    backstory="Expert at reading and interpreting structured sales data.",
    verbose=True,
)

writer_agent = Agent(
    role="Report Writer",
    goal="Produce a concise executive summary from the analysed data",
    backstory="Specialises in translating data insights into clear business narratives.",
    verbose=True,
)

analysis_task = Task(
    description="Read the Q1 sales CSV and compute total revenue, top 3 products, and YoY growth.",
    expected_output="A structured dict with revenue, top_products, and yoy_growth keys.",
    agent=data_agent,
)

summary_task = Task(
    description="Write a 200-word executive summary based on the analysis results.",
    expected_output="A polished executive summary paragraph.",
    agent=writer_agent,
    context=[analysis_task],
)

crew = Crew(
    agents=[data_agent, writer_agent],
    tasks=[analysis_task, summary_task],
    verbose=True,
)

# Attach HDP — zero changes to Crew configuration required
middleware.configure(crew)


# ---------------------------------------------------------------------------
# 5. Run the crew
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = crew.kickoff(inputs={"quarter": "Q1 2026"})
    print("\n=== Crew output ===")
    print(result)

    # ---------------------------------------------------------------------------
    # 6. Design consideration #4: verify the delegation chain offline
    # ---------------------------------------------------------------------------

    token = middleware.export_token()
    verification = verify_chain(token, public_key)

    print("\n=== HDP Audit Trail ===")
    print(f"Valid:     {verification.valid}")
    print(f"Token ID:  {verification.token_id}")
    print(f"Hops:      {verification.hop_count}")
    if verification.violations:
        print(f"Violations: {verification.violations}")

    for hop in verification.hop_results:
        status = "✓" if hop.valid else "✗"
        print(f"  [{status}] hop {hop.seq} — {hop.agent_id}")

    # Design consideration #5: full token JSON also saved to crewAI storage
    print("\n=== Full token (for audit log) ===")
    print(middleware.export_token_json())
