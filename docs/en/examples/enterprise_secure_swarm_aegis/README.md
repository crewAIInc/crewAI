# Enterprise Secure Swarm (Powered by Aegis AIP)

As CrewAI swarms scale into enterprise production, relying purely on an LLM's internal "safety alignment" becomes a massive compliance liability. Chief Risk Officers (CROs) will not approve deployments if autonomous agents possess "God-mode" access to production databases or financial APIs.

This example demonstrates how to seamlessly secure a CrewAI Swarm using **Aegis**—a Zero-Trust network proxy and Identity Access Management (IAM) layer for non-human workers.

## The Architecture: "Thin Client, Fat Server"

By wrapping your CrewAI `BaseTool` with the `aegis-aip` open-source SDK, you achieve **Defense in Depth**:
1. **No Hardcoded API Keys:** Agents authenticate dynamically, receiving short-lived, cryptographically signed Ed25519 tokens (IBCTs).
2. **Mathematical Bounding:** If an agent hallucinates or suffers a prompt injection, the Aegis cloud proxy evaluates the token's parameters and drops the execution mathematically at the network layer.
3. **Immutable Audit Logs:** Every intercepted execution is logged directly to a real-time SIEM dashboard for compliance auditing.

## Supported Enterprise Boundaries
Aegis protects against the four major AI threat vectors out of the box:
* **Financial Bounds:** Caps transaction amounts (e.g., maximum $500 refund).
* **Data Exfiltration Bounds:** Restricts database queries to authorized tenant IDs (e.g., Row Level Security enforcement).
* **Destructive Action Bounds:** Blocks unrecoverable commands (e.g., `DROP TABLE`, `DELETE /users`).
* **Communication Bounds:** Limits outbound messaging to whitelisted domains.

## Quickstart (Under 5 Minutes)

### 1. Install Dependencies
```bash
pip install crewai langchain-openai aegis-aip
```

### 2. Set Environment Variables
The Aegis Client requires zero complex configuration. Simply provide your target control plane and the identity of the agent.
```bash
export OPENAI_API_KEY="your-openai-key"
export AEGIS_AGENT_ID="your_enterprise_user_id"
export AEGIS_CONTROL_PLANE_URL="https://aegis-live-node.onrender.com" # Or your self-hosted Aegis instance
```

### 3. Run the Live-Fire Test
In this live-fire example, we intentionally subject the `financial_agent` to a malicious prompt injection, demanding a $50,000 refund. 

Run the script to watch the Aegis Proxy intercept and mathematically drop the hallucinated execution in ~12ms, without breaking the CrewAI orchestration loop.

```bash
python aegis_crewai_example.py
```

**Expected Terminal Output:**
```text
--- INITIATING SECURE CREWAI SWARM ---
[AEGIS BLOCKED] Execution intercepted: Hallucination blocked: $50000 exceeds maximum limit of $500
--- EXECUTION RESULT ---
Task failed due to security interception.
```

To learn more about the Agentic Identity Protocol (AIP), visit the [Aegis GitHub Repository](https://github.com/Yash-0620/aegis-control-plane).