# OCI Generative AI Agent Tool

This tool lets CrewAI agents invoke an Oracle Cloud Infrastructure Generative AI agent endpoint.

## Installation

```bash
uv pip install 'crewai-tools[oci]'
```

## Usage

```python
from crewai_tools import OCIGenAIInvokeAgentTool

agent_tool = OCIGenAIInvokeAgentTool(
    agent_endpoint_id="ocid1.genaiagentendpoint.oc1..exampleuniqueID"
)
```
