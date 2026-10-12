# Jev Decision Tool

Use `JevDecisionTool` to ask one TypeSafe Jev Choice question about task state
and return the selected label, probability distribution, and confidence.

Install the optional dependency before using the tool:

```bash
uv add --optional typesafe-sdk typesafe-sdk
```

Set `TYPESAFE_API_KEY`, then opt in from a task or direct tool call:

```python
from crewai_tools import JevDecisionTool

tool = JevDecisionTool()

result = tool.run(
    state="The customer was charged twice and wants a refund.",
    question="Which team should handle this request?",
    choices={
        "billing": "Payment, invoice, subscription, or refund issues",
        "technical": "Product bugs or integration problems",
        "support": "General account or usage help",
    },
)

if result["success"]:
    print(result["choice"])
    print(result["confidence"])
else:
    print(result["error"])
```
