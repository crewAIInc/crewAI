# OutageDeck Status Tool

`OutageDeckStatusTool` lets a CrewAI agent check current infrastructure-provider status, incidents, and individual service health without an API key.

```python
from crewai_tools import OutageDeckStatusTool

tool = OutageDeckStatusTool()

provider = tool.run(operation="provider_status", slug="github")
active_incidents = tool.run(
    operation="list_incidents",
    provider="github",
    state="active",
    limit=10,
)
service = tool.run(operation="service_status", slug="github-actions")
```

The tool returns JSON with a stable `success` flag. Successful results include normalized status or incident data and a live OutageDeck URL. Failures return `{"success":false,"error":"..."}`.

Supported operations:

| Operation | Required input | Optional input |
| --- | --- | --- |
| `provider_status` | `slug` | — |
| `list_incidents` | — | `provider`, `state`, `severity`, `page`, `limit` |
| `service_status` | `slug` | — |

Provider and service slugs use lowercase letters, numbers, and single hyphens. Incident state is `active` or `resolved`; severity is `minor`, `major`, `critical`, or `maintenance`.

The API is read-only for these operations and the tool fixes requests to OutageDeck's production HTTPS origin. See the [OutageDeck API documentation](https://outagedeck.com/developers/api?utm_source=crewai&utm_medium=integration&utm_campaign=crewai_tool) for the underlying response contract and anonymous rate limit.
