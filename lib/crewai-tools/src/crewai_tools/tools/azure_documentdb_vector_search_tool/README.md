# Azure DocumentDB Vector Search Tool

`AzureDocumentDBVectorSearchTool` stores and retrieves text documents using
[Azure DocumentDB's native vector search](https://learn.microsoft.com/en-us/azure/documentdb/vector-search).
It is separate from `MongoDBVectorSearchTool`: it uses DocumentDB's
`$search.cosmosSearch` protocol, not MongoDB Atlas `$vectorSearch`.
This integration is not for Amazon DocumentDB or Azure Cosmos DB for NoSQL.

## Installation and resource setup

```shell
pip install "crewai-tools[azure-documentdb]"
```

Create an [Azure DocumentDB cluster](https://learn.microsoft.com/en-us/azure/documentdb/quickstart-python-vector-search)
with vector search available on your tier. Allow your application's network
through the cluster firewall, create a database user with read/write and index
creation permissions, and obtain the MongoDB-compatible connection string from
the Azure portal. Use the supplied TLS and authentication parameters; do not
disable certificate validation for Azure connections.

Create your database and collection before creating an index. Index creation
is explicit, not performed at tool initialization or on every query. The helper
creates an IVF index; for HNSW or DiskANN, provision an index on `embedding`
following the Azure documentation and your tier's requirements.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `AZURE_DOCUMENTDB_CONNECTION_STRING` | Database connection string; alternatively pass `connection_string` to the constructor. |
| `OPENAI_API_KEY` | OpenAI embedding API key, unless using Azure OpenAI. |
| `AZURE_OPENAI_ENDPOINT` | Set to select Azure OpenAI instead of OpenAI. |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI embedding API key, unless using Entra ID. |
| `OPENAI_API_VERSION` | API version supported by your Azure OpenAI resource. |

Database authentication uses the connection string, with a renewable Entra ID
token callback when it specifies `authMechanism=MONGODB-OIDC`. Database and
embedding access require separate permissions, even when using the same identity.
Keep credentials in environment variables or a secret manager, never in source
control.

`embedding_model` defaults to `text-embedding-3-small` and `dimensions` to 1536.
For Azure OpenAI, pass your **embedding deployment name** as `embedding_model`.
Use a text-embedding-3-compatible model/deployment supporting the `dimensions`
parameter. The same model and dimensions must be used for ingestion and queries,
and the index dimensions must match. Changing models requires re-embedding data.

## Keyless authentication with Microsoft Entra ID

Enable [Entra authentication on DocumentDB](https://learn.microsoft.com/en-us/azure/documentdb/how-to-connect-role-based-access-control)
and register your identity with database permissions to create collections and
indexes, read/write documents, and drop test collections. Azure subscription
Contributor alone does not grant database access. For Azure OpenAI, assign
**Cognitive Services OpenAI User** or another role allowing embedding inference.
Both service firewalls must allow your client to connect.

For local development, use your existing Azure CLI login:

```powershell
az login --tenant "<tenant-id>"
$env:AZURE_DOCUMENTDB_CONNECTION_STRING = "mongodb+srv://<cluster>.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=MONGODB-OIDC&retrywrites=false&maxIdleTimeMS=120000"
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
$env:OPENAI_API_VERSION = "2023-05-15"
```

Pass the resource's base endpoint, not the full embeddings URL. Select the
embedding deployment with `embedding_model`:

```python
import os
from azure.identity import AzureCliCredential
from crewai_tools import AzureDocumentDBVectorSearchTool

with AzureCliCredential(tenant_id=os.environ["AZURE_TENANT_ID"]) as credential:
    tool = AzureDocumentDBVectorSearchTool(
        database_name="knowledge",
        collection_name="documents",
        embedding_model="text-embedding-3-small",
        azure_credential=credential,
    )
    try:
        results = tool.run(query="How do AI agents collaborate?")
    finally:
        tool.close()
```

Set `AZURE_TENANT_ID` to the tenant used for `az login`. No client secret,
database password, or OpenAI API key is needed for this flow.

The injected `azure_credential` is used for DocumentDB OIDC and Azure OpenAI
bearer tokens. Tokens are reacquired as needed, not stored as static passwords.
The caller owns the injected credential and must close it after the tool.
For production on Azure, inject `ManagedIdentityCredential`; for a service
principal, inject `EnvironmentCredential` with `AZURE_TENANT_ID`,
`AZURE_CLIENT_ID`, and its secret/certificate or other supported credential.
An explicit credential avoids falling back to a different identity.

If no credential is passed, an OIDC connection string selects
`DefaultAzureCredential` automatically, and that credential is also used for
Azure OpenAI. For a password-authenticated database with keyless Azure OpenAI,
set `use_azure_identity=True`. The tool closes credentials it creates internally.
The existing OpenAI/Azure OpenAI API-key path remains available when Entra is not
selected.

## Ingestion and CrewAI Agent usage

```python
import json
import os

from crewai import Agent
from crewai_tools import (
    AzureDocumentDBVectorSearchConfig,
    AzureDocumentDBVectorSearchTool,
)

tool = AzureDocumentDBVectorSearchTool(
    # Omit this argument to read AZURE_DOCUMENTDB_CONNECTION_STRING automatically.
    connection_string=os.environ["AZURE_DOCUMENTDB_CONNECTION_STRING"],
    database_name="knowledge",
    collection_name="documents",
    embedding_model="text-embedding-3-small",  # Azure: use your deployment name.
    dimensions=1536,
    query_config=AzureDocumentDBVectorSearchConfig(limit=4),
)

try:
    # The collection must exist; create it once using your provisioning process.
    # On a new collection, ingest before building the vector index.
    ids = tool.add_texts(
        ["CrewAI orchestrates collaborative AI agents.", "DocumentDB supports vector search."],
        metadatas=[{"category": "agents"}, {"category": "database"}],
        ids=["crewai-intro", "documentdb-intro"],  # Optional; otherwise UUIDs are generated.
        batch_size=100,
    )
    tool.create_vector_search_index(similarity="COS", num_lists=1)
    print(json.loads(tool.run(query="How do I coordinate AI agents?")))

    agent = Agent(
        role="Knowledge researcher",
        goal="Find relevant information in our DocumentDB knowledge base",
        backstory="You ground your answers in retrieved documents.",
        tools=[tool],
        verbose=True,
    )
    # Run your Crew with this agent before closing the tool.
finally:
    tool.close()
```

`add_texts` accepts an iterable of strings, including generators. It stores
`{ "_id": "...", "text": "...", "metadata": {...}, "embedding": [...] }`.
IDs and metadata must match the text count; duplicate IDs raise write errors
(insertion is not an upsert). Empty input returns an empty ID list. Each batch
gets its own embedding request and insert; earlier batches persist if a later
batch fails. Database and embedding errors propagate to the caller.

The index helper sends this native command (with your collection and dimensions):

```json
{
  "createIndexes": "documents",
  "indexes": [{
    "name": "crewai_vector_index",
    "key": {"embedding": "cosmosSearch"},
    "cosmosSearchOptions": {
      "kind": "vector-ivf",
      "numLists": 1,
      "similarity": "COS",
      "dimensions": 1536
    }
  }]
}
```

Supported similarity values are `COS`, `L2`, and `IP`. Tune `num_lists` for your
data size according to Azure's guidance. The helper does not drop or rebuild
existing indexes; conflicting definitions raise database errors.

## Query options and output

```python
config = AzureDocumentDBVectorSearchConfig(
    limit=5,
    pre_filter={"metadata.category": "agents"},
    include_score=True,
    include_embeddings=False,
)
```

Pass this as `query_config`. Filters run **during** vector search, not after
limiting results. Provision scalar indexes for filter fields as required by your
cluster, for example `collection.create_index("metadata.category")`.
`limit` must be positive; score output defaults to enabled and embeddings to
excluded. The tool returns a JSON array string containing the stored document
fields and optional `score`, or `[]` when there are no matches.
BSON-only values such as ObjectId and dates use MongoDB Extended JSON
(`{"$oid": "..."}`, `{"$date": "..."}`), so `json.loads` always works.
Consumers should not assume scores have the same scale across similarity metrics.

## Podman Desktop end-to-end tests

Start the Podman Desktop engine, then run the official open-source DocumentDB
local image. These PowerShell commands bind the database to loopback only:

```powershell
podman pull ghcr.io/documentdb/documentdb/documentdb-local:latest
podman run -d --name crewai-documentdb-e2e -p 127.0.0.1:10260:10260 ghcr.io/documentdb/documentdb/documentdb-local:latest --username crewai_e2e --password local_e2e_only
$env:AZURE_DOCUMENTDB_E2E_CONNECTION_STRING = "mongodb://crewai_e2e:local_e2e_only@127.0.0.1:10260/?tls=true&tlsAllowInvalidCertificates=true&directConnection=true"
uv run --package crewai-tools --extra azure-documentdb pytest lib\crewai-tools\tests\tools\test_azure_documentdb_vector_search_e2e.py -o addopts="" -n 0
podman rm -f crewai-documentdb-e2e
Remove-Item Env:\AZURE_DOCUMENTDB_E2E_CONNECTION_STRING
```

The self-signed certificate exception is **only for the local test container**.
If Windows refuses the connection while the container is healthy, WSL may not
be forwarding its loopback port. In a separate PowerShell terminal, keep an SSH
tunnel to the Podman machine running:

```powershell
$machine = (podman machine inspect podman-machine-default | ConvertFrom-Json)[0]
ssh -i $machine.SSHConfig.IdentityPath -p $machine.SSHConfig.Port -o ExitOnForwardFailure=yes -N -L 127.0.0.1:10261:127.0.0.1:10260 "$($machine.SSHConfig.RemoteUsername)@127.0.0.1"
```

Use port `10261` instead of `10260` in the test connection string in the first
terminal, and stop the tunnel with Ctrl+C after testing.

The tests use a local HTTP embedding fixture through the real OpenAI SDK, so no
cloud credentials or paid embedding calls are needed. They verify real ingestion,
IVF index creation, similarity ordering, filters, limits, score/embedding output,
and duplicate-ID errors. Each test creates and drops its own collection.
Without the E2E connection variable these tests are skipped. This validates the
DocumentDB wire protocol locally, not Azure-managed firewall, TLS, tier support,
or a live Azure OpenAI deployment.

Run the unit tests separately:

```powershell
uv run --package crewai-tools --extra azure-documentdb pytest lib\crewai-tools\tests\tools\test_azure_documentdb_vector_search_tool.py
```

## Live Azure end-to-end test

This opt-in test uses actual Azure OpenAI embeddings and a unique DocumentDB
collection. It incurs embedding charges and requires permission to create and
delete that collection. It never drops a pre-existing collection or the database.

```powershell
$env:AZURE_DOCUMENTDB_LIVE_CONNECTION_STRING = $env:AZURE_DOCUMENTDB_CONNECTION_STRING
$env:AZURE_DOCUMENTDB_LIVE_OPENAI_ENDPOINT = $env:AZURE_OPENAI_ENDPOINT
$env:AZURE_DOCUMENTDB_LIVE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
$env:AZURE_DOCUMENTDB_LIVE_OPENAI_API_VERSION = "2023-05-15"
$env:AZURE_DOCUMENTDB_LIVE_IDENTITY = "azure_cli"
$env:AZURE_DOCUMENTDB_LIVE_TENANT_ID = "<tenant-id>"
$env:AZURE_DOCUMENTDB_LIVE_DATABASE = "crewai_e2e"
uv run --package crewai-tools --extra azure-documentdb pytest lib\crewai-tools\tests\tools\test_azure_documentdb_vector_search_e2e.py -k live_azure -o addopts="" -n 0
```

Alternatively set `AZURE_DOCUMENTDB_LIVE_IDENTITY=environment` (the default) to
use a service principal through `EnvironmentCredential`. CLI and environment
modes never fall back to each other. API-key environment settings are cleared
within the test, ensuring both services authenticate with Entra tokens. The test
verifies ingestion, indexing, similarity scores, filters, output options, and
collection cleanup. Without the live connection variable it is skipped.
