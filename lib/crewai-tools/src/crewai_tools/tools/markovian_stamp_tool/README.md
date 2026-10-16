# MarkovianStampTool

The **MarkovianStampTool** creates a verifiable provenance receipt for any text
using the [Markovian Protocol](https://markovianprotocol.com). It is useful when
an agent needs to prove that an output, decision, or document existed at a point
in time.

The tool computes a SHA-256 hash of the text locally and sends only that hash
(and the optional label) to the Markovian API. The text itself is never sent.
The hash is recorded in the public witnessed transparency log, and the tool
returns the hash, a Merkle root, the log index when the API reports one, and a
public verify URL. Anyone
holding the original text can recompute its SHA-256 and compare it to the
receipt, and anyone can open the verify URL with no account.

Markovian proves that data existed, not that it is correct.

---

## Description

This tool:

* Accepts any **text**, hashes it locally, and stamps the hash.
* Returns the **data hash**, **Merkle root**, and a **public verify URL**, plus the
  **log index** when the API reports one.
* Requires **no account, wallet, or API key**.
* Uses only `requests`, which is already a dependency of `crewai-tools`.

---

## Arguments

| Argument | Type  | Required | Description                                                      |
| -------- | ----- | -------- | ---------------------------------------------------------------- |
| `data`   | `str` | Yes      | The content to stamp (an agent output, decision, or document).   |
| `label`  | `str` | No       | Optional human-readable label, sent in plain text with the hash. |

Constructor options: `base_url`, `timeout`.

---

## Usage

```python
from crewai_tools import MarkovianStampTool

tool = MarkovianStampTool()
receipt = tool.run(data="The market thesis approved by the agent at 15:00 UTC.")
print(receipt)
```

Example output:

```text
Markovian provenance receipt:
  data_hash: 3a28e136d037a943936213fa790b8eb50f112940be8717eefeeea56e5b8316fa (sha256 of the stamped text)
  merkle_root: 11234ad23f343bd99b8ee173016e72a77eda97d432bc336f8187529d8c757b1e
  log_index: 8330
  verify_url: https://api.markovianprotocol.com/verify/11234ad23f343bd99b8ee173016e72a77eda97d432bc336f8187529d8c757b1e
```

Give an agent a one-line way to stamp its final answer:

```python
from crewai import Agent
from crewai_tools import MarkovianStampTool

agent = Agent(
    role="Analyst",
    goal="Produce analysis and stamp it for provenance.",
    backstory="Stamps every deliverable so it can be independently verified.",
    tools=[MarkovianStampTool()],
)
```
