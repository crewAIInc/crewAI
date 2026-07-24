# MarkovianStampTool

The **MarkovianStampTool** creates a verifiable provenance receipt for any text
using the [Markovian Protocol](https://markovianprotocol.com). It is useful when
an agent needs to prove that an output, decision, or document existed at a point
in time.

Stamping commits a hash of the data to the chain, anchored to Bitcoin, and
returns a Merkle root plus a public verify URL. Anyone can confirm the record at
`https://api.quantsynth.net/verify/<merkle_root>` with no account.

Markovian proves that data existed, not that it is correct. Provenance, not
truth.

---

## Description

This tool:

* Accepts any **text** and stamps it on the Markovian Protocol.
* Returns a **Merkle root**, **block height**, and a **public verify URL**.
* Requires **no account, wallet, or API key** for the free tier.
* Accepts an optional **API key** for attributed or pro usage.

It prefers the [`markovian`](https://pypi.org/project/markovian/) SDK and falls
back to a plain HTTP POST when the SDK is not installed.

---

## Installation

```bash
pip install markovian
```

`requests` is already a dependency of `crewai-tools`, so the tool also works
without the SDK installed.

---

## Arguments

| Argument | Type  | Required | Description                                                      |
| -------- | ----- | -------- | ---------------------------------------------------------------- |
| `data`   | `str` | Yes      | The content to stamp (an agent output, decision, or document).   |
| `label`  | `str` | No       | Optional human-readable label attached to the stamp.             |

Constructor options: `api_key` (optional), `wallet` (optional), `base_url`,
`timeout`. `MARKOVIAN_API_KEY` is read from the environment when set.

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
Markovian provenance receipt (markovian-provenance/v1):
  merkle_root: 65ddefa2b8d3fb994f2a4037f9dd8278688138bf1c5eaa9cdb64c73c02663466
  block_height: 135213
  verify_url: https://api.quantsynth.net/verify/65ddefa2b8d3fb994f2a4037f9dd8278688138bf1c5eaa9cdb64c73c02663466
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
