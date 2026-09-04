# ChronoVerifyImageVerificationTool

## Description

This tool verifies when a photo was taken and its provenance using the [ChronoVerify](https://chronoverify.com) API. For each image it checks EXIF and XMP metadata, validates C2PA Content Credentials against the official trust lists, and runs classical pixel forensics, returning a single typed verdict with a 0 to 100 confidence score.

The verdict is one of:

| Verdict | Meaning |
|---|---|
| `provenance_confirmed` | Cryptographically valid C2PA Content Credentials from a signer on a recognised trust list |
| `consistent` | Metadata and pixel evidence agree with the claimed capture time and device |
| `inconclusive` | Not enough evidence to support or contradict the claim |
| `metadata_anomaly` | Metadata is missing, contradictory, or altered |
| `manipulation_indicated` | Pixel forensics or metadata indicate editing |

The response also includes the extracted capture time, capture device, capture location, a `c2pa` block, and file integrity hashes. The full response schema is published at `https://chronoverify.com/v1/verify.schema.json`.

**Scope**: ChronoVerify validates provenance. It is not a deepfake or AI-generation detector. Verdicts are investigative triage to support human review, not proof, and should not be the sole basis for any automated decision.

## Installation

Install the crewai_tools package:

```shell
pip install 'crewai[tools]'
```

No extra dependencies are required.

## Authentication (optional)

The tool works without any credentials on ChronoVerify's free keyless path, which is rate limited per IP.

For higher limits, set the `CHRONOVERIFY_API_KEY` environment variable. A free key (100 verifications per month, no card) can be requested with:

```shell
curl -X POST https://chronoverify.com/v1/keys/free -d "email=you@example.com"
```

```shell
export CHRONOVERIFY_API_KEY='cv_live_...'
```

## Example

```python
from crewai import Agent, Task
from crewai_tools import ChronoVerifyImageVerificationTool

tool = ChronoVerifyImageVerificationTool()

verifier = Agent(
    role="Image Provenance Analyst",
    goal="Verify when submitted photos were taken and where they came from",
    backstory=(
        "You examine image metadata, Content Credentials, and forensic "
        "signals to assess whether a photo's claimed origin holds up."
    ),
    tools=[tool],
    verbose=True,
)

task = Task(
    description=(
        "Verify the capture time and provenance of the image at "
        "https://example.com/photo.jpg and summarize the evidence."
    ),
    agent=verifier,
    expected_output="A short provenance assessment citing the verdict and confidence.",
)
```

## Arguments

- `image_path` (str, optional): Local path of the image file to verify.
- `image_url` (str, optional): Direct URL of the image to verify. Must be a direct link to the image bytes; redirects and hosts that block automated fetches fail, so prefer `image_path` when the file is available locally.

Provide exactly one of the two.

Constructor options:

- `api_base_url` (str, optional): Base URL of the ChronoVerify API. Defaults to `https://chronoverify.com`.
- `request_timeout` (int, optional): Request timeout in seconds. Defaults to 120.
