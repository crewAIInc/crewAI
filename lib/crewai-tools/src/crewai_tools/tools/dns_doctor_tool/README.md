# DNS Doctor Tools Documentation

## Description
Three tools over the hosted [DNS Doctor](https://dnsdoctor.dev) API that give an agent deterministic DNS and email-authentication checks for a domain:

- `DnsDoctorScanTool`: the full report (SPF, DKIM, DMARC, MX, DNS health, blacklists, domain and TLS expiry) with per-check verdicts and copy-paste fix records from a validating engine, never a model-generated record.
- `DnsDoctorDmarcUpgradeTool`: the next safe DMARC record for a domain, alignment-gated, with the rationale.
- `DnsDoctorPropagationTool`: whether a DNS change has propagated, read from six locations on four continents.

No API key is needed. Responses are relayed verbatim as JSON text.

## Installation
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import DnsDoctorScanTool, DnsDoctorDmarcUpgradeTool, DnsDoctorPropagationTool

scan = DnsDoctorScanTool()
upgrade = DnsDoctorDmarcUpgradeTool()
propagation = DnsDoctorPropagationTool()

print(scan.run(domain="example.com"))
print(upgrade.run(domain="example.com"))
print(propagation.run(name="www.example.com", record_type="A", expected_value="203.0.113.10"))
```

## Arguments
- `DnsDoctorScanTool`: `domain` (required).
- `DnsDoctorDmarcUpgradeTool`: `domain` (required).
- `DnsDoctorPropagationTool`: `name` (required), `record_type` (default `A`), `expected_value` (optional; omit to check consistency only).

## Environment
- `DNSDOCTOR_API_TOKEN` (optional): raises the anonymous per-caller rate limit and unlocks nothing else. Past the free allowance the API answers `402` with an x402 offer; the tool returns that as text.
- `DNSDOCTOR_API_BASE` (optional): override the API origin.

## Rules the tools follow
- Present any returned record verbatim; never rewrite or reformat it.
- A `temperror` check status is a transient lookup failure, not a failure of the domain. A transport failure (rate limit, 5xx, network) is returned as text that says it is not a verdict.
- `record` can be `null` in the DMARC upgrade response: the `rationale` is the answer; do not compose a record to fill the gap.
- SPF is diagnose-only: the scan reports SPF findings but the tools never propose SPF edits.
- A human must approve every DNS change; nothing is applied automatically.
