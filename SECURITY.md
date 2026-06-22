# Security Policy

Thanks for helping keep CrewAI and its users safe.

## Reporting a Vulnerability

**Please do not open public GitHub issues for security vulnerabilities in CrewAI.**

To report a security issue, use GitHub's private vulnerability reporting feature:

1. Go to <https://github.com/crewAIInc/crewAI/security/advisories/new>
2. Provide a clear description, affected version(s), and reproduction steps.

A maintainer will respond to triage the report. If you cannot use GitHub's
private reporting, please contact the maintainers via the channels listed
in the project [README](README.md).

### What to include

A good report typically contains:

- The affected component(s) (e.g. `crewai.tools`, `crewai.agent`)
- The CrewAI version(s) affected
- A clear description of the impact
- A minimal reproduction (PoC) or detailed steps
- Any suggested mitigation

### Disclosure process

We follow a coordinated-disclosure model:

1. The report is confirmed and the issue reproduced.
2. A fix is developed and reviewed privately (typically via a [GitHub Security Advisory](https://github.com/crewAIInc/crewAI/security/advisories)).
3. A release date is coordinated with the reporter.
4. A patched release is published and the advisory is disclosed, crediting the
   reporter unless anonymity is requested.

### Out of scope

The following are generally **not** considered security vulnerabilities in
CrewAI itself:

- Issues caused entirely by user-provided LLM prompts or tool implementations
- Rate limits or availability issues from upstream LLM providers
- Misconfiguration of credentials passed to `LLM(...)` or environment variables
- Security issues in third-party tools loaded into CrewAI; please report those
  to the tool's upstream maintainers

## Supported Versions

Security fixes are provided for the most recent release of CrewAI on `main`.
Older releases may receive critical fixes at the maintainers' discretion.
