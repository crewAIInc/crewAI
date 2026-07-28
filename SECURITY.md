# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | ✅                 |

## Reporting a Vulnerability

We take security seriously. Please do NOT open public GitHub issues for vulnerabilities.

### How to Report

1. **Preferred**: GitHub [Private Security Advisory](https://github.com/crewAIInc/crewAI/security/advisories/new)
2. **Alternative**: Email security@crewai.com

### What to Include

- Description and impact
- Affected file(s) and line numbers
- Steps to reproduce or PoC
- Suggested fix (if available)

### Response Timeline

- Acknowledgment: 48 hours
- Status update: 1 week
- Resolution: severity-dependent

## Security Best Practices

- Use sandboxed execution for agent tools
- Never pickle.load() untrusted data
- Validate all URLs before outbound requests (prevent SSRF)
- Use parameterized queries for database operations
