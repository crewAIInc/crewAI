# Secure Coding Standard

This document is authoritative for all AI coding agents, automated code
reviewers, and human contributors. Any conflict between generated code and this
standard MUST be resolved in favor of this standard.

This document defines the minimum secure coding requirements for all CrewAI
open-source repositories.

Its goals are to:

1. Produce secure-by-default code.
2. Guide AI coding agents during implementation.
3. Provide objective criteria for security code reviews.
4. Reduce false-positive vulnerability reports by defining the project's
   expected security controls.

All requirements in this document are normative.

The keywords **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY**
are interpreted as described in RFC 2119.

---

# Scope

This document specifies **insecure-coding** requirements: injection, path
traversal, unsafe deserialization, secret exposure, unbounded I/O, and similar
implementation defects.

It does **not** specify model-level or product-threat controls. Prompt
injection, tool-metadata influence, and similar AI risks belong in
`docs/security/threat-model.md`, not here.

---

# Security Principles

All code MUST follow these principles.

## Validate Untrusted Input

Assume all external input is untrusted until validated.

Examples include:

- User input
- HTTP requests
- Environment variables
- Files
- Archives
- LLM responses
- Tool outputs
- Plugin outputs
- Network responses

---

## Least Privilege

Components MUST only receive the permissions they require.

Avoid unnecessary:

- filesystem access
- network access
- process execution
- credentials
- secrets

---

## Fail Securely

Security-sensitive operations MUST fail closed.

Do not continue after:

- failed authorization
- failed validation
- partial execution
- partial transactions

Roll back where appropriate.

---

## Defense in Depth

Security MUST NOT rely on a single validation step.

Where appropriate:

- validate input
- authorize operations
- restrict capabilities
- log security events
- enforce resource limits

---

# Approved Project Abstractions

AI-generated and human-authored code MUST use these helpers instead of new
implementations of the same control.

| Control | MUST use |
|---|---|
| File and directory containment | `crewai_tools.security.safe_path.validate_file_path`, `validate_directory_path` |
| Path and error display | `format_path_for_display`, `format_error_for_display`, `format_sandbox_error` |
| URL validation | `crewai_tools.security.safe_path.validate_url` |
| HTTP fetch | `crewai_tools.security.safe_requests.safe_get`, `safe_get_bounded` |
| Archive extraction | `tarfile.extractall(..., filter="data")` when available; otherwise the `_safe_extractall` / `_safe_extract_zip` pattern in `crewai.skills.cache` and `crewai_cli.skills.main` |
| YAML | `yaml.safe_load()` |
| Project Python references | existing crew-loader confinement to the project root; MUST NOT add a parallel importer |
| SQL values | parameterized queries / bind parameters |
| SQL statement policy in NL2SQL-style tools | the existing default-deny DML / multi-statement checks in `NL2SQLTool` |

`crewai_tools.utilities.safe_path` is a compatibility re-export. New code MUST
import from `crewai_tools.security.safe_path`.

Calling `validate_url()` and then `requests.get()` is **not** sufficient. DNS
rebinding and redirect hops require `safe_get` / `safe_get_bounded`.

`CREWAI_TOOLS_ALLOW_UNSAFE_PATHS` disables both path and SSRF checks
process-wide. New code MUST NOT add another process-wide escape hatch. Prefer
`base_dir` to widen a sandbox. Managed workers MUST keep
`CREWAI_TOOLS_FORCE_SAFE_PATHS` as the override that cannot be disabled by
`ALLOW_UNSAFE_PATHS`.

Existing uses of prohibited APIs in this repository do not authorize copying
those APIs into new code.

---

# 1. Command Execution

## Risk

Unsafe process execution may result in command injection or arbitrary command
execution.

## MUST NOT

- `os.system()`
- `os.popen()`
- `subprocess.run(..., shell=True)`
- `subprocess.Popen(..., shell=True)`
- `subprocess.call(..., shell=True)`
- `subprocess.check_call(..., shell=True)`
- `subprocess.check_output(..., shell=True)`

MUST NOT construct shell commands using:

- string concatenation
- f-strings
- `%` formatting
- `.format()`

## MUST

Use:

```python
subprocess.run([...], shell=False, check=True)
```

Pass arguments as a list.

Validate all externally influenced arguments.

The executable and each argument that is not a literal or allowlisted constant
MUST be validated before use. Hardcoded optional-dependency installs such as
`subprocess.run(["uv", "add", "package"], shell=False, check=True)` are
acceptable only when no argument is derived from untrusted input.

---

# 2. Filesystem Access

## Risk

Unsafe filesystem operations may allow:

- path traversal
- symlink attacks
- arbitrary file overwrite
- unauthorized file access

## MUST NOT

- Trust user-provided paths.
- Validate paths using `startswith()`.
- Extract archives without validating every entry.

## MUST

- Canonicalize paths.
- Validate against an approved root directory.
- Reject paths escaping the allowed root.
- Validate archive contents before extraction.

In `crewai-tools`, MUST call `validate_file_path` or
`validate_directory_path`. MUST NOT reimplement containment with `startswith`,
string prefix checks, or manual `..` filtering. The helpers resolve symlinks
and `..` before comparing against `base_dir`.

Archive members MUST be validated before extraction, including symlink and
hardlink targets, not only member names. Copy the existing `_safe_extractall`
/ `_safe_extract_zip` checks; do not add `extractall` without them.

---

# 3. Dynamic Code Execution

## Risk

Executing attacker-controlled code results in arbitrary code execution.

## MUST NOT

- `eval()`
- `exec()`
- `compile()` on untrusted input
- Dynamic imports from user-controlled values

## MUST

Use:

- explicit parsing
- JSON
- Pydantic models
- controlled plugin registries
- allowlists

`ast.literal_eval` MAY be used to parse literals. It is not a substitute for
`json.loads` when the input is JSON.

Project definition Python references MUST resolve through the existing loader,
which confines imports to the project root and rejects stdlib/external targets
such as `os.system`. New code MUST NOT call `importlib.import_module` on a
value derived from untrusted input.

---

# 4. Deserialization

## Risk

Unsafe deserialization can execute arbitrary code.

## MUST NOT

- `pickle.load()`
- `pickle.loads()`
- `dill.load()`
- `dill.loads()`
- `marshal.loads()`
- `yaml.load()` with unsafe loaders

## MUST

Use:

- `json.loads()`
- `yaml.safe_load()`
- schema validation
- Pydantic models

Do not add new pickle, dill, or marshal loaders. Existing pickle usage is not
a template for new code.

---

# 5. Injection

## Risk

User-controlled data MUST NOT become executable instructions.

Includes:

- SQL Injection
- Command Injection
- Template Injection
- LDAP Injection
- XPath Injection
- NoSQL Injection
- Log Injection

## MUST NOT

- Build SQL using string formatting.
- Build commands using string concatenation.
- Insert unescaped user input into templates.

## MUST

Use:

- parameterized queries
- prepared statements
- structured logging
- output encoding

SQL values MUST be passed as bind parameters (for example SQLAlchemy
`text(query)` with a params mapping). MUST NOT interpolate untrusted strings
into the query text.

Tools that accept whole SQL statements MUST keep a default-deny write policy
and reject stacked statements in read-only mode, following `NL2SQLTool`. MUST
NOT add a new SQL tool that executes caller-supplied SQL with no statement
policy.

---

# 6. Network Requests

## Risk

Network requests can expose internal services or leak sensitive information.

## MUST NOT

- Disable TLS verification.
- Fetch arbitrary URLs without validation.
- Follow redirects blindly.
- Make requests without timeouts.

## MUST

- Validate URLs.
- Restrict supported protocols.
- Block localhost and private addresses where appropriate.
- Configure connection and read timeouts.
- Verify TLS certificates.

In `crewai-tools` and any code that fetches URLs chosen at runtime, MUST use
`safe_get` or `safe_get_bounded`. Those helpers:

- allow only `http`/`https`
- block `file://`
- reject private, loopback, link-local, and metadata addresses
- validate every redirect hop
- pin TCP to the checked IP
- strip credentials on cross-origin redirects
- default timeout to 30 seconds

`safe_get_bounded` MUST be used when the response body is read into memory.

MUST NOT pass `verify=False`. MUST NOT follow redirects with a raw
`requests`/`httpx` client.

---

# 7. Secrets

## Risk

Hardcoded or exposed secrets compromise deployments.

## MUST NOT

- Hardcode API keys.
- Hardcode passwords.
- Hardcode tokens.
- Hardcode private keys.
- Log secrets.
- Return secrets in exceptions.
- Commit credentials to source control.

## MUST

Use:

- environment variables
- secret management systems
- secret masking
- dependency injection

Logs, metrics, traces, and exceptions MUST NOT contain secrets or absolute
filesystem prefixes. Use `format_path_for_display` and
`format_error_for_display` instead of interpolating raw paths or `OSError`
text.

---

# 8. Authorization

## Risk

Every sensitive operation requires authorization.

## MUST NOT

- Trust user-supplied roles.
- Trust user-supplied ownership.
- Skip authorization checks.
- Expand tool capabilities beyond their documented purpose.

## MUST

- Verify authorization before sensitive operations.
- Enforce least privilege.
- Validate ownership server-side.
- Respect documented trust boundaries.

---

# 9. Resource Management

## Risk

Unbounded operations may result in denial of service.

## MUST NOT

- Read unbounded files.
- Create infinite loops.
- Allow unlimited retries.
- Spawn unlimited threads.
- Process unlimited input.
- Execute operations without timeouts.

## MUST

- Limit file sizes.
- Limit memory usage.
- Configure execution timeouts.
- Configure network timeouts.
- Limit concurrency.
- Validate input sizes.

HTTP response bodies MUST be capped with `safe_get_bounded`. File and archive
readers MUST enforce a size limit before loading content into memory. Network
and subprocess calls MUST set a timeout.

In `crewai-files`, MUST reuse the existing provider constraint validators
rather than adding unbounded reads.

---

# 10. Cryptography

## Risk

Weak cryptography compromises confidentiality and integrity.

## MUST NOT

- Use `random` for security-sensitive values.
- Invent cryptographic algorithms.
- Store plaintext passwords.
- Compare secrets using normal equality.
- Disable certificate validation.

## MUST

Use:

- `secrets`
- `hmac.compare_digest()`
- Argon2id, bcrypt, or scrypt
- authenticated encryption
- modern TLS

---

# Error Handling

Security-relevant exceptions MUST be handled intentionally.

## MUST NOT

- Swallow exceptions.
- Continue after partial failure.
- Expose stack traces to users.
- Leak filesystem paths or secrets.

## MUST

- Fail securely.
- Roll back incomplete operations.
- Return safe error messages.
- Log security-relevant failures.

User-facing tool errors MUST go through `format_sandbox_error` /
`format_error_for_display` so they do not advertise
`CREWAI_TOOLS_ALLOW_UNSAFE_PATHS` or leak absolute paths.

---

# Security Logging

The following SHOULD generate security logs where applicable:

- authentication attempts
- authorization failures
- input validation failures
- rejected commands
- rejected filesystem access
- security policy violations
- privilege changes
- security configuration changes

Logs MUST NOT contain:

- passwords
- API keys
- tokens
- private keys
- sensitive personal data
- absolute filesystem prefixes of rejected paths

---

# AI Code Generation

AI-generated code MUST be treated as untrusted until reviewed.

Before completing an implementation, AI coding agents MUST:

1. Review generated code against this document.
2. Remove prohibited APIs and patterns.
3. Verify that security controls have not been removed.
4. Prefer approved project abstractions over new implementations.
5. State any security assumptions made.

AI MUST NOT remove or weaken existing:

- validation
- authorization
- error handling
- logging
- resource limits

without explicit justification.

New code MUST NOT introduce APIs listed as MUST NOT, including when similar
calls already exist elsewhere in the tree.

---

# AI Security Reviews

When reviewing existing code, findings MUST distinguish between:

- Security vulnerabilities
- Hardening recommendations
- Code quality issues

The existence of a dangerous API does not, by itself, establish a security
vulnerability.

A valid finding MUST demonstrate:

1. Attacker-controlled input.
2. Reachable source-to-sink data flow.
3. A security-sensitive operation.
4. Failure of an intended security control.
5. Demonstrable security impact.

Reports MUST follow the assumptions defined in
`docs/security/threat-model.md`.

This split is intentional:

- **Implementation** follows the MUST / MUST NOT rules so new code is
  secure-by-default.
- **Vulnerability reports** require the five-point test so existing or
  gated use of a dangerous API is not filed as a CVE without a reachable
  exploit.

---

# Pull Request Checklist

Before merging, verify:

- [ ] No prohibited APIs introduced.
- [ ] Approved project abstractions used instead of new security helpers.
- [ ] Inputs are validated.
- [ ] Authorization is enforced.
- [ ] No unsafe deserialization.
- [ ] No injection vulnerabilities.
- [ ] Network requests are bounded and validated.
- [ ] Secrets are not exposed.
- [ ] Resource limits exist.
- [ ] Errors fail securely.
- [ ] Security logging is preserved.
- [ ] AI-generated code has been reviewed against this document.
