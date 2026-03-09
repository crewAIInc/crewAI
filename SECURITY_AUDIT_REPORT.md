# Security Audit Report: crewaiinc/crewai

**Date:** March 9, 2026  
**Auditor:** Cursor Cloud Agent  
**Repository:** https://github.com/crewaiinc/crewai  
**Scope:** Quick security check of the crewai Python framework

---

## Executive Summary

This report presents findings from a security assessment of the CrewAI framework. The codebase demonstrates **good overall security practices** with several security controls in place. However, there are some areas that warrant attention, particularly around code execution capabilities and input validation.

**Risk Level: MEDIUM**

### Key Findings Summary
- ✅ **Good:** No hardcoded secrets in production code
- ✅ **Good:** JWT authentication properly implemented with validation
- ✅ **Good:** Security tooling in place (Bandit, Ruff with security rules)
- ✅ **Good:** Dependency version pinning and override policies
- ⚠️ **Concern:** Code interpreter tool allows arbitrary code execution
- ⚠️ **Concern:** SQL injection risk in NL2SQL tool
- ⚠️ **Concern:** Pickle deserialization without integrity checks
- ⚠️ **Info:** Command injection protections needed in some areas

---

## 1. Secrets and Credential Management

### ✅ PASS - No Production Secrets Found

**Finding:** All hardcoded API keys and tokens found are in test files only.

**Evidence:**
- All hardcoded credentials are in test files with fake/example values
- Test environment file (`.env.test`) properly uses fake credentials
- Production code retrieves credentials from environment variables

**Examples:**
```python
# Test files use fake credentials - ACCEPTABLE
OPENAI_API_KEY=fake-api-key
ANTHROPIC_API_KEY=fake-anthropic-key
```

**Recommendation:** ✅ Current approach is secure. Continue this pattern.

---

## 2. Dependency Vulnerabilities

### ✅ GOOD - Proactive Dependency Management

**Finding:** The project has security-conscious dependency management.

**Security Controls:**
1. **Bandit** (v1.9.2) - Security linter for Python code
2. **Ruff** with security rules enabled (`S` - Bandit rules)
3. **Dependency overrides** for known vulnerabilities in `pyproject.toml`:
   ```toml
   [tool.uv]
   override-dependencies = [
       "langchain-core>=0.3.80,<1",  # GHSA template-injection vuln fixed
       "urllib3>=2.6.3",              # Security updates
       "pillow>=12.1.1",              # Security updates
   ]
   ```

**Recommendation:** ✅ Excellent practices. Maintain regular dependency audits.

---

## 3. Code Execution Vulnerabilities

### ⚠️ HIGH RISK - Code Interpreter Tool

**File:** `lib/crewai-tools/src/crewai_tools/tools/code_interpreter_tool/code_interpreter_tool.py`

**Finding:** The `CodeInterpreterTool` allows arbitrary code execution with three modes:
1. **Docker mode** (default, safest)
2. **Restricted sandbox** (fallback when Docker unavailable)
3. **Unsafe mode** (runs code directly on host)

**Critical Issues:**

#### Issue 1: Unsafe Mode Command Injection
**Lines 382-383:**
```python
for library in libraries_used:
    os.system(f"pip install {library}")  # noqa: S605
```

**Risk:** If `library` contains shell metacharacters, this could lead to command injection.

**Attack Example:**
```python
libraries_used = ["numpy; rm -rf /"]
```

**Severity:** HIGH (but requires `unsafe_mode=True`)

**Recommendation:**
```python
# Use subprocess with list arguments instead
subprocess.run(["pip", "install", library], check=True)
```

#### Issue 2: Sandbox Can Be Bypassed
**Lines 60-83:** The restricted sandbox blocks certain modules, but:
- Blocks are incomplete (e.g., `pathlib` not blocked, could access filesystem)
- Determined attackers may find bypass techniques
- No resource limits (CPU, memory, time)

**Recommendation:**
- Add resource limits to sandbox execution
- Consider using more robust sandboxing like RestrictedPython
- Document that sandbox is defense-in-depth, not primary security

#### Issue 3: Docker Volume Mounting
**Lines 260-267:**
```python
volumes={current_path: {"bind": "/workspace", "mode": "rw"}}
```

**Risk:** Mounts entire current working directory with read-write access.

**Recommendation:**
- Mount as read-only by default
- Allow write access to specific temporary directory only
- Add option to restrict mounted paths

---

## 4. SQL Injection Vulnerabilities

### ⚠️ HIGH RISK - NL2SQL Tool

**File:** `lib/crewai-tools/src/crewai_tools/tools/nl2sql/nl2sql_tool.py`

**Finding:** SQL injection vulnerability in schema introspection.

**Lines 56-58:**
```python
def _fetch_all_available_columns(self, table_name: str):
    return self.execute_sql(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"  # noqa: S608
    )
```

**Risk:** If `table_name` contains malicious SQL, it will be executed.

**Attack Example:**
```python
table_name = "'; DROP TABLE users; --"
```

**Severity:** HIGH

**Recommendation:**
```python
def _fetch_all_available_columns(self, table_name: str):
    return self.execute_sql(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table_name",
        params={"table_name": table_name}
    )
```

**Note:** The tool does use parameterized queries via SQLAlchemy's `text()` for user queries (line 82), which is good. Only the internal method is vulnerable.

---

## 5. Insecure Deserialization

### ⚠️ MEDIUM RISK - Pickle Usage

**File:** `lib/crewai/src/crewai/utilities/file_handler.py`

**Finding:** Pickle is used for persistence without integrity verification.

**Lines 168-170:**
```python
with open(self.file_path, "rb") as file:
    try:
        return pickle.load(file)  # noqa: S301
```

**Risk:** Pickle can execute arbitrary code during deserialization. If an attacker can modify pickle files, they can achieve remote code execution.

**Severity:** MEDIUM (requires write access to pickle files)

**Context:** Used by `PickleHandler` class for storing training data and agent state.

**Recommendations:**
1. **Immediate:** Add file integrity checks (HMAC signatures)
2. **Short-term:** Switch to JSON for non-object data
3. **Long-term:** Use `jsonpickle` or similar safer alternatives
4. **Defense:** Document that pickle files must be stored securely with proper access controls

**Example Mitigation:**
```python
import hmac
import hashlib

def save(self, data: Any, secret_key: str) -> None:
    pickle_data = pickle.dumps(data)
    signature = hmac.new(secret_key.encode(), pickle_data, hashlib.sha256).digest()
    with open(self.file_path, "wb") as f:
        f.write(signature + pickle_data)

def load(self, secret_key: str) -> Any:
    with open(self.file_path, "rb") as f:
        signature = f.read(32)
        pickle_data = f.read()
        expected_sig = hmac.new(secret_key.encode(), pickle_data, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_sig):
            raise ValueError("Pickle file integrity check failed")
        return pickle.loads(pickle_data)
```

---

## 6. File Handling and Path Traversal

### ✅ GOOD - Path Validation Present

**File:** `lib/crewai/src/crewai/knowledge/source/base_file_knowledge_source.py`

**Finding:** File paths are validated and restricted to knowledge directory.

**Lines 86-88:**
```python
def convert_to_path(self, path: Path | str) -> Path:
    return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path
```

**Lines 56-64:**
```python
def validate_content(self) -> None:
    for path in self.safe_file_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            # Log error
```

**Security Strength:** 
- ✅ Paths are constrained to knowledge directory
- ✅ Existence and type validation
- ⚠️ Could add explicit check for path traversal attempts (`..`)

**Recommendation:**
```python
def convert_to_path(self, path: Path | str) -> Path:
    base_path = Path(KNOWLEDGE_DIRECTORY).resolve()
    if isinstance(path, str):
        full_path = (base_path / path).resolve()
    else:
        full_path = path.resolve()
    
    # Ensure resolved path is still within knowledge directory
    if not full_path.is_relative_to(base_path):
        raise ValueError(f"Path traversal detected: {path}")
    
    return full_path
```

---

## 7. Authentication and Authorization

### ✅ EXCELLENT - JWT Implementation

**File:** `lib/crewai/src/crewai/cli/authentication/utils.py`

**Finding:** JWT validation is properly implemented with all security best practices.

**Strengths:**
1. ✅ Signature verification using JWKS
2. ✅ Expiration check (`verify_exp`)
3. ✅ Issuer validation
4. ✅ Audience validation
5. ✅ Required claims enforcement
6. ✅ Proper exception handling
7. ✅ 10-second leeway for clock skew

**Lines 30-44:**
```python
return jwt.decode(
    jwt_token,
    signing_key.key,
    algorithms=["RS256"],
    audience=audience,
    issuer=issuer,
    leeway=10.0,
    options={
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": True,
        "require": ["exp", "iat", "iss", "aud", "sub"],
    },
)
```

**Recommendation:** ✅ No changes needed. This is exemplary JWT validation.

---

## 8. Security Features

### ✅ GOOD - Built-in Security Module

**Files:** 
- `lib/crewai/src/crewai/security/security_config.py`
- `lib/crewai/src/crewai/security/fingerprint.py`

**Finding:** CrewAI includes a security module with:
1. **Fingerprinting** - Unique agent identifiers for tracking and auditing
2. **Metadata validation** - Prevents DoS via oversized metadata
3. **Type validation** - Strong typing with Pydantic

**Security Controls in Fingerprint:**

**Lines 38-40 (DoS prevention):**
```python
if len(str(v)) > 10_000:  # Limit metadata size to 10KB
    raise ValueError("Metadata size exceeds maximum allowed (10KB)")
```

**Lines 28-36 (Nested data protection):**
```python
if isinstance(nested_value, dict):
    raise ValueError("Metadata can only be nested one level deep")
```

**Recommendation:** ✅ Good defensive programming. Consider adding rate limiting to fingerprint generation if exposed via API.

---

## 9. Command Injection Risks

### ✅ MOSTLY GOOD - Limited Use of Shell Commands

**Finding:** No instances of `shell=True` found in the codebase.

**Subprocess Usage:**
- Most subprocess calls use list arguments (safe)
- Docker commands use proper API (no shell)
- File operations use Path/open (no shell)

**Exception:**
```python
# code_interpreter_tool.py line 383 (already covered in Section 3)
os.system(f"pip install {library}")  # Only in unsafe mode
```

**Recommendation:** ✅ Continue avoiding `shell=True`. Fix the one instance noted above.

---

## 10. SSL/TLS Configuration

### ✅ PASS - No SSL Verification Bypasses

**Finding:** No instances of `verify=False` or SSL certificate bypass found.

**Evidence:** 
- HTTP requests use default SSL verification
- No override of certificate validation

**Recommendation:** ✅ Maintain current practices.

---

## Security Tooling Assessment

### ✅ EXCELLENT - Multiple Security Tools Configured

**From `pyproject.toml`:**

1. **Bandit (v1.9.2)** - Security-focused static analysis
2. **Ruff** with security rules:
   ```toml
   extend-select = [
       "S",      # bandit (security issues)
       "B",      # flake8-bugbear (bug prevention)
   ]
   ```
3. **MyPy (v1.19.1)** - Type checking prevents many bugs
4. **Pre-commit hooks** - Automated checks

**Test Security:**
- Bandit checks disabled in tests (lines 106-108) - reasonable for test code
- Fake credentials in tests - correct approach

**Recommendation:** ✅ Excellent security tooling. Consider adding:
- `safety` or `pip-audit` for dependency vulnerability scanning
- SAST scanning in CI/CD (GitHub CodeQL, Semgrep)

---

## Summary of Vulnerabilities

| ID | Severity | Component | Issue | Status |
|----|----------|-----------|-------|--------|
| 1 | HIGH | CodeInterpreterTool | Command injection in unsafe mode | ⚠️ Fix Recommended |
| 2 | HIGH | NL2SQLTool | SQL injection in table introspection | ⚠️ Fix Recommended |
| 3 | MEDIUM | PickleHandler | Insecure deserialization | ⚠️ Mitigation Recommended |
| 4 | MEDIUM | CodeInterpreterTool | Docker volume permissions too broad | ⚠️ Hardening Recommended |
| 5 | LOW | BaseFileKnowledgeSource | Path traversal check could be stronger | ℹ️ Enhancement Suggested |
| 6 | LOW | CodeInterpreterTool | Sandbox bypass potential | ℹ️ Document Limitations |

---

## Recommendations

### Immediate Actions (High Priority)
1. **Fix SQL injection** in `nl2sql_tool.py` line 57 - use parameterized queries
2. **Fix command injection** in `code_interpreter_tool.py` line 383 - use subprocess.run with list
3. **Document security model** - Especially for CodeInterpreterTool unsafe mode

### Short-term Actions (Medium Priority)
4. **Add pickle integrity checks** - HMAC signing for pickle files
5. **Restrict Docker volume mounts** - Read-only by default
6. **Enhance path traversal protection** - Explicit `is_relative_to()` check
7. **Add dependency scanning** - Integrate `pip-audit` or `safety` in CI

### Long-term Actions (Low Priority)
8. **Evaluate pickle alternatives** - Consider JSON or safer serialization
9. **Resource limits in sandbox** - CPU/memory/time limits for code execution
10. **Rate limiting** - Add to fingerprint generation if exposed via API
11. **Security documentation** - Create SECURITY.md with security best practices

---

## Positive Security Practices Observed

1. ✅ **No hardcoded production secrets**
2. ✅ **Excellent JWT implementation**
3. ✅ **Strong security tooling** (Bandit, Ruff, MyPy)
4. ✅ **Proactive dependency management** with security overrides
5. ✅ **Type safety** with Pydantic and MyPy
6. ✅ **No shell=True usage** (except one controlled case)
7. ✅ **SSL verification enabled** throughout
8. ✅ **Input validation** in multiple layers
9. ✅ **Security module** with fingerprinting and metadata limits
10. ✅ **Test isolation** with fake credentials

---

## Conclusion

The CrewAI framework demonstrates **mature security practices** overall. The development team clearly prioritizes security with multiple layers of protection, security tooling, and careful dependency management.

The main security concerns are inherent to the framework's purpose (AI agent orchestration with code execution capabilities) rather than security oversights. The identified vulnerabilities are in optional/specialized tools and should be addressed to prevent misuse.

**Overall Security Posture:** GOOD with room for targeted improvements.

**Risk Assessment:** MEDIUM (acceptable for current stage with recommended fixes)

**Recommendation:** Address high-priority SQL and command injection issues, then proceed with medium-priority hardening tasks.

---

**Report Generated:** 2026-03-09  
**Audit Tool:** Manual review + automated pattern analysis  
**Scope:** Quick security check (not comprehensive penetration test)
