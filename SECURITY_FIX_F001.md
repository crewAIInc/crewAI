# Security Fix: F-001 - Sandbox Escape in CodeInterpreterTool

## Vulnerability Summary

**Severity:** Critical  
**Impact:** Remote Code Execution (RCE) on host system  
**CVSS Score:** 9.8 (Critical)  
**Status:** ✅ Fixed

## Description

The `CodeInterpreterTool` in crewAI contained a critical security vulnerability that allowed attackers to execute arbitrary code on the host system when Docker was unavailable.

### The Problem

When Docker was not available, the tool would fall back to `run_code_in_restricted_sandbox()`, which attempted to provide security by:
- Filtering the `__builtins__` dictionary to remove dangerous functions
- Blocking imports of certain modules (os, sys, subprocess, etc.)
- Providing a custom `__import__` function

However, **this approach does not provide real security isolation** because:
1. Python's object graph is still accessible
2. Attackers can use object introspection to recover the original `__import__` function
3. Once the real `__import__` is recovered, all module restrictions can be bypassed
4. Arbitrary code execution on the host becomes trivial

### Proof of Concept

The following code demonstrates the sandbox escape:

```python
# Classic Python sandbox escape via object introspection
for cls in ().__class__.__bases__[0].__subclasses__():
    if cls.__name__ == 'catch_warnings':
        # Get the real builtins module
        real_builtins = cls()._module.__builtins__
        real_import = real_builtins['__import__']
        # Now we can import os and execute commands
        os = real_import('os')
        os.system('whoami')  # RCE achieved
        break
```

## The Fix

### Changes Made

1. **Removed insecure fallback**: `run_code_safety()` now fails closed when Docker is unavailable
   - Raises `RuntimeError` with clear security explanation
   - Directs users to install Docker or use `unsafe_mode=True` if they trust the code

2. **Deprecated `run_code_in_restricted_sandbox()`**: 
   - Marked as deprecated and insecure in documentation
   - Added bold warnings about sandbox escape vulnerabilities
   - Kept for backward compatibility but with clear security warnings

3. **Updated `SandboxPython` class**:
   - Added security warnings to class documentation
   - Clarified that it provides NO real security boundary

4. **Updated tests**:
   - Tests now verify that Docker unavailability raises RuntimeError
   - Added test demonstrating the sandbox escape vulnerability
   - Tests that use the deprecated sandbox now include security warnings in comments

5. **Updated documentation**:
   - README now emphasizes Docker as a REQUIRED dependency
   - Explains why the fallback was removed
   - Provides clear guidance on `unsafe_mode` flag

### Secure Usage

**Recommended (Secure):**
```python
from crewai_tools import CodeInterpreterTool

# Requires Docker to be installed and running
tool = CodeInterpreterTool()
```

**Only if Docker cannot be installed AND code is fully trusted:**
```python
from crewai_tools import CodeInterpreterTool

# WARNING: No isolation, only use with trusted code!
tool = CodeInterpreterTool(unsafe_mode=True)
```

## Security Impact

### Before Fix
- ❌ Attackers could escape the "restricted" sandbox via object introspection
- ❌ Full RCE on host system when Docker unavailable
- ❌ Silent fallback gave false sense of security
- ❌ No indication that the sandbox was bypassable

### After Fix
- ✅ Fails closed when Docker unavailable (secure by default)
- ✅ Requires explicit `unsafe_mode=True` flag for unprotected execution
- ✅ Clear warnings and documentation about security requirements
- ✅ Docker isolation provides real process/filesystem/network boundaries

## Testing

A comprehensive test was added to demonstrate the vulnerability:

```python
def test_sandbox_escape_vulnerability_demonstration(printer_mock):
    """Demonstrate that the restricted sandbox is vulnerable to escape attacks.
    
    This test shows that an attacker can use Python object introspection to bypass
    the restricted sandbox and access blocked modules like 'os'. This is why the
    sandbox should never be used for untrusted code execution.
    """
    tool = CodeInterpreterTool()
    
    # Classic Python sandbox escape via object introspection
    escape_code = """
for cls in ().__class__.__bases__[0].__subclasses__():
    if cls.__name__ == 'catch_warnings':
        real_builtins = cls()._module.__builtins__
        real_import = real_builtins['__import__']
        os = real_import('os')
        result = "SANDBOX_ESCAPED" if hasattr(os, 'system') else "FAILED"
        break
"""
    
    result = tool.run_code_in_restricted_sandbox(escape_code)
    assert result == "SANDBOX_ESCAPED"
```

## Migration Guide

### For Users Currently Running Without Docker

If you were previously running CrewAI without Docker installed:

1. **Install Docker** (recommended): https://docs.docker.com/get-docker/
2. **OR** explicitly enable unsafe mode if you trust the code:
   ```python
   CodeInterpreterTool(unsafe_mode=True)
   ```

### Breaking Changes

- `CodeInterpreterTool()` now requires Docker by default
- Will raise `RuntimeError` if Docker is not available (instead of silently falling back)
- Users must explicitly opt-in to unsafe execution with `unsafe_mode=True`

## References

- **CrewAI Documentation**: https://docs.crewai.com/tools/ai-ml/codeinterpretertool
- **Docker Installation**: https://docs.docker.com/get-docker/
- **Related Files**:
  - `lib/crewai-tools/src/crewai_tools/tools/code_interpreter_tool/code_interpreter_tool.py`
  - `lib/crewai-tools/tests/tools/test_code_interpreter_tool.py`
  - `lib/crewai-tools/src/crewai_tools/tools/code_interpreter_tool/README.md`

## Timeline

- **Reported**: As part of security audit F-001
- **Fixed**: March 9, 2026
- **Commit**: 6ee0cacd7
- **Branch**: cursor/code-interpreter-sandbox-escape-c0a3

---

**Security Contact**: If you discover any security vulnerabilities, please report them responsibly.
