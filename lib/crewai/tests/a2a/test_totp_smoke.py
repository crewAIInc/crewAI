#!/usr/bin/env python3
"""
TOTP A2A Auth — standalone smoke test
Run this script. It starts a minimal A2A server with TOTP enabled,
then hits it with curl-equivalent requests to verify each scenario.

Requirements:
  pip install pyotp a2a-sdk httpx starlette uvicorn --break-system-packages

Usage:
  python3 test_totp_smoke.py
"""

import asyncio
import os
import sys
import threading
import time

import pyotp
import httpx

# ── Config ─────────────────────────────────────────────────────────────────────
BEARER_TOKEN = "test-bearer-token-abc123"
TOTP_SEED    = pyotp.random_base32()   # fresh seed each run
PORT         = 18200
BASE_URL     = f"http://127.0.0.1:{PORT}"

# ── Minimal A2A server with TOTP ───────────────────────────────────────────────
def start_server():
    """Start a minimal Starlette A2A server with TOTP auth in a background thread."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    import uvicorn

    async def handle(request: Request):
        auth = request.headers.get("authorization", "")
        totp_code = request.headers.get("x-totp", "")

        # Validate bearer
        if auth != f"Bearer {BEARER_TOKEN}":
            return JSONResponse({"error": "bad token"}, status_code=401)

        # Validate TOTP
        totp = pyotp.TOTP(TOTP_SEED)
        if not totp_code or not totp.verify(totp_code, valid_window=1):
            return JSONResponse({"error": "bad totp"}, status_code=401)

        return JSONResponse({"status": "ok", "message": "authenticated"})

    app = Starlette(routes=[Route("/a2a", handle, methods=["POST"])])
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(1.0)  # let it bind

# ── Test cases ─────────────────────────────────────────────────────────────────
async def run_tests():
    totp = pyotp.TOTP(TOTP_SEED)
    passed = 0
    failed = 0

    async with httpx.AsyncClient() as client:

        async def check(label, expected_status, headers):
            resp = await client.post(f"{BASE_URL}/a2a", headers=headers, json={})
            ok = resp.status_code == expected_status
            icon = "✅" if ok else "❌"
            print(f"  {icon} [{resp.status_code}] {label}")
            return ok

        print("\n=== TOTP A2A Auth Smoke Test ===\n")
        print(f"  Seed: {TOTP_SEED}")
        print(f"  Current code: {totp.now()}\n")

        results = await asyncio.gather(
            check(
                "Valid bearer + valid TOTP → 200",
                200,
                {"Authorization": f"Bearer {BEARER_TOKEN}", "X-TOTP": totp.now()}
            ),
            check(
                "Valid bearer + NO TOTP → 401",
                401,
                {"Authorization": f"Bearer {BEARER_TOKEN}"}
            ),
            check(
                "Valid bearer + WRONG TOTP → 401",
                401,
                {"Authorization": f"Bearer {BEARER_TOKEN}", "X-TOTP": "000000"}
            ),
            check(
                "NO bearer + valid TOTP → 401",
                401,
                {"X-TOTP": totp.now()}
            ),
            check(
                "No headers at all → 401",
                401,
                {}
            ),
        )

    passed = sum(results)
    failed = len(results) - passed
    print(f"\n  {passed}/{len(results)} passed", "🎉" if failed == 0 else "⚠️")
    return failed == 0

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Install deps if missing
    try:
        import uvicorn
    except ImportError:
        print("Installing dependencies...")
        os.system(f"{sys.executable} -m pip install uvicorn starlette --break-system-packages -q")
        import uvicorn

    print("Starting test server...")
    start_server()

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
