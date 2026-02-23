#!/usr/bin/env python3
"""
临时验证脚本：模拟 Groq 失败场景，确保 LiteLLM fallback 不会静默失败。

用法（需先安装 litellm extra）:
  uv run python scripts/verify_groq_fallback_error_handling.py

预期：应看到明确的异常（如认证错误），而非程序卡死或无输出。
"""
import os
import sys

# 强制无效的 Groq API Key，确保会触发错误而非静默失败
os.environ["GROQ_API_KEY"] = "invalid-key-for-fallback-test"

# 在导入 crewai 前设置，避免 .env 覆盖
def main() -> None:
    from crewai.llm import LLM

    print("Creating LLM with model=groq/llama3-70b-8192 (LiteLLM fallback path)...")
    llm = LLM(model="groq/llama3-70b-8192")
    assert getattr(llm, "is_litellm", False), "Expected LiteLLM fallback instance"

    print("Calling llm.call('Hello') with invalid GROQ_API_KEY...")
    try:
        result = llm.call("Hello")
        print("ERROR: No exception raised. Result:", result)
        sys.exit(1)
    except Exception as e:
        print("OK: Exception received (no silent failure):")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        # 可选：若希望严格匹配某种错误类型可在此断言
        # assert "auth" in str(e).lower() or "api" in str(e).lower() or "401" in str(e)
    print("Done. Fallback error handling verified.")

if __name__ == "__main__":
    main()
