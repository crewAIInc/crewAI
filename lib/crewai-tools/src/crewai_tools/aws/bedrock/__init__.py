from crewai_tools.aws.bedrock.agents.invoke_agent_tool import BedrockInvokeAgentTool
from crewai_tools.aws.bedrock.browser import create_browser_toolkit
from crewai_tools.aws.bedrock.code_interpreter import create_code_interpreter_toolkit
from crewai_tools.aws.bedrock.knowledge_base.retriever_tool import (
    BedrockKBRetrieverTool,
)


def __getattr__(name: str):
    if name == "AgentCoreRuntime":
        from crewai_tools.aws.bedrock.runtime import AgentCoreRuntime
        return AgentCoreRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentCoreRuntime",
    "BedrockInvokeAgentTool",
    "BedrockKBRetrieverTool",
    "create_browser_toolkit",
    "create_code_interpreter_toolkit",
]
