from pydantic import BaseModel, Field


class MCPRegistryRiskParams(BaseModel):
    server_url: str | None = Field(
        default=None,
        description="Full URL of the MCP server to check, e.g. 'https://example.com/mcp'. "
                     "Provide this or package_name.",
    )
    package_name: str | None = Field(
        default=None,
        description="Package name of the MCP server if no server_url is available. "
                     "Checks are more limited without a server_url.",
    )


class PromptInjectionBreachParams(BaseModel):
    email: str = Field(
        description="Email address to check for credential exposure sourced from "
                     "prompt-injection attacks against AI agents.",
    )
