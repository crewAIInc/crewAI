"""MCP Tool Wrapper voor on-demand MCP server verbindingen."""

import asyncio

from crewai.tools import BaseTool


MCP_CONNECTION_TIMEOUT = 15
MCP_TOOL_EXECUTION_TIMEOUT = 60
MCP_DISCOVERY_TIMEOUT = 15
MCP_MAX_RETRIES = 3


class MCPToolWrapper(BaseTool):
    """Lichtgewicht wrapper voor MCP tools die on-demand verbindt."""

    def __init__(
        self,
        mcp_server_params: dict,
        tool_name: str,
        tool_schema: dict,
        server_name: str,
    ):
        """Initialiseer de MCP tool wrapper.

        Args:
            mcp_server_params: Parameters voor verbinding met de MCP server
            tool_name: Originele naam van de tool op de MCP server
            tool_schema: Schema informatie voor de tool
            server_name: Naam van de MCP server voor prefixing
        """
        # Create tool name with server prefix to avoid conflicts
        prefixed_name = f"{server_name}_{tool_name}"

        # Handle args_schema properly - BaseTool expects a BaseModel subclass
        args_schema = tool_schema.get("args_schema")

        # Only pass args_schema if it's provided
        kwargs = {
            "name": prefixed_name,
            "description": tool_schema.get(
                "description", f"Tool {tool_name} from {server_name}"
            ),
        }

        if args_schema is not None:
            kwargs["args_schema"] = args_schema

        super().__init__(**kwargs)

        # Set instance attributes after super().__init__
        self._mcp_server_params = mcp_server_params
        self._original_tool_name = tool_name
        self._server_name = server_name

    @property
    def mcp_server_params(self) -> dict:
        """Haal de MCP server parameters op."""
        return self._mcp_server_params

    @property
    def original_tool_name(self) -> str:
        """Haal de originele tool naam op."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Haal de server naam op."""
        return self._server_name

    def _run(self, **kwargs) -> str:
        """Verbind met MCP server en voer tool uit.

        Args:
            **kwargs: Argumenten om door te geven aan de MCP tool

        Retourneert:
            Resultaat van de MCP tool uitvoering
        """
        try:
            return asyncio.run(self._run_async(**kwargs))
        except asyncio.TimeoutError:
            return f"MCP tool '{self.original_tool_name}' timeout na {MCP_TOOL_EXECUTION_TIMEOUT} seconden"
        except Exception as e:
            return f"Fout bij uitvoeren MCP tool {self.original_tool_name}: {e!s}"

    async def _run_async(self, **kwargs) -> str:
        """Async implementatie van MCP tool uitvoering met timeouts en retry logica."""
        return await self._retry_with_exponential_backoff(
            self._execute_tool_with_timeout, **kwargs
        )

    async def _retry_with_exponential_backoff(self, operation_func, **kwargs) -> str:
        """Herhaal operatie met exponentiële backoff, vermijd try-except in loop voor performance."""
        last_error = None

        for attempt in range(MCP_MAX_RETRIES):
            # Voer enkele poging uit buiten try-except loop structuur
            result, error, should_retry = await self._execute_single_attempt(
                operation_func, **kwargs
            )

            # Succes geval - retourneer direct
            if result is not None:
                return result

            # Niet-herhaalbare fout - retourneer direct
            if not should_retry:
                return error

            # Herhaalbare fout - ga door met backoff
            last_error = error
            if attempt < MCP_MAX_RETRIES - 1:
                wait_time = 2**attempt  # Exponentiële backoff
                await asyncio.sleep(wait_time)

        return (
            f"MCP tool uitvoering mislukt na {MCP_MAX_RETRIES} pogingen: {last_error}"
        )

    async def _execute_single_attempt(
        self, operation_func, **kwargs
    ) -> tuple[str | None, str, bool]:
        """Voer enkele operatie poging uit en retourneer (resultaat, foutmelding, moet_herhalen)."""
        try:
            result = await operation_func(**kwargs)
            return result, "", False

        except ImportError:
            return (
                None,
                "MCP bibliotheek niet beschikbaar. Installeer met: pip install mcp",
                False,
            )

        except asyncio.TimeoutError:
            return (
                None,
                f"Verbinding timeout na {MCP_TOOL_EXECUTION_TIMEOUT} seconden",
                True,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Classificeer fouten als herhaalbaar of niet-herhaalbaar
            if "authentication" in error_str or "unauthorized" in error_str:
                return None, f"Authenticatie mislukt voor MCP server: {e!s}", False
            if "not found" in error_str:
                return (
                    None,
                    f"Tool '{self.original_tool_name}' niet gevonden op MCP server",
                    False,
                )
            if "connection" in error_str or "network" in error_str:
                return None, f"Netwerk verbinding mislukt: {e!s}", True
            if "json" in error_str or "parsing" in error_str:
                return None, f"Server response parsing fout: {e!s}", True
            return None, f"MCP uitvoering fout: {e!s}", False

    async def _execute_tool_with_timeout(self, **kwargs) -> str:
        """Voer tool uit met timeout wrapper."""
        return await asyncio.wait_for(
            self._execute_tool(**kwargs), timeout=MCP_TOOL_EXECUTION_TIMEOUT
        )

    async def _execute_tool(self, **kwargs) -> str:
        """Voer de daadwerkelijke MCP tool aanroep uit."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        server_url = self.mcp_server_params["url"]

        try:
            # Wrap entire operation with single timeout
            async def _do_mcp_call():
                async with streamablehttp_client(
                    server_url, terminate_on_close=True
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            self.original_tool_name, kwargs
                        )

                        # Extract the result content
                        if hasattr(result, "content") and result.content:
                            if (
                                isinstance(result.content, list)
                                and len(result.content) > 0
                            ):
                                content_item = result.content[0]
                                if hasattr(content_item, "text"):
                                    return str(content_item.text)
                                return str(content_item)
                            return str(result.content)
                        return str(result)

            return await asyncio.wait_for(
                _do_mcp_call(), timeout=MCP_TOOL_EXECUTION_TIMEOUT
            )

        except asyncio.CancelledError as e:
            raise asyncio.TimeoutError("MCP operation was cancelled") from e
        except Exception as e:
            if hasattr(e, "__cause__") and e.__cause__:
                raise asyncio.TimeoutError(
                    f"MCP connection error: {e.__cause__}"
                ) from e.__cause__

            if "TaskGroup" in str(e) or "unhandled errors" in str(e):
                raise asyncio.TimeoutError(f"MCP connection error: {e}") from e
            raise
