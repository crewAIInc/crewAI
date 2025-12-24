"""Native MCP tool wrapper voor CrewAI agents.

Deze module biedt een tool wrapper die bestaande MCP client sessies hergebruikt
voor betere performance en verbindingsbeheer.
"""

import asyncio
from typing import Any

from crewai.tools import BaseTool


class MCPNativeTool(BaseTool):
    """Native MCP tool die client sessies hergebruikt.

    Deze tool wrapper wordt gebruikt wanneer agents verbinden met MCP servers via
    gestructureerde configuraties. Het hergebruikt bestaande client sessies voor
    betere performance en correct verbinding lifecycle beheer.

    In tegenstelling tot MCPToolWrapper die on-demand verbindt, gebruikt deze tool
    een gedeelde MCP client instantie die een persistente verbinding onderhoudt.
    """

    def __init__(
        self,
        mcp_client: Any,
        tool_name: str,
        tool_schema: dict[str, Any],
        server_name: str,
    ) -> None:
        """Initialiseer native MCP tool.

        Args:
            mcp_client: MCPClient instantie met actieve sessie.
            tool_name: Originele naam van de tool op de MCP server.
            tool_schema: Schema informatie voor de tool.
            server_name: Naam van de MCP server voor prefixing.
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
        self._mcp_client = mcp_client
        self._original_tool_name = tool_name
        self._server_name = server_name
        # self._logger = logging.getLogger(__name__)

    @property
    def mcp_client(self) -> Any:
        """Haal de MCP client instantie op."""
        return self._mcp_client

    @property
    def original_tool_name(self) -> str:
        """Haal de originele tool naam op."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Haal de server naam op."""
        return self._server_name

    def _run(self, **kwargs) -> str:
        """Voer tool uit met de MCP client sessie.

        Args:
            **kwargs: Argumenten om door te geven aan de MCP tool.

        Retourneert:
            Resultaat van de MCP tool uitvoering.
        """
        try:
            try:
                asyncio.get_running_loop()

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    coro = self._run_async(**kwargs)
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                return asyncio.run(self._run_async(**kwargs))

        except Exception as e:
            raise RuntimeError(
                f"Fout bij uitvoeren MCP tool {self.original_tool_name}: {e!s}"
            ) from e

    async def _run_async(self, **kwargs) -> str:
        """Async implementatie van tool uitvoering.

        Args:
            **kwargs: Argumenten om door te geven aan de MCP tool.

        Retourneert:
            Resultaat van de MCP tool uitvoering.
        """
        # Opmerking: Omdat we asyncio.run() gebruiken die elke keer een nieuwe event loop maakt,
        # altijd opnieuw verbinden on-demand omdat asyncio.run() nieuwe event loops per aanroep maakt
        # Alle MCP transport context managers (stdio, streamablehttp_client, sse_client)
        # gebruiken anyio.create_task_group() die niet over verschillende event loops kan
        if self._mcp_client.connected:
            await self._mcp_client.disconnect()

        await self._mcp_client.connect()

        try:
            result = await self._mcp_client.call_tool(self.original_tool_name, kwargs)

        except Exception as e:
            error_str = str(e).lower()
            if (
                "not connected" in error_str
                or "connection" in error_str
                or "send" in error_str
            ):
                await self._mcp_client.disconnect()
                await self._mcp_client.connect()
                # Retry the call
                result = await self._mcp_client.call_tool(
                    self.original_tool_name, kwargs
                )
            else:
                raise

        finally:
            # Altijd verbinding verbreken na tool aanroep voor schone context manager lifecycle
            # Dit voorkomt "exit cancel scope in different task" fouten
            # Alle transport context managers moeten worden afgesloten in dezelfde event loop waarin ze zijn gestart
            await self._mcp_client.disconnect()

        # Extraheer resultaat inhoud
        if isinstance(result, str):
            return result

        # Verwerk verschillende resultaat formaten
        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return str(content_item.text)
                return str(content_item)
            return str(result.content)

        return str(result)
