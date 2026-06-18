import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from crewai.memory.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

class MimirStorage(StorageBackend):
    """Storage backend powered by Mimir using the official MCP Python SDK via Stdio."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        try:
            from mcp import StdioServerParameters
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required to use MimirStorage with MCP. "
                "Please install it using: pip install mcp"
            )
        
        self.config = config or {}
        
        # Recuperiamo il percorso del database locale (default su ~/.mimir/mimir.db)
        db_path = self.config.get("db_path", "~/.mimir/mimir.db")
        
        # Configurazione corretta dei parametri del server per l'avvio del subprocesso
        self.server_params = StdioServerParameters(
            command="mimir",
            args=["--db-path", db_path]
        )

    async def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Helper asincrono per connettersi al server MCP ed eseguire un tool di Mimir."""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                return result

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        """Saves a value to the Mimir storage using the 'mimir_remember' MCP tool."""
        clean_metadata = dict(metadata) if metadata else {}
        
        body_data = {
            "value": str(value),
            "metadata": clean_metadata,
        }
        if agent:
            body_data["agent"] = agent

        try:
            # Esecuzione del ciclo asincrono per lo strumento mimir_remember
            asyncio.run(self._call_tool("mimir_remember", {
                "category": "crewai_memory",
                "key": f"memory_{hash(value)}",
                "body_json": json.dumps(body_data)
            }))
        except Exception as e:
            logger.error(f"Error saving to MimirStorage via MCP: {e}")
            raise e

    def search(self, query: str, limit: int = 3, filter: Optional[Dict[str, Any]] = None, score_threshold: float = 0.35) -> List[Any]:
        """Searches the Mimir storage using the 'mimir_recall' MCP tool."""
        if filter:
            raise NotImplementedError("Advanced filtering is not currently supported in MimirStorage search.")

        try:
            # Chiamata al tool di ricerca semantica di Mimir
            mcp_result = asyncio.run(self._call_tool("mimir_recall", {
                "query": query,
                "limit": limit
            }))
            
            # Il protocollo MCP restituisce tipicamente una lista o un oggetto Content.
            # Gestiamo il parsing dei risultati in base alla struttura restituita dal tool.
            results = mcp_result if isinstance(mcp_result, list) else getattr(mcp_result, 'content', [])
            
            formatted_results = []
            for res in results:
                # Applichiamo il filtro sulla confidenza (confidence decay) come richiesto
                score = getattr(res, 'score', 1.0)
                if score < score_threshold:
                    continue
                
                content_text = getattr(res, 'text', str(res))
                formatted_results.append(content_text)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in MimirStorage via MCP: {e}")
            raise e

    def delete(self, key: str, filter: Optional[Dict[str, Any]] = None) -> int:
        """Deletes entries from Mimir storage using the 'mimir_forget' MCP tool."""
        if filter and any(k for k in filter if k != "record_ids"):
            raise NotImplementedError(
                "MimirStorage.delete() currently only supports deletion by 'record_ids'."
            )

        deleted_count = 0
        try:
            record_ids = filter.get("record_ids") if filter else [key]
            if record_ids:
                for r_id in record_ids:
                    # Chiamata al tool di rimozione ufficiale
                    asyncio.run(self._call_tool("mimir_forget", {
                        "id": r_id
                    }))
                    deleted_count += 1
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting from MimirStorage via MCP: {e}")
            raise e