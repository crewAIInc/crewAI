import logging
from typing import Any, Dict, List, Optional
from crewai.memory.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

class MimirStorage(StorageBackend):
    """Storage backend powered by the official mimir-client SDK."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        try:
            from mimir_client import MimirClient
        except ImportError:
            raise ImportError(
                "The 'mimir-client' package is required to use MimirStorage. "
                "Please install it using: pip install mimir-client"
            )
        
        self.config = config or {}
        self.client = MimirClient(**self.config)

    # Diventa ASYNC per supportare l'SDK di Mimir (Fix Immagine 4)
    async def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        """Saves a value to the Mimir storage using async artifact creation."""
        clean_metadata = dict(metadata) if metadata else {}
        if agent:
            clean_metadata["agent"] = agent

        try:
            # Passiamo i parametri corretti richiesti esplicitamente dall'SDK (Fix Immagine 4)
            await self.client.create_artifact(
                artifact_type="memory",
                content=str(value),
                metadata=clean_metadata
            )
        except Exception as e:
            logger.error(f"Error saving to MimirStorage: {e}")
            raise e

    def search(self, query: str, limit: int = 3, filter: Optional[Dict[str, Any]] = None, score_threshold: float = 0.35) -> List[Any]:
        """Searches the Mimir storage using full-text search."""
        if filter:
            raise NotImplementedError("Advanced filtering is not currently supported in MimirStorage search.")

        try:
            # Usiamo search_fulltext per query testuali dirette senza bisogno di embedding (Fix Immagine 5)
            results = self.client.search_fulltext(query=query, limit=limit)
            
            formatted_results = []
            for res in results:
                if hasattr(res, 'score') and res.score < score_threshold:
                    continue
                formatted_results.append(getattr(res, 'content', str(res)))
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in MimirStorage: {e}")
            # Rilanciamo l'errore per non mascherare i fallimenti (Fix Immagine 6)
            raise e

    def delete(self, key: str, filter: Optional[Dict[str, Any]] = None) -> int:
        """Deletes entries from Mimir storage and returns the deleted count."""
        if filter and any(k for k in filter if k != "record_ids"):
            raise NotImplementedError(
                "MimirStorage.delete() currently only supports deletion by 'record_ids'."
            )

        deleted_count = 0
        try:
            record_ids = filter.get("record_ids") if filter else [key]
            if record_ids:
                for r_id in record_ids:
                    self.client.delete_artifact(artifact_id=r_id)
                    deleted_count += 1
            
            # Restituiamo un intero come richiesto dal protocollo StorageBackend (Fix Immagine 7)
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting from MimirStorage: {e}")
            raise e