import logging
from typing import Any, Dict, List, Optional
from crewai.memory.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

class MimirStorage(StorageBackend):
    """Storage backend powered by the official mimir-client SDK."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Importiamo il client reale qui dentro per sicurezza
        try:
            from mimir_client import MimirClient
        except ImportError:
            raise ImportError(
                "The 'mimir-client' package is required to use MimirStorage. "
                "Please install it using: pip install mimir-client"
            )
        
        self.config = config or {}
        # Inizializziamo il client ufficiale di Mimir
        self.client = MimirClient(**self.config)

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        """Saves a value to the Mimir storage using artifact creation."""
        # Creiamo una copia pulita dei metadati per evitare mutazioni in-place (Fix Immagine 3)
        clean_metadata = dict(metadata) if metadata else {}
        if agent:
            clean_metadata["agent"] = agent

        payload = {
            "text": str(value),
            "metadata": clean_metadata
        }

        # L'SDK di Mimir usa la creazione di artifact/documenti per salvare la memoria
        try:
            self.client.create_artifact(payload=payload)
        except Exception as e:
            logger.error(f"Error saving to MimirStorage: {e}")
            raise e

    def search(self, query: str, limit: int = 3, filter: Optional[Dict[str, Any]] = None, score_threshold: float = 0.35) -> List[Any]:
        """Searches the Mimir storage using semantic vector search."""
        # Se l'utente richiede filtri complessi non supportati, lanciamo un errore chiaro
        if filter:
            raise NotImplementedError("Advanced filtering is not currently supported in MimirStorage search.")

        try:
            # L'SDK richiede una ricerca semantica basata su vettori o testo (Fix Immagine 4)
            # Nota: A seconda della configurazione, Mimir estrae internamente il vettore dalla query testuale
            results = self.client.search_semantic(query_text=query, limit=limit)
            
            # Formattiamo i risultati per l'interfaccia di CrewAI
            formatted_results = []
            for res in results:
                # Filtriamo in base allo score se presente
                if hasattr(res, 'score') and res.score < score_threshold:
                    continue
                formatted_results.append(getattr(res, 'text', str(res)))
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in MimirStorage: {e}")
            return []

    def delete(self, key: str, filter: Optional[Dict[str, Any]] = None) -> None:
        """Deletes entries from Mimir storage."""
        # Se l'utente passa filtri che non possiamo gestire, blocchiamo esplicitamente (Fix Immagine 5)
        if filter and any(k for k in filter if k != "record_ids"):
            raise NotImplementedError(
                "MimirStorage.delete() currently only supports deletion by 'record_ids'."
            )

        try:
            # Utilizziamo l'eliminazione nativa tramite ID dell'artifact
            record_ids = filter.get("record_ids") if filter else [key]
            if record_ids:
                for r_id in record_ids:
                    self.client.delete_artifact(artifact_id=r_id)
        except Exception as e:
            logger.error(f"Error deleting from MimirStorage: {e}")
            raise e