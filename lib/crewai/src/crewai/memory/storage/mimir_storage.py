import os
import shutil
import subprocess
import json
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple

from crewai.memory.storage.backend import StorageBackend
# CodeRabbit Fix: Direct import to fail-fast and avoid masking integration issues
from crewai.memory.storage.interface import MemoryRecord

logger = logging.getLogger(__name__)

class MimirStorage(StorageBackend):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Resolve db_path from config dictionary, expanding '~' to home directory
        raw_db_path = self.config.get("db_path", "~/mimir_db")
        self.db_path = os.path.expanduser(raw_db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        # Verify mimir binary availability in common paths or system PATH
        self.mimir_path = self._find_mimir_binary()
        if not self.mimir_path:
            raise FileNotFoundError(
                "The 'mimir' binary could not be found. Please ensure it is installed "
                "and available in PATH or at common locations (~/.cargo/bin/mimir, /usr/local/bin/mimir)."
            )

    def _find_mimir_binary(self) -> Optional[str]:
        """Checks common paths and system PATH for the mimir binary."""
        path_binary = shutil.which("mimir")
        if path_binary:
            return path_binary
            
        common_paths = [
            os.path.expanduser("~/.cargo/bin/mimir"),
            "/usr/local/bin/mimir",
            "/usr/bin/mimir"
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _validate_inputs(self, category: str, query: Optional[str] = None) -> None:
        """Validates input arguments to safeguard against CLI/flag injection attacks."""
        if category and not re.match(r"^[A-Za-z0-9_-]+$", category):
            raise ValueError(f"Malicious characters detected in scope/category: '{category}'")
        if query and query.startswith("-"):
            raise ValueError("Query string cannot start with a hyphen to prevent flag injection.")

    def save(self, records: List[MemoryRecord]) -> None:
        """Saves a list of MemoryRecords conforming to the StorageBackend protocol."""
        for record in records:
            value_str = str(record.value)
            
            # Generate a persistent deterministic hash key using hashlib MD5
            hash_suffix = hashlib.md5(value_str.encode('utf-8')).hexdigest()[:12]
            key = f"memory_{hash_suffix}"
            
            # Scope memories using config metadata or default category
            category = record.metadata.get("agent_id", "default")
            self._validate_inputs(category)
            
            # Prepare payload
            payload = {
                "key": key,
                "value": value_str,
                "category": category,
                "metadata": record.metadata
            }
            
            # Call the subprocess using '--db' flag per Mimir CLI docs (with 10s timeout)
            try:
                cmd = [self.mimir_path, "--db", self.db_path, "store", json.dumps(payload)]
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
                logger.info(f"Successfully stored memory with key: {key}")
            except subprocess.TimeoutExpired as te:
                logger.error(f"Mimir store operation timed out: {te}")
                raise te
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to store memory in Mimir: {e.stderr}")
                raise e

    def search(
        self, 
        query: Any, 
        limit: Optional[int] = None,          
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        min_score: Optional[float] = None,   
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[MemoryRecord, float]]:
        """Searches memories and returns a list of (MemoryRecord, score) tuples."""
        query_str = query if isinstance(query, str) else str(query)
        
        actual_limit = limit if limit is not None else 3
        
        category = scope_prefix if scope_prefix else "default"
        
        if categories and len(categories) > 0:
            category = categories[0]
            
        self._validate_inputs(category, query_str)
        
        try:
            cmd = [
                self.mimir_path, 
                "--db", self.db_path, 
                "search", 
                query_str, 
                "--limit", str(actual_limit),  
                "--category", category
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            
            raw_results = json.loads(result.stdout)
            formatted_results = []
            
            for res in raw_results:
                content_text = res.get("value", res.get("text", ""))
                score = float(res.get("score", 0.0))
                meta = res.get("metadata", {})
                
                if min_score is not None and score < min_score:
                    continue
                
                if metadata_filter:
                    match = True
                    for k, v in metadata_filter.items():
                        if meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                
                # Construct official MemoryRecord instances
                record = MemoryRecord(value=content_text, metadata=meta)
                formatted_results.append((record, score))
                
            return formatted_results
        except subprocess.TimeoutExpired as te:
            logger.error(f"Mimir search operation timed out: {te}")
            raise te
        except subprocess.CalledProcessError as e:
            logger.error(f"Search failed in Mimir: {e.stderr}")
            raise e