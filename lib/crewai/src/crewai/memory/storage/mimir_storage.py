import os
import shutil
import subprocess
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MimirStorage:
    def __init__(self, db_path: str = "~/mimir_db"):
        # Resolve db_path, expanding '~' to home directory
        self.db_path = os.path.expanduser(db_path)
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
        # 1. Check system PATH
        path_binary = shutil.which("mimir")
        if path_binary:
            return path_binary
            
        # 2. Check common installation paths
        common_paths = [
            os.path.expanduser("~/.cargo/bin/mimir"),
            "/usr/local/bin/mimir",
            "/usr/bin/mimir"
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None) -> None:
        # Generate a persistent deterministic hash key using hashlib MD5
        value_str = str(value)
        hash_suffix = hashlib.md5(value_str.encode('utf-8')).hexdigest()[:12]
        key = f"memory_{hash_suffix}"
        
        # Scope memories by agent or config category if agent_id is provided
        category = agent_id if agent_id else "default"
        
        # Prepare payload
        payload = {
            "key": key,
            "value": value_str,
            "category": category,
            "metadata": metadata or {}
        }
        
        # Call the subprocess using '--db' flag per Mimir CLI docs
        try:
            cmd = [self.mimir_path, "--db", self.db_path, "store", json.dumps(payload)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully stored memory with key: {key}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to store memory in Mimir: {e.stderr}")
            raise e

    def search(self, query: str, limit: int = 3, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        category = agent_id if agent_id else "default"
        
        try:
            cmd = [
                self.mimir_path, 
                "--db", self.db_path, 
                "search", 
                query, 
                "--limit", str(limit), 
                "--category", category
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            raw_results = json.loads(result.stdout)
            formatted_results = []
            
            # Map raw outputs to structured results with text, score, and metadata
            for res in raw_results:
                # Fallback to empty dict or default scores if fields are missing dynamically
                content_text = res.get("value", res.get("text", ""))
                score = res.get("score", 0.0)
                meta = res.get("metadata", {})
                
                formatted_results.append({
                    "text": content_text,
                    "score": score,
                    "metadata": meta
                })
                
            return formatted_results
        except subprocess.CalledProcessError as e:
            logger.error(f"Search failed in Mimir: {e.stderr}")
            return []