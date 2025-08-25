import json

import boto3
from typing import Any, Dict, List
import datetime

from crewai.memory.storage.interface import Storage


class AgentCoreStorage(Storage):
    """
    A simple in-memory implementation mimicking AWS Bedrock AgentCore Memory Events API.
    This implementation helps understand what data is saved by CrewAI before implementing
    the actual AWS Bedrock AgentCore integration.
    """

    def __init__(self, type, crew=None, config=None):
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew

        # Handle configuration
        self.config = (
            config or getattr(crew, "memory_config", {}).get("config", {}) or {}
        )

        self._extract_config_values()

        self.client = boto3.client("bedrock-agentcore", region_name=self.region_name)

    def _validate_type(self, type):
        supported_types = {"external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for AgentCoreStorage. Must be one of: {', '.join(supported_types)}"
            )

    def _extract_config_values(self):
        """Extract configuration values from config dictionary."""
        cfg = self.config
        self.agent_core_run_id = cfg.get("run_id")
        self.agent_id = cfg.get("agent_id", self._get_agent_name())
        self.user_id = cfg.get("user_id", "")

        # AgentCore specific values
        self.actor_id = cfg.get("actor_id")
        self.session_id = cfg.get("session_id")
        self.memory_id = cfg.get("memory_id")
        self.region_name = cfg.get("region_name", "us-west-2")
        self.strategies = cfg.get("strategies", [])

        print(
            f"AgentCoreStorage initialized with: memory_type={self.memory_type}, "
            f"agent_id={self.agent_id}, user_id={self.user_id}, run_id={self.agent_core_run_id}"
        )
        print(
            f"AgentCore API values: actor_id={self.actor_id}, session_id={self.session_id}, memory_id={self.memory_id}"
        )

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save memory item to in-memory storage.

        Args:
            value (Any): The content to save
            metadata (Dict[str, Any]): Metadata for the memory item
        """
        print("\n=== AgentCore Storage: save() INPUTS ===")
        print(f"Value type: {type(value)}")
        print(f"Value: {value}")
        print(f"Metadata: {metadata}")
        print("===================================\n")

        # Create event with unique ID
        timestamp = datetime.datetime.now(datetime.timezone.utc)

        content = {"value": value, "metadata": metadata}
        content_str = json.dumps(content, default=str)

        params = {
            "memoryId": self.memory_id,
            "actorId": self.actor_id,
            "eventTimestamp": timestamp,
            "sessionId": self.session_id,
            "payload": [{
                "conversational": {
                    "content": {
                        "text": content_str
                    },
                    "role": "ASSISTANT",
                }
            }],
        }

        response = self.client.create_event(**params)
        event = response["event"]
        print(f"Created event: {event['eventId']}")

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.35
    ) -> List[Any]:
        """
        Search for memory items in our in-memory storage.
        In a real implementation, this would use AgentCore's search capabilities.

        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            score_threshold (float): Minimum relevance score for results

        Returns:
            List[Dict]: List of matching memory items
        """
        print("\n=== AgentCore Storage: search() INPUTS ===")
        print(f"Query: {query}")
        print(f"Limit: {limit}")
        print(f"Score threshold: {score_threshold}")
        print(f"Memory type: {self.memory_type}")
        print(f"User ID: {self.user_id}")
        print(f"Agent ID: {self.agent_id}")
        print(f"Run ID: {self.agent_core_run_id}")
        print("=======================================\n")
        # For now, just return all events that match the memory type
        # In a real implementation, we would use actual search capabilities

        # Filter by memory type and other criteria
        filtered_events = []

        all_events = []
        next_token = None

        params = {
            "memoryId": self.memory_id,
            "actorId": self.actor_id,
            "sessionId": self.session_id,
            "maxResults": min(100, limit - len(all_events)),
        }
        while len(all_events) < limit:
            if next_token:
                params["nextToken"] = next_token

            response = self.client.list_events(**params)
            events = response.get("events", [])
            all_events.extend(events)
            next_token = response.get("nextToken")
            if not next_token or len(all_events) >= limit:
                break

        for event in all_events[:limit]:
            for payload in event["payload"]:
                memory = payload["conversational"]["content"]["text"]
                result = {
                    "memory": memory
                }
                filtered_events.append(result)
        
        filtered_events.reverse()
        
        return filtered_events

    def reset(self) -> None:
        """
        Clear all memory items.
        """
        pass

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _get_agent_name(self) -> str:
        """
        Get agent name from crew configuration.
        """
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        # Limit agent name length to prevent issues
        if len(agents) > 255:
            agents = agents[:255]
        return agents
