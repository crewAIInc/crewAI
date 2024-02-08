import json
import os
import platform
import socket

import pkg_resources
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


class Telemetry:
    """A class to handle anonymous telemetry for the crewai package.

    The data being collected is for development purpose, all data is anonymous.

    There is NO data being collected on the prompts, tasks descriptions
    agents backstories or goals nor responses or any data that is being
    processed by the agents, nor any secrets and env vars.

    Data collected includes:
    - Version of crewAI
    - Version of Python
    - General OS (e.g. number of CPUs, macOS/Windows/Linux)
    - Number of agents and tasks in a crew
    - Crew Process being used
    - If Agents are using memory or allowing delegation
    - If Tasks are being executed in parallel or sequentially
    - Language model being used
    - Roles of agents in a crew
    - Tools names available
    """

    def __init__(self):
        telemetry_endpoint = "http://telemetry.crewai.com:4318"
        self.resource = Resource(attributes={SERVICE_NAME: "crewAI-telemetry"})
        provider = TracerProvider(resource=self.resource)
        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{telemetry_endpoint}/v1/traces")
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    def crew_creation(self, crew):
        """Records the creation of a crew."""
        try:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Created")
            self.add_attribute(
                span, "crewai_version", pkg_resources.get_distribution("crewai").version
            )
            self.add_attribute(span, "python_version", platform.python_version())
            self.add_attribute(span, "hostname", socket.gethostname())
            self.add_attribute(span, "crewid", str(crew.id))
            self.add_attribute(span, "crew_process", crew.process)
            self.add_attribute(span, "crew_language", crew.language)
            self.add_attribute(span, "crew_number_of_tasks", len(crew.tasks))
            self.add_attribute(span, "crew_number_of_agents", len(crew.agents))
            self.add_attribute(
                span,
                "crew_agents",
                json.dumps(
                    [
                        {
                            "id": str(agent.id),
                            "role": agent.role,
                            "memory_enabled?": agent.memory,
                            "llm": json.dumps(self._safe_llm_attributes(agent.llm)),
                            "delegation_enabled?": agent.allow_delegation,
                            "tools_names": [tool.name for tool in agent.tools],
                        }
                        for agent in crew.agents
                    ]
                ),
            )
            self.add_attribute(
                span,
                "crew_tasks",
                json.dumps(
                    [
                        {
                            "id": str(task.id),
                            "async_execution?": task.async_execution,
                            "tools_names": [tool.name for tool in task.tools],
                        }
                        for task in crew.tasks
                    ]
                ),
            )
            self.add_attribute(span, "platform", platform.platform())
            self.add_attribute(span, "platform_release", platform.release())
            self.add_attribute(span, "platform_system", platform.system())
            self.add_attribute(span, "platform_version", platform.version())
            self.add_attribute(span, "cpus", os.cpu_count())
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception:
            pass

    def add_attribute(self, span, key, value):
        """Add an attribute to a span."""
        try:
            return span.set_attribute(key, value)
        except Exception:
            pass

    def _safe_llm_attributes(self, llm):
        attributes = ["name", "model_name", "base_url", "model", "top_k", "temperature"]
        safe_attributes = {k: v for k, v in vars(llm).items() if k in attributes}
        safe_attributes["class"] = llm.__class__.__name__
        return safe_attributes
