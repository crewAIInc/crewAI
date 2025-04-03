import uuid
from abc import ABC, abstractmethod
from copy import copy as shallow_copy
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.tools_handler import ToolsHandler
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.security.security_config import SecurityConfig
from crewai.tools.base_tool import BaseTool, Tool
from crewai.utilities import I18N, Logger, RPMController
from crewai.utilities.config import process_config
from crewai.utilities.converter import Converter
from crewai.utilities.string_utils import interpolate_only

T = TypeVar("T", bound="BaseAgent")


class BaseAgent(ABC, BaseModel):
    """Abstract Base Class for all third party agents compatible with CrewAI.

    Attributes:
        id (UUID4): Unique identifier for the agent.
        role (str): Role of the agent.
        goal (str): Objective of the agent.
        backstory (str): Backstory of the agent.
        cache (bool): Whether the agent should use a cache for tool usage.
        config (Optional[Dict[str, Any]]): Configuration for the agent.
        verbose (bool): Verbose mode for the Agent Execution.
        max_rpm (Optional[int]): Maximum number of requests per minute for the agent execution.
        allow_delegation (bool): Allow delegation of tasks to agents.
        tools (Optional[List[Any]]): Tools at the agent's disposal.
        max_iter (int): Maximum iterations for an agent to execute a task.
        agent_executor (InstanceOf): An instance of the CrewAgentExecutor class.
        llm (Any): Language model that will run the agent.
        crew (Any): Crew to which the agent belongs.
        i18n (I18N): Internationalization settings.
        cache_handler (InstanceOf[CacheHandler]): An instance of the CacheHandler class.
        tools_handler (InstanceOf[ToolsHandler]): An instance of the ToolsHandler class.
        max_tokens: Maximum number of tokens for the agent to generate in a response.
        knowledge_sources: Knowledge sources for the agent.
        knowledge_storage: Custom knowledge storage for the agent.
        security_config: Security configuration for the agent, including fingerprinting.


    Methods:
        execute_task(task: Any, context: Optional[str] = None, tools: Optional[List[BaseTool]] = None) -> str:
            Abstract method to execute a task.
        create_agent_executor(tools=None) -> None:
            Abstract method to create an agent executor.
        _parse_tools(tools: List[BaseTool]) -> List[Any]:
            Abstract method to parse tools.
        get_delegation_tools(agents: List["BaseAgent"]):
            Abstract method to set the agents task tools for handling delegation and question asking to other agents in crew.
        get_output_converter(llm, model, instructions):
            Abstract method to get the converter class for the agent to create json/pydantic outputs.
        interpolate_inputs(inputs: Dict[str, Any]) -> None:
            Interpolate inputs into the agent description and backstory.
        set_cache_handler(cache_handler: CacheHandler) -> None:
            Set the cache handler for the agent.
        copy() -> "BaseAgent":
            Create a copy of the agent.
        set_rpm_controller(rpm_controller: RPMController) -> None:
            Set the rpm controller for the agent.
        set_private_attrs() -> "BaseAgent":
            Set private attributes.
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=False))
    _rpm_controller: Optional[RPMController] = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _original_role: Optional[str] = PrivateAttr(default=None)
    _original_goal: Optional[str] = PrivateAttr(default=None)
    _original_backstory: Optional[str] = PrivateAttr(default=None)
    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent", default=None, exclude=True
    )
    cache: bool = Field(
        default=True, description="Whether the agent should use a cache for tool usage."
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    allow_delegation: bool = Field(
        default=False,
        description="Enable agent to delegate and ask questions among each other.",
    )
    tools: Optional[List[BaseTool]] = Field(
        default_factory=list, description="Tools at agents' disposal"
    )
    max_iter: int = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )
    agent_executor: InstanceOf = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    llm: Any = Field(
        default=None, description="Language model that will run the agent."
    )
    crew: Any = Field(default=None, description="Crew to which the agent belongs.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default_factory=ToolsHandler,
        description="An instance of the ToolsHandler class.",
    )
    tools_results: List[Dict[str, Any]] = Field(
        default=[], description="Results of the tools used by the agent."
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens for the agent's execution."
    )
    knowledge: Optional[Knowledge] = Field(
        default=None, description="Knowledge for the agent."
    )
    knowledge_sources: Optional[List[BaseKnowledgeSource]] = Field(
        default=None,
        description="Knowledge sources for the agent.",
    )
    knowledge_storage: Optional[Any] = Field(
        default=None,
        description="Custom knowledge storage for the agent.",
    )
    security_config: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration for the agent, including fingerprinting.",
    )
    callbacks: List[Callable] = Field(
        default=[], description="Callbacks to be used for the agent"
    )

    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values):
        return process_config(values, cls)

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools: List[Any]) -> List[BaseTool]:
        """Validate and process the tools provided to the agent.

        This method ensures that each tool is either an instance of BaseTool
        or an object with 'name', 'func', and 'description' attributes. If the
        tool meets these criteria, it is processed and added to the list of
        tools. Otherwise, a ValueError is raised.
        """
        processed_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed_tools.append(tool)
            elif (
                hasattr(tool, "name")
                and hasattr(tool, "func")
                and hasattr(tool, "description")
            ):
                # Tool has the required attributes, create a Tool instance
                processed_tools.append(Tool.from_langchain(tool))
            else:
                raise ValueError(
                    f"Invalid tool type: {type(tool)}. "
                    "Tool must be an instance of BaseTool or "
                    "an object with 'name', 'func', and 'description' attributes."
                )
        return processed_tools

    @model_validator(mode="after")
    def validate_and_set_attributes(self):
        # Validate required fields
        for field in ["role", "goal", "backstory"]:
            if getattr(self, field) is None:
                raise ValueError(
                    f"{field} must be provided either directly or through config"
                )

        # Set private attributes
        self._logger = Logger(verbose=self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        if not self._token_process:
            self._token_process = TokenProcess()

        # Initialize security_config if not provided
        if self.security_config is None:
            self.security_config = SecurityConfig()

        return self

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(verbose=self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        if not self._token_process:
            self._token_process = TokenProcess()
        return self

    @property
    def key(self):
        source = [
            self._original_role or self.role,
            self._original_goal or self.goal,
            self._original_backstory or self.backstory,
        ]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

    @abstractmethod
    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        pass

    @abstractmethod
    def create_agent_executor(self, tools=None) -> None:
        pass

    @abstractmethod
    def get_delegation_tools(self, agents: List["BaseAgent"]) -> List[BaseTool]:
        """Set the task tools that init BaseAgenTools class."""
        pass

    @abstractmethod
    def get_output_converter(
        self, llm: Any, text: str, model: type[BaseModel] | None, instructions: str
    ) -> Converter:
        """Get the converter class for the agent to create json/pydantic outputs."""
        pass

    def copy(self: T) -> T:  # type: ignore # Signature of "copy" incompatible with supertype "BaseModel"
        """Create a deep copy of the Agent."""
        exclude = {
            "id",
            "_logger",
            "_rpm_controller",
            "_request_within_rpm_limit",
            "_token_process",
            "agent_executor",
            "tools",
            "tools_handler",
            "cache_handler",
            "llm",
            "knowledge_sources",
            "knowledge_storage",
            "knowledge",
        }

        # Copy llm
        existing_llm = shallow_copy(self.llm)
        copied_knowledge = shallow_copy(self.knowledge)
        copied_knowledge_storage = shallow_copy(self.knowledge_storage)
        # Properly copy knowledge sources if they exist
        existing_knowledge_sources = None
        if self.knowledge_sources:
            # Create a shared storage instance for all knowledge sources
            shared_storage = (
                self.knowledge_sources[0].storage if self.knowledge_sources else None
            )

            existing_knowledge_sources = []
            for source in self.knowledge_sources:
                copied_source = (
                    source.model_copy()
                    if hasattr(source, "model_copy")
                    else shallow_copy(source)
                )
                # Ensure all copied sources use the same storage instance
                copied_source.storage = shared_storage
                existing_knowledge_sources.append(copied_source)

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}
        copied_agent = type(self)(
            **copied_data,
            llm=existing_llm,
            tools=self.tools,
            knowledge_sources=existing_knowledge_sources,
            knowledge=copied_knowledge,
            knowledge_storage=copied_knowledge_storage,
        )

        return copied_agent

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = interpolate_only(
                input_string=self._original_role, inputs=inputs
            )
            self.goal = interpolate_only(
                input_string=self._original_goal, inputs=inputs
            )
            self.backstory = interpolate_only(
                input_string=self._original_backstory, inputs=inputs
            )

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.tools_handler = ToolsHandler()
        if self.cache:
            self.cache_handler = cache_handler
            self.tools_handler.cache = cache_handler
        self.create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    def set_knowledge(self, crew_embedder: Optional[Dict[str, Any]] = None):
        pass
