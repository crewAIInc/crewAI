from __future__ import annotations

from concurrent.futures import Future
from copy import copy as shallow_copy
import datetime
from hashlib import md5
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import (
    Any,
    ClassVar,
    cast,
    get_args,
    get_origin,
)
import uuid
import warnings

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.security import Fingerprint, SecurityConfig
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.tools.base_tool import BaseTool
from crewai.utilities.config import process_config
from crewai.utilities.constants import NOT_SPECIFIED, _NotSpecified
from crewai.utilities.converter import Converter, convert_to_model
from crewai.utilities.guardrail import (
    process_guardrail,
)
from crewai.utilities.guardrail_types import (
    GuardrailCallable,
    GuardrailType,
    GuardrailsType,
)
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import interpolate_only


_printer = Printer()


class Task(BaseModel):
    """Klasse die een uit te voeren taak representeert.

    Elke taak moet minimaal een beschrijving hebben. De agent voert acties uit met tools
    om de taak te voltooien.

    Attributen:
        agent: Agent verantwoordelijk voor taakuitvoering. Representeert de entiteit die de taak uitvoert.
        async_execution: Boolean vlag die asynchrone taakuitvoering aangeeft.
        action_based: Of de taak actie-georiënteerd is (tools uitvoeren) of tekst-georiënteerd.
        callback: Functie/object uitgevoerd na taakafronding voor aanvullende acties.
        config: Dictionary met taak-specifieke configuratie parameters.
        context: Lijst van Task instanties die taakcontext of inputdata leveren.
        description: Beschrijvende tekst die doel en uitvoering van de taak detailleert.
        expected_output: Optionele definitie van verwachte taakuitkomst (alleen voor tekst-taken).
        output_file: Bestandspad voor opslag van taakoutput.
        create_directory: Of de directory voor output_file aangemaakt moet worden als deze niet bestaat.
        output_json: Pydantic model voor structurering van JSON output.
        output_pydantic: Pydantic model voor taakoutput.
        security_config: Beveiligingsconfiguratie inclusief fingerprinting.
        tools: Lijst van tools/resources beperkt voor taakuitvoering.
        allow_crewai_trigger_context: Optionele vlag om crewai_trigger_payload injectie te controleren.
                              None (standaard): Auto-injecteer alleen voor eerste taak.
                              True: Injecteer altijd trigger payload voor deze taak.
                              False: Injecteer nooit trigger payload, zelfs niet voor eerste taak.
    """

    __hash__ = object.__hash__
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    executed_actions: list[dict[str, Any]] = Field(default_factory=list)
    i18n: I18N = Field(default_factory=get_i18n)
    name: str | None = Field(default=None)
    prompt_context: str | None = None
    description: str = Field(description="Beschrijving van de daadwerkelijke taak.")
    expected_output: str | None = Field(
        default=None,
        description="Optionele definitie van verwachte output (alleen voor tekst-taken)."
    )
    action_based: bool = Field(
        default=True,
        description="Of de taak actie-georiënteerd is (tools uitvoeren) of tekst-georiënteerd."
    )
    config: dict[str, Any] | None = Field(
        description="Configuratie voor de agent",
        default=None,
    )
    callback: Any | None = Field(
        description="Callback om uit te voeren nadat de taak is voltooid.", default=None
    )
    agent: BaseAgent | None = Field(
        description="Agent verantwoordelijk voor uitvoering van de taak.", default=None
    )
    context: list[Task] | None | _NotSpecified = Field(
        description="Andere taken waarvan de output als context voor deze taak wordt gebruikt.",
        default=NOT_SPECIFIED,
    )
    async_execution: bool | None = Field(
        description="Of de taak asynchroon moet worden uitgevoerd of niet.",
        default=False,
    )
    output_json: type[BaseModel] | None = Field(
        description="Een Pydantic model om een JSON output te maken.",
        default=None,
    )
    output_pydantic: type[BaseModel] | None = Field(
        description="Een Pydantic model om een Pydantic output te maken.",
        default=None,
    )
    response_model: type[BaseModel] | None = Field(
        description="Een Pydantic model voor gestructureerde LLM outputs met native provider features.",
        default=None,
    )
    output_file: str | None = Field(
        description="Een bestandspad om een bestandsoutput te maken.",
        default=None,
    )
    create_directory: bool | None = Field(
        description="Of de directory voor output_file aangemaakt moet worden als deze niet bestaat.",
        default=True,
    )
    output: TaskOutput | None = Field(
        description="Taakoutput, het uiteindelijke resultaat na uitvoering", default=None
    )
    tools: list[BaseTool] | None = Field(
        default_factory=list,
        description="Tools waar de agent beperkt is tot voor deze taak.",
    )
    security_config: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Beveiligingsconfiguratie voor de taak.",
    )
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unieke identifier voor het object, niet ingesteld door gebruiker.",
    )
    human_input: bool | None = Field(
        description="Of de taak een menselijke review van het eindantwoord van de agent moet hebben",
        default=False,
    )
    markdown: bool | None = Field(
        description="Of de taak de agent moet instrueren om het eindantwoord geformatteerd in Markdown te retourneren",
        default=False,
    )
    converter_cls: type[Converter] | None = Field(
        description="Een converter klasse gebruikt om gestructureerde output te exporteren",
        default=None,
    )
    processed_by_agents: set[str] = Field(default_factory=set)
    guardrail: GuardrailType | None = Field(
        default=None,
        description="Functie of string beschrijving van een guardrail om taakoutput te valideren voordat naar volgende taak wordt gegaan",
    )
    guardrails: GuardrailsType | None = Field(
        default=None,
        description="Lijst van guardrails om taakoutput te valideren voordat naar volgende taak wordt gegaan. Ondersteunt ook een enkele guardrail functie of string beschrijving",
    )

    max_retries: int | None = Field(
        default=None,
        description="[VEROUDERD] Maximaal aantal herhalingen wanneer guardrail faalt. Gebruik guardrail_max_retries in plaats daarvan. Wordt verwijderd in v1.0.0",
    )
    guardrail_max_retries: int = Field(
        default=3, description="Maximaal aantal herhalingen wanneer guardrail faalt"
    )
    retry_count: int = Field(default=0, description="Huidig aantal herhalingen")
    start_time: datetime.datetime | None = Field(
        default=None, description="Starttijd van de taakuitvoering"
    )
    end_time: datetime.datetime | None = Field(
        default=None, description="Eindtijd van de taakuitvoering"
    )
    allow_crewai_trigger_context: bool | None = Field(
        default=None,
        description="Of deze taak 'Trigger Payload: {crewai_trigger_payload}' moet toevoegen aan de taakbeschrijving wanneer crewai_trigger_payload bestaat in crew inputs.",
    )
    _guardrail: GuardrailCallable | None = PrivateAttr(default=None)
    _guardrails: list[GuardrailCallable] = PrivateAttr(
        default_factory=list,
    )
    _guardrail_retry_counts: dict[int, int] = PrivateAttr(
        default_factory=dict,
    )
    _original_description: str | None = PrivateAttr(default=None)
    _original_expected_output: str | None = PrivateAttr(default=None)
    _original_output_file: str | None = PrivateAttr(default=None)
    _thread: threading.Thread | None = PrivateAttr(default=None)
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("guardrail")
    @classmethod
    def validate_guardrail_function(
        cls, v: str | GuardrailCallable | None
    ) -> str | GuardrailCallable | None:
        """
        Als v een callable is, valideer dat de guardrail functie de correcte signatuur en gedrag heeft.
        Als v een string is, retourneer het zoals het is.

        Terwijl type hints statische controle bieden, zorgt deze validator voor runtime veiligheid door:
        1. Te verifiëren dat de functie exact één parameter accepteert (de TaskOutput)
        2. Te controleren of return type annotaties overeenkomen met Tuple[bool, Any] indien aanwezig
        3. Duidelijke, directe foutmeldingen te geven voor debugging

        Deze runtime validatie is cruciaal omdat:
        - Type hints optioneel zijn en genegeerd kunnen worden tijdens runtime
        - Functie signaturen directe validatie nodig hebben voor taakuitvoering
        - Duidelijke foutmeldingen gebruikers helpen guardrail implementatie problemen te debuggen

        Args:
            v: De guardrail functie om te valideren of een string die de guardrail taak beschrijft

        Retourneert:
            De gevalideerde guardrail functie of een string die de guardrail taak beschrijft

        Gooit:
            ValueError: Als de functie signatuur ongeldig is of return annotatie
                       niet overeenkomt met Tuple[bool, Any]
        """
        if v is not None and callable(v):
            sig = inspect.signature(v)
            positional_args = [
                param
                for param in sig.parameters.values()
                if param.default is inspect.Parameter.empty
            ]
            if len(positional_args) != 1:
                raise ValueError("Guardrail functie moet exact één parameter accepteren")

            # Check return annotation if present, but don't require it
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                return_annotation_args = get_args(return_annotation)
                if not (
                    get_origin(return_annotation) is tuple
                    and len(return_annotation_args) == 2
                    and return_annotation_args[0] is bool
                    and (
                        return_annotation_args[1] is Any
                        or return_annotation_args[1] is str
                        or return_annotation_args[1] is TaskOutput
                        or return_annotation_args[1] == str | TaskOutput
                    )
                ):
                    raise ValueError(
                        "Als return type is geannoteerd, moet het Tuple[bool, Any] zijn"
                    )
        return v

    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        return process_config(values, cls)

    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        # Alleen description is verplicht - expected_output is optioneel voor actie-georiënteerde taken
        if self.description is None:
            raise ValueError(
                "description must be provided either directly or through config"
            )
        return self

    @model_validator(mode="after")
    def ensure_guardrail_is_callable(self) -> Task:
        if callable(self.guardrail):
            self._guardrail = self.guardrail
        elif isinstance(self.guardrail, str):
            from crewai.tasks.llm_guardrail import LLMGuardrail

            if self.agent is None:
                raise ValueError("Agent is vereist om LLMGuardrail te gebruiken")

            self._guardrail = cast(
                GuardrailCallable,
                LLMGuardrail(description=self.guardrail, llm=self.agent.llm),
            )

        return self

    @model_validator(mode="after")
    def ensure_guardrails_is_list_of_callables(self) -> Task:
        guardrails = []
        if self.guardrails is not None:
            if isinstance(self.guardrails, (list, tuple)):
                if len(self.guardrails) > 0:
                    for guardrail in self.guardrails:
                        if callable(guardrail):
                            guardrails.append(guardrail)
                        elif isinstance(guardrail, str):
                            if self.agent is None:
                                raise ValueError(
                                    "Agent is vereist om niet-programmatische guardrails te gebruiken"
                                )
                            from crewai.tasks.llm_guardrail import LLMGuardrail

                            guardrails.append(
                                cast(
                                    GuardrailCallable,
                                    LLMGuardrail(
                                        description=guardrail, llm=self.agent.llm
                                    ),
                                )
                            )
                        else:
                            raise ValueError("Guardrail moet een callable of een string zijn")
            else:
                if callable(self.guardrails):
                    guardrails.append(self.guardrails)
                elif isinstance(self.guardrails, str):
                    if self.agent is None:
                        raise ValueError(
                            "Agent is vereist om niet-programmatische guardrails te gebruiken"
                        )
                    from crewai.tasks.llm_guardrail import LLMGuardrail

                    guardrails.append(
                        cast(
                            GuardrailCallable,
                            LLMGuardrail(
                                description=self.guardrails, llm=self.agent.llm
                            ),
                        )
                    )
                else:
                    raise ValueError("Guardrail moet een callable of een string zijn")

        self._guardrails = guardrails
        if self._guardrails:
            self.guardrail = None
            self._guardrail = None

        return self

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: UUID4 | None) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @field_validator("output_file")
    @classmethod
    def output_file_validation(cls, value: str | None) -> str | None:
        """Valideer het output bestandspad.

        Args:
            value: Het output bestandspad om te valideren. Kan None of een string zijn.
                  Als het pad template variabelen bevat (bijv. {var}), worden leidende slashes behouden.
                  Voor normale paden worden leidende slashes verwijderd.

        Retourneert:
            Het gevalideerde en mogelijk gewijzigde pad, of None als geen pad werd opgegeven.

        Gooit:
            ValueError: Als het pad ongeldige karakters bevat, path traversal pogingen,
                      of andere beveiligingsproblemen.
        """
        if value is None:
            return None

        # Basis beveiligingscontroles
        if ".." in value:
            raise ValueError(
                "Path traversal pogingen zijn niet toegestaan in output_file paden"
            )

        # Controleer eerst op shell expansion
        if value.startswith(("~", "$")):
            raise ValueError(
                "Shell expansion karakters zijn niet toegestaan in output_file paden"
            )

        # Controleer dan andere shell speciale karakters
        if any(char in value for char in ["|", ">", "<", "&", ";"]):
            raise ValueError(
                "Shell speciale karakters zijn niet toegestaan in output_file paden"
            )

        # Strip leidende slash niet als het een template pad is met variabelen
        if "{" in value or "}" in value:
            # Valideer template variabele formaat
            template_vars = [part.split("}")[0] for part in value.split("{")[1:]]
            for var in template_vars:
                if not var.isidentifier():
                    raise ValueError(f"Ongeldige template variabele naam: {var}")
            return value

        # Strip leidende slash voor normale paden
        if value.startswith("/"):
            return value[1:]
        return value

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> Task:
        """Stel attributen in op basis van de agent configuratie."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def check_tools(self) -> Self:
        """Controleer of de tools zijn ingesteld."""
        if not self.tools and self.agent and self.agent.tools:
            self.tools = self.agent.tools
        return self

    @model_validator(mode="after")
    def check_output(self) -> Self:
        """Controleer of een output type is ingesteld."""
        output_types = [self.output_json, self.output_pydantic]
        if len([type for type in output_types if type]) > 1:
            raise PydanticCustomError(
                "output_type",
                "Slechts één output type kan worden ingesteld, ofwel output_pydantic of output_json.",
                {},
            )
        return self

    @model_validator(mode="after")
    def handle_max_retries_deprecation(self) -> Self:
        if self.max_retries is not None:
            warnings.warn(
                "The 'max_retries' parameter is deprecated and will be removed in CrewAI v1.0.0. "
                "Please use 'guardrail_max_retries' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.guardrail_max_retries = self.max_retries
        return self

    def execute_sync(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> TaskOutput:
        """Voer de taak synchroon uit."""
        return self._execute_core(agent, context, tools)

    @property
    def key(self) -> str:
        description = self._original_description or self.description
        expected_output = self._original_expected_output or self.expected_output
        source = [description, expected_output]

        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

    @property
    def execution_duration(self) -> float | None:
        if not self.start_time or not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def execute_async(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> Future[TaskOutput]:
        """Voer de taak asynchroon uit."""
        future: Future[TaskOutput] = Future()
        threading.Thread(
            daemon=True,
            target=self._execute_task_async,
            args=(agent, context, tools, future),
        ).start()
        return future

    def _execute_task_async(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
        future: Future[TaskOutput],
    ) -> None:
        """Voer de taak asynchroon uit met context handling."""
        try:
          result = self._execute_core(agent, context, tools)
          future.set_result(result)
        except Exception as e:
          future.set_exception(e)

    async def aexecute_sync(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> TaskOutput:
        """Voer de taak asynchroon uit met native async/await."""
        return await self._aexecute_core(agent, context, tools)

    async def _aexecute_core(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
    ) -> TaskOutput:
        """Voer de kern uitvoeringslogica van de taak asynchroon uit."""
        try:
            agent = agent or self.agent
            self.agent = agent
            if not agent:
                raise Exception(
                    f"De taak '{self.description}' heeft geen agent toegewezen, daarom kan het niet direct worden uitgevoerd en moet het worden uitgevoerd in een Crew met een specifiek proces dat dit ondersteunt, zoals hiërarchisch."
                )

            self.start_time = datetime.datetime.now()

            self.prompt_context = context
            tools = tools or self.tools or []

            self.processed_by_agents.add(agent.role)
            crewai_event_bus.emit(self, TaskStartedEvent(context=context, task=self))  # type: ignore[no-untyped-call]
            result = await agent.aexecute_task(
                task=self,
                context=context,
                tools=tools,
            )

            if not self._guardrails and not self._guardrail:
                pydantic_output, json_output = self._export_output(result)
            else:
                pydantic_output, json_output = None, None

            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
                actions_executed=self.executed_actions,
                execution_success=len(self.executed_actions) > 0 or bool(result),
            )

            if self._guardrails:
                for idx, guardrail in enumerate(self._guardrails):
                    task_output = await self._ainvoke_guardrail_function(
                        task_output=task_output,
                        agent=agent,
                        tools=tools,
                        guardrail=guardrail,
                        guardrail_index=idx,
                    )

            if self._guardrail:
                task_output = await self._ainvoke_guardrail_function(
                    task_output=task_output,
                    agent=agent,
                    tools=tools,
                    guardrail=self._guardrail,
                )

            self.output = task_output
            self.end_time = datetime.datetime.now()

            if self.callback:
                self.callback(self.output)

            crew = self.agent.crew  # type: ignore[union-attr]
            if crew and crew.task_callback and crew.task_callback != self.callback:
                crew.task_callback(self.output)

            if self.output_file:
                content = (
                    json_output
                    if json_output
                    else (
                        pydantic_output.model_dump_json() if pydantic_output else result
                    )
                )
                self._save_file(content)
            crewai_event_bus.emit(
                self,
                TaskCompletedEvent(output=task_output, task=self),  # type: ignore[no-untyped-call]
            )
            return task_output
        except Exception as e:
            self.end_time = datetime.datetime.now()
            crewai_event_bus.emit(self, TaskFailedEvent(error=str(e), task=self))  # type: ignore[no-untyped-call]
            raise e  # Re-raise the exception after emitting the event

    def _execute_core(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
    ) -> TaskOutput:
        """Voer de kern uitvoeringslogica van de taak uit."""
        try:
            agent = agent or self.agent
            self.agent = agent
            if not agent:
                raise Exception(
                    f"De taak '{self.description}' heeft geen agent toegewezen, daarom kan het niet direct worden uitgevoerd en moet het worden uitgevoerd in een Crew met een specifiek proces dat dit ondersteunt, zoals hiërarchisch."
                )

            self.start_time = datetime.datetime.now()

            self.prompt_context = context
            tools = tools or self.tools or []

            self.processed_by_agents.add(agent.role)
            crewai_event_bus.emit(self, TaskStartedEvent(context=context, task=self))  # type: ignore[no-untyped-call]
            result = agent.execute_task(
                task=self,
                context=context,
                tools=tools,
            )

            if not self._guardrails and not self._guardrail:
                pydantic_output, json_output = self._export_output(result)
            else:
                pydantic_output, json_output = None, None

            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
                actions_executed=self.executed_actions,
                execution_success=len(self.executed_actions) > 0 or bool(result),
            )

            if self._guardrails:
                for idx, guardrail in enumerate(self._guardrails):
                    task_output = self._invoke_guardrail_function(
                        task_output=task_output,
                        agent=agent,
                        tools=tools,
                        guardrail=guardrail,
                        guardrail_index=idx,
                    )

            # backwards support
            if self._guardrail:
                task_output = self._invoke_guardrail_function(
                    task_output=task_output,
                    agent=agent,
                    tools=tools,
                    guardrail=self._guardrail,
                )

            self.output = task_output
            self.end_time = datetime.datetime.now()

            if self.callback:
                self.callback(self.output)

            crew = self.agent.crew  # type: ignore[union-attr]
            if crew and crew.task_callback and crew.task_callback != self.callback:
                crew.task_callback(self.output)

            if self.output_file:
                content = (
                    json_output
                    if json_output
                    else (
                        pydantic_output.model_dump_json() if pydantic_output else result
                    )
                )
                self._save_file(content)
            crewai_event_bus.emit(
                self,
                TaskCompletedEvent(output=task_output, task=self),  # type: ignore[no-untyped-call]
            )
            return task_output
        except Exception as e:
            self.end_time = datetime.datetime.now()
            crewai_event_bus.emit(self, TaskFailedEvent(error=str(e), task=self))  # type: ignore[no-untyped-call]
            raise e  # Re-raise the exception after emitting the event

    def prompt(self) -> str:
        """Genereert de taak prompt met optionele markdown formattering.

        Voor actie-georiënteerde taken (action_based=True) wordt een prompt gegenereerd
        die de agent instrueert om tools te gebruiken en acties uit te voeren.

        Voor tekst-georiënteerde taken (action_based=False) wordt een prompt gegenereerd
        met de verwachte output.

        Retourneert:
            str: De geformatteerde prompt string met de taakbeschrijving,
                 en actie-instructies of verwachte output afhankelijk van het taaktype.
        """
        description = self.description

        should_inject = self.allow_crewai_trigger_context

        if should_inject and self.agent:
            crew = getattr(self.agent, "crew", None)
            if crew and hasattr(crew, "_inputs") and crew._inputs:
                trigger_payload = crew._inputs.get("crewai_trigger_payload")
                if trigger_payload is not None:
                    description += f"\n\nTrigger Payload: {trigger_payload}"

        tasks_slices = [description]

        # Voor actie-georiënteerde taken: gebruik de action_task prompt
        if self.action_based:
            action_instruction = self.i18n.slice("action_task")
            tasks_slices.append(action_instruction)
        elif self.expected_output:
            # Voor tekst-georiënteerde taken: gebruik expected_output prompt
            output = self.i18n.slice("expected_output").format(
                expected_output=self.expected_output
            )
            tasks_slices.append(output)

        if self.markdown:
            markdown_instruction = """Your final answer MUST be formatted in Markdown syntax.
Follow these guidelines:
- Use # for headers
- Use ** for bold text
- Use * for italic text
- Use - or * for bullet points
- Use `code` for inline code
- Use ```language for code blocks"""
            tasks_slices.append(markdown_instruction)
        return "\n".join(tasks_slices)

    def interpolate_inputs_and_add_conversation_history(
        self, inputs: dict[str, str | int | float | dict[str, Any] | list[Any]]
    ) -> None:
        """Interpoleer inputs in de taakbeschrijving, verwachte output, en output bestandspad.
           Voeg conversatie geschiedenis toe indien aanwezig.

        Args:
            inputs: Dictionary die template variabelen mapt naar hun waarden.
                   Ondersteunde waarde types zijn strings, integers, en floats.

        Gooit:
            ValueError: Als een vereiste template variabele ontbreekt in inputs.
        """
        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output
        if self.output_file is not None and self._original_output_file is None:
            self._original_output_file = self.output_file

        if not inputs:
            return

        try:
            self.description = interpolate_only(
                input_string=self._original_description, inputs=inputs
            )
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable '{e.args[0]}' in description"
            ) from e
        except ValueError as e:
            raise ValueError(f"Error interpolating description: {e!s}") from e

        try:
            self.expected_output = interpolate_only(
                input_string=self._original_expected_output, inputs=inputs
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error interpolating expected_output: {e!s}") from e

        if self.output_file is not None:
            try:
                self.output_file = interpolate_only(
                    input_string=self._original_output_file, inputs=inputs
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error interpolating output_file path: {e!s}") from e

        if inputs.get("crew_chat_messages"):
            conversation_instruction = self.i18n.slice(
                "conversation_history_instruction"
            )

            crew_chat_messages_json = str(inputs["crew_chat_messages"])

            try:
                crew_chat_messages = json.loads(crew_chat_messages_json)
            except json.JSONDecodeError as e:
                _printer.print(
                    f"An error occurred while parsing crew chat messages: {e}",
                    color="red",
                )
                raise

            conversation_history = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in crew_chat_messages
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            )

            self.description += (
                f"\n\n{conversation_instruction}\n\n{conversation_history}"
            )

    def increment_tools_errors(self) -> None:
        """Verhoog de tools fouten teller."""
        self.tools_errors += 1

    def increment_delegations(self, agent_name: str | None) -> None:
        """Verhoog de delegaties teller."""
        if agent_name:
            self.processed_by_agents.add(agent_name)
        self.delegations += 1

    def add_executed_action(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        success: bool = True,
    ) -> None:
        """Voeg een uitgevoerde actie toe aan de taak.

        Args:
            tool_name: Naam van de uitgevoerde tool.
            arguments: Argumenten die aan de tool zijn meegegeven.
            result: Resultaat van de tool uitvoering.
            success: Of de tool uitvoering succesvol was.
        """
        self.executed_actions.append({
            "tool": tool_name,
            "arguments": arguments or {},
            "result": str(result) if result is not None else None,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat(),
        })

    def copy(  # type: ignore
        self, agents: list[BaseAgent], task_mapping: dict[str, Task]
    ) -> Task:
        """Maakt een diepe kopie van de Task terwijl het originele klasse type behouden blijft.

        Args:
            agents: Lijst van agents beschikbaar voor de taak.
            task_mapping: Dictionary die taak IDs mapt naar Task instanties.

        Retourneert:
            Een kopie van de taak met hetzelfde klasse type als het origineel.
        """
        exclude = {
            "id",
            "agent",
            "context",
            "tools",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        cloned_context = (
            self.context
            if self.context is NOT_SPECIFIED
            else [task_mapping[context_task.key] for context_task in self.context]
            if isinstance(self.context, list)
            else None
        )

        def get_agent_by_role(role: str) -> BaseAgent | None:
            return next((agent for agent in agents if agent.role == role), None)

        cloned_agent = get_agent_by_role(self.agent.role) if self.agent else None
        cloned_tools = shallow_copy(self.tools) if self.tools else []

        return self.__class__(
            **copied_data,
            context=cloned_context,
            agent=cloned_agent,
            tools=cloned_tools,
        )

    def _export_output(
        self, result: str
    ) -> tuple[BaseModel | None, dict[str, Any] | None]:
        pydantic_output: BaseModel | None = None
        json_output: dict[str, Any] | None = None

        if self.output_pydantic or self.output_json:
            model_output = convert_to_model(
                result,
                self.output_pydantic,
                self.output_json,
                self.agent,
                self.converter_cls,
            )

            if isinstance(model_output, BaseModel):
                pydantic_output = model_output
            elif isinstance(model_output, dict):
                json_output = model_output
            elif isinstance(model_output, str):
                try:
                    json_output = json.loads(model_output)
                except json.JSONDecodeError:
                    json_output = None

        return pydantic_output, json_output

    def _get_output_format(self) -> OutputFormat:
        if self.output_json:
            return OutputFormat.JSON
        if self.output_pydantic:
            return OutputFormat.PYDANTIC
        return OutputFormat.RAW

    def _save_file(self, result: dict[str, Any] | str | Any) -> None:
        """Sla taakoutput op naar een bestand.

        Opmerking:
            Voor cross-platform bestand schrijven, vooral op Windows, overweeg FileWriterTool
            van het crewai_tools pakket te gebruiken:
                pip install 'crewai[tools]'
                from crewai_tools import FileWriterTool

        Args:
            result: Het resultaat om op te slaan in het bestand. Kan een dict of elk stringifiable object zijn.

        Gooit:
            ValueError: Als output_file niet is ingesteld
            RuntimeError: Als er een fout is bij het schrijven naar het bestand. Voor cross-platform
                compatibiliteit, vooral op Windows, gebruik FileWriterTool van crewai_tools
                pakket.
        """
        if self.output_file is None:
            raise ValueError("output_file is niet ingesteld.")

        filewriter_recommendation = (
            "Voor cross-platform bestand schrijven, vooral op Windows, "
            "gebruik FileWriterTool van crewai_tools pakket."
        )

        try:
            resolved_path = Path(self.output_file).expanduser().resolve()
            directory = resolved_path.parent

            if self.create_directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            elif not self.create_directory and not directory.exists():
                raise RuntimeError(
                    f"Directory {directory} bestaat niet en create_directory is False"
                )

            with resolved_path.open("w", encoding="utf-8") as file:
                if isinstance(result, dict):
                    import json

                    json.dump(result, file, ensure_ascii=False, indent=2)
                else:
                    file.write(str(result))
        except (OSError, IOError) as e:
            raise RuntimeError(
                "\n".join(
                    [f"Kon output bestand niet opslaan: {e}", filewriter_recommendation]
                )
            ) from e
        return

    def __repr__(self) -> str:
        return f"Task(description={self.description}, expected_output={self.expected_output})"

    @property
    def fingerprint(self) -> Fingerprint:
        """Haal de fingerprint van de taak op.

        Retourneert:
            Fingerprint: De fingerprint van de taak
        """
        return self.security_config.fingerprint

    def _invoke_guardrail_function(
        self,
        task_output: TaskOutput,
        agent: BaseAgent,
        tools: list[BaseTool],
        guardrail: GuardrailCallable | None,
        guardrail_index: int | None = None,
    ) -> TaskOutput:
        if not guardrail:
            return task_output

        if guardrail_index is not None:
            current_retry_count = self._guardrail_retry_counts.get(guardrail_index, 0)
        else:
            current_retry_count = self.retry_count

        max_attempts = self.guardrail_max_retries + 1

        for attempt in range(max_attempts):
            guardrail_result = process_guardrail(
                output=task_output,
                guardrail=guardrail,
                retry_count=current_retry_count,
                event_source=self,
                from_task=self,
                from_agent=agent,
            )

            if guardrail_result.success:
                # Guardrail geslaagd
                if guardrail_result.result is None:
                    raise Exception(
                        "Taak guardrail retourneerde None als resultaat. Dit is niet toegestaan."
                    )

                if isinstance(guardrail_result.result, str):
                    task_output.raw = guardrail_result.result
                    pydantic_output, json_output = self._export_output(
                        guardrail_result.result
                    )
                    task_output.pydantic = pydantic_output
                    task_output.json_dict = json_output
                elif isinstance(guardrail_result.result, TaskOutput):
                    task_output = guardrail_result.result

                return task_output

            # Guardrail mislukt
            if attempt >= self.guardrail_max_retries:
                # Max herhalingen bereikt
                guardrail_name = (
                    f"guardrail {guardrail_index}"
                    if guardrail_index is not None
                    else "guardrail"
                )
                raise Exception(
                    f"Taak faalde {guardrail_name} validatie na {self.guardrail_max_retries} herhalingen. "
                    f"Laatste fout: {guardrail_result.error}"
                )

            if guardrail_index is not None:
                current_retry_count += 1
                self._guardrail_retry_counts[guardrail_index] = current_retry_count
            else:
                self.retry_count += 1
                current_retry_count = self.retry_count

            context = self.i18n.errors("validation_error").format(
                guardrail_result_error=guardrail_result.error,
                task_output=task_output.raw,
            )
            printer = Printer()
            printer.print(
                content=f"Guardrail {guardrail_index if guardrail_index is not None else ''} geblokkeerd (poging {attempt + 1}/{max_attempts}), opnieuw proberen vanwege: {guardrail_result.error}\n",
                color="yellow",
            )

            # Regenereer output van agent
            result = agent.execute_task(
                task=self,
                context=context,
                tools=tools,
            )

            pydantic_output, json_output = self._export_output(result)
            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
                actions_executed=self.executed_actions,
                execution_success=len(self.executed_actions) > 0 or bool(result),
            )

        return task_output

    async def _ainvoke_guardrail_function(
        self,
        task_output: TaskOutput,
        agent: BaseAgent,
        tools: list[BaseTool],
        guardrail: GuardrailCallable | None,
        guardrail_index: int | None = None,
    ) -> TaskOutput:
        """Roep de guardrail functie asynchroon aan."""
        if not guardrail:
            return task_output

        if guardrail_index is not None:
            current_retry_count = self._guardrail_retry_counts.get(guardrail_index, 0)
        else:
            current_retry_count = self.retry_count

        max_attempts = self.guardrail_max_retries + 1

        for attempt in range(max_attempts):
            guardrail_result = process_guardrail(
                output=task_output,
                guardrail=guardrail,
                retry_count=current_retry_count,
                event_source=self,
                from_task=self,
                from_agent=agent,
            )

            if guardrail_result.success:
                if guardrail_result.result is None:
                    raise Exception(
                        "Taak guardrail retourneerde None als resultaat. Dit is niet toegestaan."
                    )

                if isinstance(guardrail_result.result, str):
                    task_output.raw = guardrail_result.result
                    pydantic_output, json_output = self._export_output(
                        guardrail_result.result
                    )
                    task_output.pydantic = pydantic_output
                    task_output.json_dict = json_output
                elif isinstance(guardrail_result.result, TaskOutput):
                    task_output = guardrail_result.result

                return task_output

            if attempt >= self.guardrail_max_retries:
                guardrail_name = (
                    f"guardrail {guardrail_index}"
                    if guardrail_index is not None
                    else "guardrail"
                )
                raise Exception(
                    f"Task failed {guardrail_name} validation after {self.guardrail_max_retries} retries. "
                    f"Last error: {guardrail_result.error}"
                )

            if guardrail_index is not None:
                current_retry_count += 1
                self._guardrail_retry_counts[guardrail_index] = current_retry_count
            else:
                self.retry_count += 1
                current_retry_count = self.retry_count

            context = self.i18n.errors("validation_error").format(
                guardrail_result_error=guardrail_result.error,
                task_output=task_output.raw,
            )
            printer = Printer()
            printer.print(
                content=f"Guardrail {guardrail_index if guardrail_index is not None else ''} blocked (attempt {attempt + 1}/{max_attempts}), retrying due to: {guardrail_result.error}\n",
                color="yellow",
            )

            result = await agent.aexecute_task(
                task=self,
                context=context,
                tools=tools,
            )

            pydantic_output, json_output = self._export_output(result)
            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
                actions_executed=self.executed_actions,
                execution_success=len(self.executed_actions) > 0 or bool(result),
            )

        return task_output
