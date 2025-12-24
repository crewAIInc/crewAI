from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from inspect import Parameter, signature
import json
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

from pydantic import (
    BaseModel,
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
)
from typing_extensions import TypeIs

from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_utils import generate_model_description


_printer = Printer()

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """Controleer of een callable async is."""
    return asyncio.iscoroutinefunction(func)


def _is_awaitable(value: R | Awaitable[R]) -> TypeIs[Awaitable[R]]:
    """Type vernauwing controle voor awaitable waarden."""
    return asyncio.iscoroutine(value) or asyncio.isfuture(value)


class EnvVar(BaseModel):
    name: str
    description: str
    required: bool = True
    default: str | None = None


class BaseTool(BaseModel, ABC):
    class _ArgsSchemaPlaceholder(PydanticBaseModel):
        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="De unieke naam van de tool die duidelijk het doel communiceert."
    )
    description: str = Field(
        description="Wordt gebruikt om het model te vertellen hoe/wanneer/waarom de tool te gebruiken."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=list,
        description="Lijst van omgevingsvariabelen gebruikt door de tool.",
    )
    args_schema: type[PydanticBaseModel] = Field(
        default=_ArgsSchemaPlaceholder,
        validate_default=True,
        description="Het schema voor de argumenten die de tool accepteert.",
    )

    description_updated: bool = Field(
        default=False, description="Vlag om te controleren of de beschrijving is bijgewerkt."
    )

    cache_function: Callable[..., bool] = Field(
        default=lambda _args=None, _result=None: True,
        description="Functie die wordt gebruikt om te bepalen of de tool gecached moet worden, moet een boolean retourneren. Indien None, wordt de tool gecached.",
    )
    result_as_answer: bool = Field(
        default=False,
        description="Vlag om te controleren of de tool het eindantwoord van de agent moet zijn.",
    )
    max_usage_count: int | None = Field(
        default=None,
        description="Maximaal aantal keren dat deze tool gebruikt kan worden. None betekent onbeperkt gebruik.",
    )
    current_usage_count: int = Field(
        default=0,
        description="Huidig aantal keren dat deze tool is gebruikt.",
    )

    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(
        cls, v: type[PydanticBaseModel]
    ) -> type[PydanticBaseModel]:
        if v != cls._ArgsSchemaPlaceholder:
            return v

        run_sig = signature(cls._run)
        fields: dict[str, Any] = {}

        for param_name, param in run_sig.parameters.items():
            if param_name in ("self", "return"):
                continue
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue

            annotation = param.annotation if param.annotation != param.empty else Any

            if param.default is param.empty:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, param.default)

        if not fields:
            arun_sig = signature(cls._arun)
            for param_name, param in arun_sig.parameters.items():
                if param_name in ("self", "return"):
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

        return create_model(f"{cls.__name__}Schema", **fields)

    @field_validator("max_usage_count", mode="before")
    @classmethod
    def validate_max_usage_count(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_usage_count must be a positive integer")
        return v

    def model_post_init(self, __context: Any) -> None:
        self._generate_description()

        super().model_post_init(__context)

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _printer.print(f"Using Tool: {self.name}", color="cyan")
        result = self._run(*args, **kwargs)

        # If _run is async, we safely run it
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        self.current_usage_count += 1

        return result

    async def arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Voer de tool asynchroon uit.

        Args:
            *args: Positionele argumenten om door te geven aan de tool.
            **kwargs: Keyword argumenten om door te geven aan de tool.

        Retourneert:
            Het resultaat van de tool uitvoering.
        """
        result = await self._arun(*args, **kwargs)
        self.current_usage_count += 1
        return result

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async implementatie van de tool. Override voor async ondersteuning."""
        raise NotImplementedError(
            f"{self.__class__.__name__} implementeert _arun niet. "
            "Override _arun voor async ondersteuning of gebruik run() voor sync uitvoering."
        )

    def reset_usage_count(self) -> None:
        """Reset de huidige gebruiksteller naar nul."""
        self.current_usage_count = 0

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Sync implementatie van de tool.

        Subklassen moeten deze methode implementeren voor synchrone uitvoering.

        Args:
            *args: Positionele argumenten voor de tool.
            **kwargs: Keyword argumenten voor de tool.

        Retourneert:
            Het resultaat van de tool uitvoering.
        """

    def to_structured_tool(self) -> CrewStructuredTool:
        """Converteer deze tool naar een CrewStructuredTool instantie."""
        self._set_args_schema()
        structured_tool = CrewStructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self._run,
            result_as_answer=self.result_as_answer,
            max_usage_count=self.max_usage_count,
            current_usage_count=self.current_usage_count,
        )
        structured_tool._original_tool = self
        return structured_tool

    @classmethod
    def from_langchain(cls, tool: Any) -> BaseTool:
        """Maak een Tool instantie van een CrewStructuredTool.

        Deze methode neemt een CrewStructuredTool object en converteert het naar een
        Tool instantie. Het zorgt ervoor dat de opgegeven tool een aanroepbaar 'func'
        attribuut heeft en leidt het argument schema af indien niet expliciet opgegeven.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("De opgegeven tool moet een aanroepbaar 'func' attribuut hebben.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            func_signature = signature(tool.func)
            fields: dict[str, Any] = {}
            for name, param in func_signature.parameters.items():
                if name == "self":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue
                param_annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )
                if param.default is param.empty:
                    fields[name] = (param_annotation, ...)
                else:
                    fields[name] = (param_annotation, param.default)
            if fields:
                args_schema = create_model(f"{tool.name}Input", **fields)
            else:
                args_schema = create_model(
                    f"{tool.name}Input", __base__=PydanticBaseModel
                )

        return cls(
            name=getattr(tool, "name", "Unnamed Tool"),
            description=getattr(tool, "description", ""),
            func=tool.func,
            args_schema=args_schema,
        )

    def _set_args_schema(self) -> None:
        if self.args_schema is None:
            run_sig = signature(self._run)
            fields: dict[str, Any] = {}

            for param_name, param in run_sig.parameters.items():
                if param_name in ("self", "return"):
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

            self.args_schema = create_model(
                f"{self.__class__.__name__}Schema", **fields
            )

    def _generate_description(self) -> None:
        """Genereer de tool beschrijving met een JSON schema voor argumenten."""
        schema = generate_model_description(self.args_schema)
        args_json = json.dumps(schema["json_schema"]["schema"], indent=2)
        self.description = (
            f"Tool Naam: {self.name}\n"
            f"Tool Argumenten: {args_json}\n"
            f"Tool Beschrijving: {self.description}"
        )


class Tool(BaseTool, Generic[P, R]):
    """Tool die een aanroepbare functie omhult.


    Type Parameters:
        P: ParamSpec die de parameters van de functie vastlegt.
        R: Het return type van de functie.
    """

    func: Callable[P, R | Awaitable[R]]

    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Voert de tool synchroon uit.

        Args:
            *args: Positionele argumenten voor de tool.
            **kwargs: Keyword argumenten voor de tool.

        Retourneert:
            Het resultaat van de tool uitvoering.
        """
        _printer.print(f"Tool gebruiken: {self.name}", color="cyan")
        result = self.func(*args, **kwargs)

        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        self.current_usage_count += 1
        return result  # type: ignore[return-value]

    def _run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Voert de omhulde functie uit.

        Args:
            *args: Positionele argumenten voor de functie.
            **kwargs: Keyword argumenten voor de functie.

        Retourneert:
            Het resultaat van de functie uitvoering.
        """
        return self.func(*args, **kwargs)  # type: ignore[return-value]

    async def arun(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Voert de tool asynchroon uit.

        Args:
            *args: Positionele argumenten voor de tool.
            **kwargs: Keyword argumenten voor de tool.

        Retourneert:
            Het resultaat van de tool uitvoering.
        """
        result = await self._arun(*args, **kwargs)
        self.current_usage_count += 1
        return result

    async def _arun(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Voert de omhulde functie asynchroon uit.

        Args:
            *args: Positionele argumenten voor de functie.
            **kwargs: Keyword argumenten voor de functie.

        Retourneert:
            Het resultaat van de async functie uitvoering.

        Raises:
            NotImplementedError: Als de omhulde functie niet async is.
        """
        result = self.func(*args, **kwargs)
        if _is_awaitable(result):
            return await result
        raise NotImplementedError(
            f"{self.name} heeft geen async functie. "
            "Gebruik run() voor sync uitvoering of geef een async functie op."
        )

    @classmethod
    def from_langchain(cls, tool: Any) -> Tool[..., Any]:
        """Maak een Tool instantie van een CrewStructuredTool.

        Deze methode neemt een CrewStructuredTool object en converteert het naar een
        Tool instantie. Het zorgt ervoor dat de opgegeven tool een aanroepbaar 'func'
        attribuut heeft en leidt het argument schema af indien niet expliciet opgegeven.

        Args:
            tool: Het CrewStructuredTool object om te converteren.

        Retourneert:
            Een nieuwe Tool instantie gemaakt van de opgegeven CrewStructuredTool.

        Raises:
            ValueError: Als de opgegeven tool geen aanroepbaar 'func' attribuut heeft.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("De opgegeven tool moet een aanroepbaar 'func' attribuut hebben.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            func_signature = signature(tool.func)
            fields: dict[str, Any] = {}
            for name, param in func_signature.parameters.items():
                if name == "self":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue
                param_annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )
                if param.default is param.empty:
                    fields[name] = (param_annotation, ...)
                else:
                    fields[name] = (param_annotation, param.default)
            if fields:
                args_schema = create_model(f"{tool.name}Input", **fields)
            else:
                args_schema = create_model(
                    f"{tool.name}Input", __base__=PydanticBaseModel
                )

        return cls(
            name=getattr(tool, "name", "Unnamed Tool"),
            description=getattr(tool, "description", ""),
            func=tool.func,
            args_schema=args_schema,
        )


def to_langchain(
    tools: list[BaseTool | CrewStructuredTool],
) -> list[CrewStructuredTool]:
    """Converteer een lijst van tools naar CrewStructuredTool instanties."""
    return [t.to_structured_tool() if isinstance(t, BaseTool) else t for t in tools]


P2 = ParamSpec("P2")
R2 = TypeVar("R2")


@overload
def tool(func: Callable[P2, R2], /) -> Tool[P2, R2]: ...


@overload
def tool(
    name: str,
    /,
    *,
    result_as_answer: bool = ...,
    max_usage_count: int | None = ...,
) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]: ...


@overload
def tool(
    *,
    result_as_answer: bool = ...,
    max_usage_count: int | None = ...,
) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]: ...


def tool(
    *args: Callable[P2, R2] | str,
    result_as_answer: bool = False,
    max_usage_count: int | None = None,
) -> Tool[P2, R2] | Callable[[Callable[P2, R2]], Tool[P2, R2]]:
    """Decorator om een Tool te maken van een functie.

    Kan op drie manieren worden gebruikt:
    1. @tool - decorator zonder argumenten, gebruikt functienaam
    2. @tool("naam") - decorator met aangepaste naam
    3. @tool(result_as_answer=True) - decorator met opties

    Args:
        *args: Ofwel de functie om te decoreren of een aangepaste tool naam.
        result_as_answer: Indien True, wordt het tool resultaat het eindantwoord van de agent.
        max_usage_count: Maximaal aantal keren dat deze tool gebruikt kan worden. None betekent onbeperkt.

    Retourneert:
        Een Tool instantie.

    Voorbeeld:
        @tool
        def begroet(naam: str) -> str:
            '''Begroet iemand.'''
            return f"Hallo, {naam}!"

        resultaat = begroet.run("Wereld")
    """

    def _make_with_name(tool_name: str) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]:
        def _make_tool(f: Callable[P2, R2]) -> Tool[P2, R2]:
            if f.__doc__ is None:
                raise ValueError("Functie moet een docstring hebben")
            if f.__annotations__ is None:
                raise ValueError("Functie moet type annotaties hebben")

            func_sig = signature(f)
            fields: dict[str, Any] = {}

            for param_name, param in func_sig.parameters.items():
                if param_name == "return":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

            class_name = "".join(tool_name.split()).title()
            args_schema = create_model(class_name, **fields)

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
                args_schema=args_schema,
                result_as_answer=result_as_answer,
                max_usage_count=max_usage_count,
                current_usage_count=0,
            )

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])
    if len(args) == 0:

        def decorator(f: Callable[P2, R2]) -> Tool[P2, R2]:
            return _make_with_name(f.__name__)(f)

        return decorator
    raise ValueError("Ongeldige argumenten")
