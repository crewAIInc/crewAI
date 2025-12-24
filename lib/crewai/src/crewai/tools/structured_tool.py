from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
import json
import textwrap
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel, Field, create_model

from crewai.utilities.logger import Logger


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool


class ToolUsageLimitExceededError(Exception):
    """Exceptie die wordt opgeworpen wanneer een tool zijn maximale gebruikslimiet heeft bereikt."""


class CrewStructuredTool:
    """Een gestructureerde tool die op elk aantal inputs kan opereren.

    Deze tool is bedoeld om StructuredTool te vervangen met een aangepaste implementatie
    die beter integreert met het CrewAI ecosysteem.
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        func: Callable[..., Any],
        result_as_answer: bool = False,
        max_usage_count: int | None = None,
        current_usage_count: int = 0,
    ) -> None:
        """Initialiseer de gestructureerde tool.

        Args:
            name: De naam van de tool
            description: Een beschrijving van wat de tool doet
            args_schema: Het pydantic model voor de argumenten van de tool
            func: De functie die wordt uitgevoerd wanneer de tool wordt aangeroepen
            result_as_answer: Of de output direct geretourneerd moet worden
            max_usage_count: Maximaal aantal keren dat deze tool gebruikt kan worden. None betekent onbeperkt gebruik.
            current_usage_count: Huidig aantal keren dat deze tool is gebruikt.
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        self._logger = Logger()
        self.result_as_answer = result_as_answer
        self.max_usage_count = max_usage_count
        self.current_usage_count = current_usage_count
        self._original_tool: BaseTool | None = None

        # Valideer dat de functie signatuur overeenkomt met het schema
        self._validate_function_signature()

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,
        args_schema: type[BaseModel] | None = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> CrewStructuredTool:
        """Maak een tool van een functie.

        Args:
            func: De functie om een tool van te maken
            name: De naam van de tool. Standaard de functienaam
            description: De beschrijving van de tool. Standaard de functie docstring
            return_direct: Of de output direct geretourneerd moet worden
            args_schema: Optioneel schema voor de functie argumenten
            infer_schema: Of het schema afgeleid moet worden van de functie signatuur
            **kwargs: Extra argumenten om door te geven aan de tool

        Retourneert:
            Een CrewStructuredTool instantie

        Voorbeeld:
            >>> def optellen(a: int, b: int) -> int:
            ...     '''Tel twee getallen op'''
            ...     return a + b
            >>> tool = CrewStructuredTool.from_function(optellen)
        """
        name = name or func.__name__
        description = description or inspect.getdoc(func)

        if description is None:
            raise ValueError(
                f"Functie {name} moet een docstring hebben als beschrijving niet is opgegeven."
            )

        # Schoon de beschrijving op
        description = textwrap.dedent(description).strip()

        if args_schema is not None:
            # Gebruik opgegeven schema
            schema = args_schema
        elif infer_schema:
            # Leid schema af van functie signatuur
            schema = cls._create_schema_from_function(name, func)
        else:
            raise ValueError(
                "Ofwel args_schema moet zijn opgegeven of infer_schema moet True zijn."
            )

        return cls(
            name=name,
            description=description,
            args_schema=schema,
            func=func,
            result_as_answer=return_direct,
        )

    @staticmethod
    def _create_schema_from_function(
        name: str,
        func: Callable,
    ) -> type[BaseModel]:
        """Maak een Pydantic schema van een functie signatuur.

        Args:
            name: De naam om te gebruiken voor het schema
            func: De functie om een schema van te maken

        Retourneert:
            Een Pydantic model klasse
        """
        # Haal functie signatuur op
        sig = inspect.signature(func)

        # Haal type hints op
        type_hints = get_type_hints(func)

        # Maak veld definities
        fields = {}
        for param_name, param in sig.parameters.items():
            # Sla self/cls over voor methodes
            if param_name in ("self", "cls"):
                continue

            # Haal type annotatie op
            annotation = type_hints.get(param_name, Any)

            # Haal standaard waarde op
            default = ... if param.default == param.empty else param.default

            # Voeg veld toe
            fields[param_name] = (annotation, Field(default=default))

        # Maak model
        schema_name = f"{name.title()}Schema"
        return create_model(schema_name, **fields)  # type: ignore[call-overload]

    def _validate_function_signature(self) -> None:
        """Valideer dat de functie signatuur overeenkomt met het args schema."""
        sig = inspect.signature(self.func)
        schema_fields = self.args_schema.model_fields

        # Controleer vereiste parameters
        for param_name, param in sig.parameters.items():
            # Sla self/cls over voor methodes
            if param_name in ("self", "cls"):
                continue

            # Sla **kwargs parameters over
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            # Valideer alleen vereiste parameters zonder standaardwaarden
            if param.default == inspect.Parameter.empty:
                if param_name not in schema_fields:
                    raise ValueError(
                        f"Vereiste functie parameter '{param_name}' "
                        f"niet gevonden in args_schema"
                    )

    def _parse_args(self, raw_args: str | dict) -> dict:
        """Parse en valideer de input argumenten tegen het schema.

        Args:
            raw_args: De ruwe argumenten om te parsen, als string of dict

        Retourneert:
            De gevalideerde argumenten als dictionary
        """
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Kon argumenten niet parsen als JSON: {e}") from e

        try:
            validated_args = self.args_schema.model_validate(raw_args)
            return validated_args.model_dump()
        except Exception as e:
            raise ValueError(f"Argumenten validatie mislukt: {e}") from e

    async def ainvoke(
        self,
        input: str | dict,
        config: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        """Roep de tool asynchroon aan.

        Args:
            input: De input argumenten
            config: Optionele configuratie
            **kwargs: Extra keyword argumenten

        Retourneert:
            Het resultaat van de tool uitvoering
        """
        parsed_args = self._parse_args(input)

        if self.has_reached_max_usage_count():
            raise ToolUsageLimitExceededError(
                f"Tool '{self.name}' heeft zijn maximale gebruikslimiet van {self.max_usage_count} bereikt. Je mag de {self.name} tool niet meer gebruiken."
            )

        self._increment_usage_count()

        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**parsed_args, **kwargs)
            # Voer sync functies uit in een thread pool
            import asyncio

            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.func(**parsed_args, **kwargs)
            )
        except Exception:
            raise

    def _run(self, *args, **kwargs) -> Any:
        """Legacy methode voor compatibiliteit."""
        # Converteer args/kwargs naar ons verwachte formaat
        input_dict = dict(zip(self.args_schema.model_fields.keys(), args, strict=False))
        input_dict.update(kwargs)
        return self.invoke(input_dict)

    def invoke(
        self, input: str | dict, config: dict | None = None, **kwargs: Any
    ) -> Any:
        """Hoofdmethode voor tool uitvoering."""
        parsed_args = self._parse_args(input)

        if self.has_reached_max_usage_count():
            raise ToolUsageLimitExceededError(
                f"Tool '{self.name}' heeft zijn maximale gebruikslimiet van {self.max_usage_count} bereikt. Je mag de {self.name} tool niet meer gebruiken."
            )

        self._increment_usage_count()

        if inspect.iscoroutinefunction(self.func):
            return asyncio.run(self.func(**parsed_args, **kwargs))

        result = self.func(**parsed_args, **kwargs)

        if asyncio.iscoroutine(result):
            return asyncio.run(result)

        return result

    def has_reached_max_usage_count(self) -> bool:
        """Controleer of de tool zijn maximale gebruiksaantal heeft bereikt."""
        return (
            self.max_usage_count is not None
            and self.current_usage_count >= self.max_usage_count
        )

    def _increment_usage_count(self) -> None:
        """Verhoog het gebruiksaantal."""
        self.current_usage_count += 1
        if self._original_tool is not None:
            self._original_tool.current_usage_count = self.current_usage_count

    @property
    def args(self) -> dict:
        """Haal het input argumenten schema van de tool op."""
        return self.args_schema.model_json_schema()["properties"]

    def __repr__(self) -> str:
        return (
            f"CrewStructuredTool(name='{self.name}', description='{self.description}')"
        )
