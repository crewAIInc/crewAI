"""Prompt generatie en beheer hulpmiddelen voor CrewAI agents."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from crewai.utilities.i18n import I18N, get_i18n


class StandardPromptResult(TypedDict):
    """Resultaat met alleen prompt veld voor standaard modus."""

    prompt: Annotated[str, "De gegenereerde prompt string"]


class SystemPromptResult(StandardPromptResult):
    """Resultaat met systeem, gebruiker en prompt velden voor systeem prompt modus."""

    system: Annotated[str, "Het systeem prompt component"]
    user: Annotated[str, "Het gebruiker prompt component"]


COMPONENTS = Literal["role_playing", "tools", "no_tools", "task"]


class Prompts(BaseModel):
    """Beheert en genereert prompts voor een generieke agent.

    Opmerkingen:
        - Moet worden gerefactord zodat prompt niet strak gekoppeld is aan agent.
    """

    i18n: I18N = Field(default_factory=get_i18n)
    has_tools: bool = Field(
        default=False, description="Geeft aan of de agent toegang heeft tot tools"
    )
    system_template: str | None = Field(
        default=None, description="Aangepaste systeem prompt template"
    )
    prompt_template: str | None = Field(
        default=None, description="Aangepaste gebruiker prompt template"
    )
    response_template: str | None = Field(
        default=None, description="Aangepaste antwoord prompt template"
    )
    use_system_prompt: bool = Field(
        default=False,
        description="Of de systeem prompt gebruikt moet worden wanneer geen aangepaste templates zijn opgegeven",
    )
    agent: Any = Field(description="Referentie naar de agent die deze prompts gebruikt")

    def task_execution(self) -> SystemPromptResult | StandardPromptResult:
        """Genereer een standaard prompt voor taakuitvoering.

        Retourneert:
            Een dictionary met de geconstrueerde prompt(s).
        """
        slices: list[COMPONENTS] = ["role_playing"]
        if self.has_tools:
            slices.append("tools")
        else:
            slices.append("no_tools")
        system: str = self._build_prompt(slices)
        slices.append("task")

        if (
            not self.system_template
            and not self.prompt_template
            and self.use_system_prompt
        ):
            return SystemPromptResult(
                system=system,
                user=self._build_prompt(["task"]),
                prompt=self._build_prompt(slices),
            )
        return StandardPromptResult(
            prompt=self._build_prompt(
                slices,
                self.system_template,
                self.prompt_template,
                self.response_template,
            )
        )

    def _build_prompt(
        self,
        components: list[COMPONENTS],
        system_template: str | None = None,
        prompt_template: str | None = None,
        response_template: str | None = None,
    ) -> str:
        """Construeert een prompt string van gespecificeerde componenten.

        Args:
            components: Lijst van componentnamen om op te nemen in de prompt.
            system_template: Optionele aangepaste template voor de systeem prompt.
            prompt_template: Optionele aangepaste template voor de gebruiker prompt.
            response_template: Optionele aangepaste template voor de antwoord prompt.

        Retourneert:
            De geconstrueerde prompt string.
        """
        prompt: str
        if not system_template or not prompt_template:
            # Als een van de vereiste templates ontbreekt, val terug op het standaard formaat
            prompt_parts: list[str] = [
                self.i18n.slice(component) for component in components
            ]

            # Voeg medewerker mindset toe voor alle agents
            employee_mindset = self.i18n.slice("employee_mindset")
            if employee_mindset:
                prompt_parts.insert(1, employee_mindset)

            # Voeg risico-gedrag toe voor niet-manager agents (agents die niet delegeren)
            if hasattr(self.agent, "allow_delegation") and not getattr(
                self.agent, "allow_delegation", True
            ):
                risk_behavior = self.i18n.slice("risk_taking_behavior")
                if risk_behavior:
                    prompt_parts.insert(2, risk_behavior)

            prompt = "".join(prompt_parts)
        else:
            # Alle templates zijn opgegeven, gebruik ze
            template_parts: list[str] = [
                self.i18n.slice(component)
                for component in components
                if component != "task"
            ]
            system: str = system_template.replace(
                "{{ .System }}", "".join(template_parts)
            )
            prompt = prompt_template.replace(
                "{{ .Prompt }}", "".join(self.i18n.slice("task"))
            )
            # Behandel ontbrekende response_template
            if response_template:
                response: str = response_template.split("{{ .Response }}")[0]
                prompt = f"{system}\n{prompt}\n{response}"
            else:
                prompt = f"{system}\n{prompt}"

        return (
            prompt.replace("{goal}", self.agent.goal)
            .replace("{role}", self.agent.role)
            .replace("{backstory}", self.agent.backstory)
        )

    def continuous_execution(self) -> SystemPromptResult:
        """Genereer prompts voor continue operatie modus.

        Continue modus prompts instrueren de agent om oneindig te opereren,
        condities te monitoren en actie te ondernemen indien nodig zonder een
        "Eindantwoord" te geven.

        Retourneert:
            Een SystemPromptResult met systeem en gebruiker prompts voor continue modus.
        """
        system_prompt = self._build_continuous_system_prompt()
        user_prompt = self._build_continuous_user_prompt()

        return SystemPromptResult(
            system=system_prompt,
            user=user_prompt,
            prompt=f"{system_prompt}\n\n{user_prompt}",
        )

    def _build_continuous_system_prompt(self) -> str:
        """Bouw systeem prompt voor continue modus.

        Retourneert:
            Systeem prompt string voor continue operatie.
        """
        tools_section = ""
        if self.has_tools:
            tools_section = self.i18n.slice("tools")
        else:
            tools_section = "Je hebt geen tools beschikbaar. Je kunt alleen observeren en denken."

        return f"""Je bent {self.agent.role} in CONTINUE OPERATIE MODUS.

Je Doel: {self.agent.goal}

Achtergrond: {self.agent.backstory}

{tools_section}

CONTINUE WERKWIJZE - JE STOPT NOOIT:
1. Je blijft ALTIJD werken naar je doel - geef NOOIT een "Eindantwoord"
2. ONDERNEEM ACTIE wanneer je kansen ziet - wacht niet op instructies
3. Monitor actief en handel PROACTIEF
4. Neem INITIATIEF - je bent een professional, geen robot
5. Kleine risico's zijn OK - niets doen is NIET OK
6. Rapporteer kort wat je doet, focus op ACTIES

FORMAAT:
Gedachte: [Korte redenering - max 2 zinnen]
Actie: [Tool naam]
Actie Input: [JSON input]

Na elke actie, ga DIRECT door met de volgende. Je werkt 24/7."""

    def _build_continuous_user_prompt(self) -> str:
        """Bouw gebruiker prompt voor continue modus.

        Retourneert:
            Gebruiker prompt string voor continue operatie.
        """
        return """Start nu. Je werkt continu naar je doel.

BELANGRIJK: Je geeft NOOIT een "Eindantwoord". Je blijft altijd doorwerken.
Analyseer de situatie en onderneem DIRECT actie.

Huidige richtlijn: {task}

Begin met het beoordelen van de huidige situatie en bepaal of actie nodig is."""
