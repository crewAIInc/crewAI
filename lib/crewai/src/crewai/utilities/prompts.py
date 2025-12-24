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

        return f"""Je bent {self.agent.role} opererend in CONTINUE MONITORING MODUS.

Je Doel: {self.agent.goal}

Achtergrond: {self.agent.backstory}

{tools_section}

CONTINUE OPERATIE INSTRUCTIES:
1. Je draait continu - je mag NOOIT een "Eindantwoord" geven
2. Monitor condities en onderneem actie indien nodig met je beschikbare tools
3. Rapporteer observaties en acties duidelijk
4. Als niets directe actie vereist, vermeld wat je observeert
5. Wees altijd klaar om te reageren op veranderende condities
6. Houd bij wat je hebt gedaan en geobserveerd

Wanneer je iets observeert of actie wilt ondernemen, antwoord in dit formaat:
Gedachte: [Je redenering over wat je observeert of wilt doen]
Actie: [De tool die je wilt gebruiken, of "Observeren" als je alleen monitort]
Actie Input: [De input voor de tool]

Na elk observatie- of actieresultaat, ga door met monitoren en besluit over je volgende actie."""

    def _build_continuous_user_prompt(self) -> str:
        """Bouw gebruiker prompt voor continue modus.

        Retourneert:
            Gebruiker prompt string voor continue operatie.
        """
        return """Begin continue operatie. Monitor condities en onderneem gepaste acties.
Onthoud: Je opereert continu. Probeer niet te eindigen of een eindantwoord te geven.
In plaats daarvan, observeer, onderneem actie indien nodig en ga door met monitoren.

Huidige richtlijn: {task}

Begin met het beoordelen van de huidige situatie en bepaal of actie nodig is."""
