"""Governance module voor enterprise toegangscontrole en escalatie.

Dit module biedt een complete set klassen voor het implementeren van
governance, toegangscontrole en escalatiemechanismen.

Voorbeeld:
    ```python
    from crewai.governance import (
        ToegangsControle,
        ToegangsRegel,
        EscalatieManager,
        EscalatieRegel,
        EscalatieTriggerType,
        EscalatieDoelType,
    )

    # Setup toegangscontrole
    toegang = ToegangsControle()
    toegang.voeg_regel_toe(ToegangsRegel(
        resource_type="tool",
        principal_type="agent",
        principal_id=agent_id,
        actie="uitvoeren",
        effect="toestaan"
    ))

    # Controleer toegang
    toegestaan, reden = toegang.controleer_toegang(
        principal_id=agent_id,
        principal_type="agent",
        resource_id=tool_id,
        resource_type="tool",
        actie="uitvoeren"
    )

    # Setup escalatie
    escalatie_mgr = EscalatieManager()
    escalatie_mgr.voeg_regel_toe(EscalatieRegel(
        naam="Timeout",
        trigger_type=EscalatieTriggerType.TIMEOUT,
        trigger_waarde=3600,
        escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
        actie=EscalatieActieType.MELDEN
    ))
    ```
"""

from crewai.governance.access_control import (
    # Types
    ActieType,
    EffectType,
    PrincipalType,
    ResourceType,
    # Voorwaarden
    BudgetVoorwaarde,
    TijdsVoorwaarde,
    ToegangsVoorwaarden,
    # Regels en controle
    ToegangsControle,
    ToegangsRegel,
    # Helper functies
    maak_allow_all_regel,
    maak_budget_regel,
    maak_werkuren_regel,
)
from crewai.governance.escalation import (
    # Enums
    EscalatieActieType,
    EscalatieDoelType,
    EscalatieStatus,
    EscalatieTriggerType,
    # Klassen
    Escalatie,
    EscalatieManager,
    EscalatieRegel,
    # Helper functies
    maak_standaard_escalatie_regels,
)

__all__ = [
    # Access control types
    "ActieType",
    "EffectType",
    "PrincipalType",
    "ResourceType",
    # Access control conditions
    "BudgetVoorwaarde",
    "TijdsVoorwaarde",
    "ToegangsVoorwaarden",
    # Access control classes
    "ToegangsControle",
    "ToegangsRegel",
    # Access control helpers
    "maak_allow_all_regel",
    "maak_budget_regel",
    "maak_werkuren_regel",
    # Escalation enums
    "EscalatieActieType",
    "EscalatieDoelType",
    "EscalatieStatus",
    "EscalatieTriggerType",
    # Escalation classes
    "Escalatie",
    "EscalatieManager",
    "EscalatieRegel",
    # Escalation helpers
    "maak_standaard_escalatie_regels",
]
