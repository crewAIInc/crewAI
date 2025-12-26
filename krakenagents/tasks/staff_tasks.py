"""STAFF agent taken voor QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_ceo_tasks(ceo: Agent) -> list[Task]:
    """Haal taken op voor STAFF-00 CEO."""
    return [
        # TASK-CEO-00 — ACTIVEER TRADING DESKS (PRIORITEIT 1)
        create_task(
            description=(
                "KRITIEKE EERSTE ACTIE: Activeer de trading desks om trading operaties te starten.\n\n"
                "Je MOET deze stappen in volgorde uitvoeren:\n\n"
                "STAP 1: Controleer desk status\n"
                "- Roep de tool aan: get_desk_status\n"
                "- Input: 'all'\n"
                "- Dit toont of Spot en Futures desks actief zijn\n\n"
                "STAP 2: Start Spot Desk trading\n"
                "- Roep de tool aan: delegate_to_spot_desk\n"
                "- directive: 'Analyseer BTC en ETH spot markten. Controleer huidige prijzen, order books, "
                "en recente trades. Identificeer eventuele arbitrage of trading kansen. Voer kleine "
                "test trades uit als de condities gunstig zijn.'\n"
                "- priority: 'high'\n\n"
                "STAP 3: Start Futures Desk trading\n"
                "- Roep de tool aan: delegate_to_futures_desk\n"
                "- directive: 'Analyseer BTC en ETH perpetual futures. Controleer funding rates, open interest, "
                "en basis vs spot. Zoek naar carry trade kansen of directionele trades. "
                "Voer posities uit als edge wordt geïdentificeerd.'\n"
                "- priority: 'high'\n\n"
                "GA NIET verder met andere taken totdat beide desks zijn geactiveerd."
            ),
            expected_output=(
                "Bevestiging met:\n"
                "1. Desk status controle voltooid\n"
                "2. Spot Desk geactiveerd met trading directive\n"
                "3. Futures Desk geactiveerd met trading directive\n"
                "Beide desks moeten actief zijn voordat deze taak compleet is."
            ),
            agent=ceo,
        ),
        # TASK-CEO-01 — Monitor trading operaties
        create_task(
            description=(
                "Monitor actieve trading operaties over beide desks.\n"
                "1. Gebruik get_desk_status met input 'all' om beide desks te controleren\n"
                "2. Review eventuele alerts van het risico dashboard\n"
                "3. Zorg dat desks hun directives uitvoeren\n"
                "4. Als een desk is gestopt, herstart deze met delegate_to_spot_desk of delegate_to_futures_desk"
            ),
            expected_output="Trading operaties statusrapport met eventuele interventies.",
            agent=ceo,
        ),
    ]


def get_ceo_interview_tasks(ceo: Agent) -> list[Task]:
    """Haal interview case taken op voor STAFF-00 CEO."""
    return [
        # TASK-CEO-INT-01 — Weekend crash + venue problemen
        create_task(
            description=(
                "Interview Case 1: Weekend crash + venue problemen. "
                "Scenario: Grote markt crash tijdens weekend met venue withdrawals bevroren. "
                "Evalueer respons: risicoreductie, communicatie, contingency activatie."
            ),
            expected_output="Case respons document met crisis management aanpak.",
            agent=ceo,
        ),
        # TASK-CEO-INT-02 — Beste strategie werkt, hoe te schalen?
        create_task(
            description=(
                "Interview Case 2: Beste strategie werkt, hoe te schalen? "
                "Scenario: Een strategie levert 3x verwachte returns. "
                "Evalueer: Hoe te schalen zonder edge te verminderen of risico disproportioneel te verhogen."
            ),
            expected_output="Schaal beslissingskader met risico overwegingen.",
            agent=ceo,
        ),
        # TASK-CEO-INT-03 — Trader breekt regels maar maakt winst
        create_task(
            description=(
                "Interview Case 3: Trader breekt risico regels maar maakt winst. "
                "Scenario: Trader overschreed positie limieten maar genereerde significante PnL. "
                "Evalueer: Hoe om te gaan met regelovertredingen die resulteerden in winsten."
            ),
            expected_output="Disciplinair kader dat risico cultuur handhaaft en eerlijk is.",
            agent=ceo,
        ),
    ]


def get_ceo_compensation_tasks(ceo: Agent) -> list[Task]:
    """Haal compensatiestructuur taken op voor STAFF-00 CEO."""
    return [
        # TASK-CEO-COMP-01 — Risico-gecorrigeerde performance bonus
        create_task(
            description=(
                "Stel bonus in op risico-gecorrigeerde performance (niet alleen PnL). "
                "Ontwerp compensatiestructuur die Sharpe ratio beloont, niet alleen returns. "
                "Inclusief drawdown boetes in bonus berekening."
            ),
            expected_output="Risico-gecorrigeerde bonus berekeningsmethodologie.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-02 — Harde drempels
        create_task(
            description=(
                "Implementeer harde drempels voor compensatie. "
                "Definieer minimum performance drempels voordat bonus wordt uitbetaald. "
                "Stel maximum exposure limieten in die bonus clawback triggeren bij overtreding."
            ),
            expected_output="Compensatie drempel document met duidelijke triggers.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-03 — Uitgestelde bonus (skin in the game)
        create_task(
            description=(
                "Implementeer uitgestelde bonus structuur voor skin in the game. "
                "Vest deel van bonus over 2-3 jaar. "
                "Sta clawback toe bij latere verliezen."
            ),
            expected_output="Uitgestelde compensatiestructuur met vesting schema.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-04 — Bonus kicker voor buitengewone performance
        create_task(
            description=(
                "Bonus kicker voor buitengewone performance (extreme winsten extra beloond). "
                "Ontwerp accelerator voor uitzonderlijke returns boven drempel. "
                "Inclusief deferral/clawback als bescherming tegen latere verliezen."
            ),
            expected_output="Performance kicker structuur met beschermende maatregelen.",
            agent=ceo,
        ),
    ]


def get_ceo_90day_tasks(ceo: Agent) -> list[Task]:
    """Haal eerste 90 dagen taken op voor STAFF-00 CEO."""
    return [
        # TASK-CEO-90D-01 (Day 1-30) — Complete board team
        create_task(
            description=(
                "Dag 1-30: Voltooi board team (CIO/CRO/COO/CFO/Compliance/Security/etc.). "
                "Huur in of bevestig alle STAFF posities. "
                "Zorg voor duidelijke rapportage lijnen en verantwoordelijkheden."
            ),
            expected_output="Voltooid organigram met alle STAFF posities ingevuld.",
            agent=ceo,
        ),
        # TASK-CEO-90D-02 (Day 1-30) — Risk charter + kill-switch live
        create_task(
            description=(
                "Dag 1-30: Finaliseer risico charter en kill-switch ladder. "
                "Documenteer alle risico limieten, drempels en escalatie procedures. "
                "Test kill-switch activatie met dry runs."
            ),
            expected_output="Live risico charter document met geteste kill-switch procedures.",
            agent=ceo,
        ),
        # TASK-CEO-90D-03 (Day 1-30) — Desk separation complete
        create_task(
            description=(
                "Dag 1-30: Voltooi desk separatie technisch en operationeel. "
                "Scheid accounts, goedkeuringen en rapportages tussen Spot en Futures. "
                "Verifieer dat er geen cross-desk afhankelijkheden bestaan."
            ),
            expected_output="Desk separatie audit rapport met bevestiging van volledige scheiding.",
            agent=ceo,
        ),
        # TASK-CEO-90D-04 (Day 31-60) — Strategy pipeline
        create_task(
            description=(
                "Dag 31-60: Stel strategie pipeline op: research → pilot → go/no-go. "
                "Definieer gates en criteria voor elke fase. "
                "Creëer standaard templates voor strategie voorstellen."
            ),
            expected_output="Strategie pipeline documentatie met templates.",
            agent=ceo,
        ),
        # TASK-CEO-90D-05 (Day 31-60) — Execution KPIs + cost dashboards
        create_task(
            description=(
                "Dag 31-60: Lanceer executie KPI's en kosten dashboards. "
                "Track implementation shortfall, fees, slippage. "
                "Maak dashboards toegankelijk voor alle relevante personeelsleden."
            ),
            expected_output="Live executie KPI dashboard met kosten tracking.",
            agent=ceo,
        ),
        # TASK-CEO-90D-06 (Day 31-60) — First pilots with small risk
        create_task(
            description=(
                "Dag 31-60: Lanceer eerste strategie pilots met kleine risico budgetten. "
                "Begin met meest veelbelovende strategieën uit pipeline. "
                "Monitor nauwkeurig en verzamel performance data."
            ),
            expected_output="Pilot lanceer rapport met initiële performance data.",
            agent=ceo,
        ),
        # TASK-CEO-90D-07 (Day 61-90) — Scale what works, kill what doesn't
        create_task(
            description=(
                "Dag 61-90: Schaal strategieën op die werken; stop die niet werken. "
                "Gebruik pilot data om geïnformeerde beslissingen te maken. "
                "Heralloceer kapitaal van mislukkingen naar successen."
            ),
            expected_output="Kill/schaal beslissingsrapport met kapitaal herallocatie.",
            agent=ceo,
        ),
        # TASK-CEO-90D-08 (Day 61-90) — 24/7 escalation + incident drills
        create_task(
            description=(
                "Dag 61-90: Stel 24/7 escalatie dekking op en voer incident drills uit. "
                "Test responstijden en besluitvorming onder druk. "
                "Documenteer geleerde lessen en verbeter procedures."
            ),
            expected_output="Incident drill rapport met escalatie dekking bevestiging.",
            agent=ceo,
        ),
        # TASK-CEO-90D-09 (Day 61-90) — Monthly allocation cycle operational
        create_task(
            description=(
                "Dag 61-90: Maandelijkse allocatie cyclus draait als een machine. "
                "Standaardiseer voorstel formaat en tijdlijn. "
                "Zorg voor data-gedreven beslissingen met duidelijke rationale."
            ),
            expected_output="Eerste complete allocatie cyclus documentatie.",
            agent=ceo,
        ),
        # TASK-CEO-90D-10 (Day 61-90) — High-performance culture
        create_task(
            description=(
                "Dag 61-90: Stel high-performance cultuur op: stimuleer agressieve alpha initiatieven "
                "(hoog risico, hoge beloning) binnen risico limieten, met discipline als fundament. "
                "Beloon uitzonderlijke performance, beboet regelovertreding."
            ),
            expected_output="Cultuur statement met incentive alignment documentatie.",
            agent=ceo,
        ),
    ]


def get_group_cio_tasks(group_cio: Agent) -> list[Task]:
    """Haal taken op voor STAFF-01 Group CIO."""
    return [
        # TASK-GCIO-01 — Collect desk proposals + create allocation proposal
        create_task(
            description=(
                "Verzamel desk allocatie voorstellen en creëer Groep allocatie aanbeveling. "
                "Overweeg risico budgetten, capaciteit en performance metrieken."
            ),
            expected_output="Geconsolideerd allocatie voorstel met rationale voor elk desk.",
            agent=group_cio,
        ),
        # TASK-GCIO-02 — Monthly kill/scale review
        create_task(
            description=(
                "Voer maandelijkse kill/schaal review uit op basis van data. Identificeer ondermaats presterende strategieën "
                "om te stoppen en succesvolle strategieën om te schalen."
            ),
            expected_output="Kill/schaal aanbeveling rapport met ondersteunende data analyse.",
            agent=group_cio,
        ),
        # TASK-GCIO-03 — Publish allocation decision
        create_task(
            description=(
                "Publiceer allocatie beslissing met rationale en KPI's per desk/pod. "
                "Communiceer duidelijk naar alle stakeholders. "
                "Stel verwachtingen voor volgende review cyclus."
            ),
            expected_output="Gepubliceerde allocatie beslissing met KPI's.",
            agent=group_cio,
        ),
        # TASK-GCIO-04 — Scout high-alpha opportunities
        create_task(
            description=(
                "Verken high-alpha kansen in nieuwe markten of strategieën. "
                "Reserveer pilot kapitaal met CRO goedkeuring voor het testen van potentieel. "
                "Stel nieuwe markt entrees of strategie innovaties voor."
            ),
            expected_output="High-alpha kansen rapport met pilot aanbevelingen.",
            agent=group_cio,
        ),
    ]


def get_group_cro_tasks(group_cro: Agent) -> list[Task]:
    """Haal taken op voor STAFF-02 Group CRO."""
    return [
        # TASK-GCRO-01 — Define risk charter + hard rails
        create_task(
            description=(
                "Definieer en onderhoud Groep risico charter met harde rails. "
                "Documenteer limieten, drempels en escalatie procedures."
            ),
            expected_output="Bijgewerkt risico charter document met alle limieten en procedures.",
            agent=group_cro,
        ),
        # TASK-GCRO-02 — Enforce veto process
        create_task(
            description=(
                "Handhaaf veto proces bij overtredingen. Review alle overtredingsrapporten "
                "en gebruik veto autoriteit waar nodig."
            ),
            expected_output="Overtredings review rapport met veto beslissingen en rationale.",
            agent=group_cro,
        ),
        # TASK-GCRO-03 — Kill-switch ladder testing
        create_task(
            description=(
                "Test kill-switch ladder met drills en onderhoud audit trail. "
                "Zorg dat alle kill-switch procedures functioneel en gedocumenteerd zijn."
            ),
            expected_output="Kill-switch drill rapport met testresultaten en audit trail.",
            agent=group_cro,
        ),
        # TASK-GCRO-04 — Review risico limieten periodiek
        create_task(
            description=(
                "Review risico limieten periodiek: zorg dat ze hoog genoeg zijn voor agressieve trading "
                "binnen veilige marges. Verhoog limieten voor bewezen succes (met Board goedkeuring). "
                "Balanceer risico appetijt met bescherming."
            ),
            expected_output="Risico limiet review rapport met aanpassingsaanbevelingen.",
            agent=group_cro,
        ),
    ]


def get_group_coo_tasks(group_coo: Agent) -> list[Task]:
    """Haal taken op voor STAFF-03 Group COO."""
    return [
        # TASK-GCOO-01 — Enforce Spot/Derivatives separation
        create_task(
            description=(
                "Handhaaf operationele scheiding van Spot/Derivatives. "
                "Verifieer dat accounts, goedkeuringen en rapportages correct gescheiden zijn."
            ),
            expected_output="Separatie compliance rapport met eventuele hiaten geïdentificeerd.",
            agent=group_coo,
        ),
        # TASK-GCOO-02 — Daily ops control cycle
        create_task(
            description=(
                "Voer dagelijkse ops controle cyclus uit. Zorg dat alle breaks geïdentificeerd "
                "en opgelost worden dezelfde dag."
            ),
            expected_output="Dagelijks ops controle rapport met break resolutie status.",
            agent=group_coo,
        ),
        # TASK-GCOO-03 — Incident response coordination
        create_task(
            description=(
                "Coördineer incident respons voor venue uitval, settlement problemen "
                "en ops problemen. Zorg voor correcte communicatie en resolutie."
            ),
            expected_output="Incident respons rapport met ondernomen acties en resolutie.",
            agent=group_coo,
        ),
        # TASK-GCOO-04 — Scale ops for high volume
        create_task(
            description=(
                "Schaal ops processen voor hoog volume: zorg dat tijdens piek volatiliteit "
                "reconciliations, settlements etc. vlekkeloos verlopen zonder vertraging. "
                "Stress test operationele capaciteit."
            ),
            expected_output="Ops schaalbaarheid rapport met stress test resultaten.",
            agent=group_coo,
        ),
    ]


def get_group_cfo_tasks(group_cfo: Agent) -> list[Task]:
    """Haal taken op voor STAFF-04 Group CFO."""
    return [
        # TASK-GCFO-01 — Daily NAV/PnL consolidation
        create_task(
            description=(
                "Dagelijkse NAV/PnL consolidatie over spot en derivaten. "
                "Zorg voor nauwkeurige berekening van alle posities."
            ),
            expected_output="Dagelijks NAV/PnL rapport met geconsolideerde cijfers.",
            agent=group_cfo,
        ),
        # TASK-GCFO-02 — Performance attribution
        create_task(
            description=(
                "Performance attributie per desk/pod/strategie. "
                "Identificeer top en zwakke performers."
            ),
            expected_output="Performance attributie rapport met rankings.",
            agent=group_cfo,
        ),
        # TASK-GCFO-03 — Cost dashboards + budget variances
        create_task(
            description=(
                "Kosten analyse: monitor fees/funding/borrow en identificeer optimalisatie kansen. "
                "Track budget afwijkingen en markeer afwijkingen."
            ),
            expected_output="Kosten analyse rapport met optimalisatie aanbevelingen.",
            agent=group_cfo,
        ),
        # TASK-GCFO-04 — Identificeer top vs zwakke strategieën
        create_task(
            description=(
                "Identificeer top vs zwakke strategieën: benadruk welke desks/pods meeste alpha leveren "
                "vs welke ondermaats presteren. Lever data voor gerichte allocatie beslissingen."
            ),
            expected_output="Strategie performance ranking met allocatie aanbevelingen.",
            agent=group_cfo,
        ),
        # TASK-GCFO-05 — Optimaliseer trading kosten
        create_task(
            description=(
                "Optimaliseer trading kosten: monitor fees/funding/borrow en stel verbeteringen voor "
                "(betere fee tiers, minder idle assets) om netto PnL te maximaliseren."
            ),
            expected_output="Trading kosten optimalisatie rapport met specifieke aanbevelingen.",
            agent=group_cfo,
        ),
    ]


def get_compliance_tasks(compliance: Agent) -> list[Task]:
    """Haal taken op voor STAFF-05 Compliance."""
    return [
        # TASK-COMPL-01 — Enforce compliance stops
        create_task(
            description=(
                "Handhaaf compliance stops: restricted list en markt gedrag regels. "
                "Blokkeer elke verboden trading activiteit."
            ),
            expected_output="Compliance handhavingsrapport met eventuele blokkades of waarschuwingen.",
            agent=compliance,
        ),
        # TASK-COMPL-02 — Periodic training + policy checks
        create_task(
            description=(
                "Voer periodieke training en policy checks uit. "
                "Zorg dat alle personeelsleden compliant zijn met huidige policies."
            ),
            expected_output="Training en policy compliance rapport.",
            agent=compliance,
        ),
        # TASK-COMPL-03 — Incident/case escalation workflow
        create_task(
            description=(
                "Voer incident/case escalatie workflow uit. "
                "Documenteer en track alle compliance incidenten. "
                "Escaleer naar geschikte partijen gebaseerd op ernst."
            ),
            expected_output="Incident tracking rapport met escalatie acties.",
            agent=compliance,
        ),
        # TASK-COMPL-04 — Advanced surveillance tools
        create_task(
            description=(
                "Implementeer geavanceerde tools (chain analytics voor fund flows, AI surveillance) "
                "om overtredingen vroeg te detecteren zonder trading te vertragen. "
                "Balanceer toezicht met operationele efficiëntie."
            ),
            expected_output="Surveillance tool implementatie rapport met detectie metrieken.",
            agent=compliance,
        ),
    ]


def get_security_tasks(security: Agent) -> list[Task]:
    """Haal taken op voor STAFF-06 Security."""
    return [
        # TASK-SEC-01 — Access control + key management
        create_task(
            description=(
                "Implementeer en onderhoud toegangscontrole en key management policies. "
                "Zorg voor correcte custody security."
            ),
            expected_output="Security policy implementatie rapport.",
            agent=security,
        ),
        # TASK-SEC-02 — Withdrawal/whitelist approval flows
        create_task(
            description=(
                "Handhaaf withdrawal/whitelist goedkeuringstromen met scheiding van functies. "
                "Verifieer dat alle grote transfers correcte goedkeuring volgen."
            ),
            expected_output="Goedkeuringsstroom audit rapport.",
            agent=security,
        ),
        # TASK-SEC-03 — Test incident runbooks
        create_task(
            description=(
                "Test incident runbooks: phishing, compromise, anomalieën. "
                "Voer regelmatige drills uit en update procedures op basis van resultaten. "
                "Zorg dat responstijden targets halen."
            ),
            expected_output="Incident drill resultaten met procedure updates.",
            agent=security,
        ),
        # TASK-SEC-04 — Advanced custody (MPC wallets)
        create_task(
            description=(
                "Implementeer geavanceerde custody (bijv. MPC wallets via Fireblocks/Copper) "
                "voor veilige opslag EN snelle toegang zodat trading kansen niet worden gemist. "
                "Balanceer security met operationele snelheid."
            ),
            expected_output="Geavanceerde custody implementatie rapport met performance metrieken.",
            agent=security,
        ),
    ]


def get_prime_tasks(prime: Agent) -> list[Task]:
    """Haal taken op voor STAFF-07 Prime/Venues."""
    return [
        # TASK-PRIME-01 — Venue scorecards + limits
        create_task(
            description=(
                "Onderhoud venue scorecards en stel limieten voor met CRO. "
                "Beoordeel venue gezondheid, fees en betrouwbaarheid."
            ),
            expected_output="Venue scorecard update met limiet aanbevelingen.",
            agent=prime,
        ),
        # TASK-PRIME-02 — Liquidity/fee optimization
        create_task(
            description=(
                "Optimaliseer liquiditeit en fees per venue. "
                "Identificeer kansen voor betere executie."
            ),
            expected_output="Liquiditeit/fee optimalisatie rapport.",
            agent=prime,
        ),
        # TASK-PRIME-03 — Concentration risk mitigation
        create_task(
            description=(
                "Mitigeer concentratie risico met diversificatie plan. "
                "Zorg dat geen enkele venue buitensporige exposure heeft. "
                "Onderhoud backup venues voor kritieke operaties."
            ),
            expected_output="Diversificatie plan met concentratie limieten.",
            agent=prime,
        ),
        # TASK-PRIME-04 — Low-latency connectivity
        create_task(
            description=(
                "Investeer in low-latency connectiviteit: co-locatie, dedicated lijnen naar key venues. "
                "Minimaliseer latency/slippage voor competitieve executie."
            ),
            expected_output="Connectiviteit verbeterplan met latency metrieken.",
            agent=prime,
        ),
        # TASK-PRIME-05 — Multi-venue relationships
        create_task(
            description=(
                "Onderhoud multi-venue relaties: prime brokers, OTC desks voor diepe liquiditeit "
                "en snelle executie van grote orders. "
                "Onderhandel over betere voorwaarden waar mogelijk."
            ),
            expected_output="Venue relatie rapport met onderhandelingsresultaten.",
            agent=prime,
        ),
    ]


def get_data_tasks(data: Agent) -> list[Task]:
    """Haal taken op voor STAFF-08 Data."""
    return [
        # TASK-DATA-01 — Core datasets + QA checks
        create_task(
            description=(
                "Stel core datasets op met QA checks. "
                "Zorg voor data kwaliteit en compleetheid."
            ),
            expected_output="Dataset kwaliteit rapport met QA resultaten.",
            agent=data,
        ),
        # TASK-DATA-02 — Standardize desk dashboards
        create_task(
            description=(
                "Standaardiseer desk dashboards voor risico/perf/executie/research. "
                "Zorg voor consistente metrieken over desks."
            ),
            expected_output="Dashboard standaardisatie rapport.",
            agent=data,
        ),
        # TASK-DATA-03 — Alerting on data corruption/outliers
        create_task(
            description=(
                "Implementeer alerting op data corruptie en outliers. "
                "Detecteer en markeer anomalieën voordat ze trading beslissingen beïnvloeden. "
                "Onderhoud data integriteit standaarden."
            ),
            expected_output="Data kwaliteit alerting systeem met detectie metrieken.",
            agent=data,
        ),
        # TASK-DATA-04 — Alternative data integration
        create_task(
            description=(
                "Introduceer alternatieve data: social sentiment, zoektrends, developer metrieken, etc. "
                "Integreer in dashboards voor traders om edge te krijgen."
            ),
            expected_output="Alternatieve data integratie rapport met bronlijst.",
            agent=data,
        ),
        # TASK-DATA-05 — ML experimentation
        create_task(
            description=(
                "Experimenteer met machine learning op data: predictive models, anomaly detection "
                "om verborgen alpha te vinden. Valideer signalen rigoureus voor gebruik."
            ),
            expected_output="ML experimentatie rapport met gevalideerde signaal kandidaten.",
            agent=data,
        ),
    ]


def get_people_tasks(people: Agent) -> list[Task]:
    """Haal taken op voor STAFF-09 People."""
    return [
        # TASK-PEOPLE-01 — Hiring scorecards + interview cases
        create_task(
            description=(
                "Rol hiring scorecards en interview cases uit (inclusief CEO cases). "
                "Zorg voor consistente evaluatie van kandidaten."
            ),
            expected_output="Hiring scorecard implementatie rapport.",
            agent=people,
        ),
        # TASK-PEOPLE-02 — Bonus/clawback/deferral structure
        create_task(
            description=(
                "Implementeer bonus/clawback/deferral structuur. "
                "Zorg dat compensatie aansluit bij risico-gecorrigeerde performance."
            ),
            expected_output="Compensatiestructuur implementatie rapport.",
            agent=people,
        ),
        # TASK-PEOPLE-03 — Performance cycle
        create_task(
            description=(
                "Voer performance cyclus uit: metrieken → feedback → consequenties. "
                "Zorg voor tijdige reviews en bruikbare feedback. "
                "Handel ondermaats presteerders besluitvaardig af."
            ),
            expected_output="Performance cyclus rapport met ondernomen acties.",
            agent=people,
        ),
        # TASK-PEOPLE-04 — High-risk/high-reward culture
        create_task(
            description=(
                "Cultiveer hoog-risico/hoge-beloning cultuur: beloon buitensporige winsten (binnen regels) "
                "met snellere promotie en grotere bonus. "
                "Pak regelovertreders onmiddellijk aan om discipline te handhaven."
            ),
            expected_output="Cultuur assessment rapport met incentive alignment.",
            agent=people,
        ),
    ]


def get_staff_tasks(agents: dict[str, Agent], mode: str = "trading") -> list[Task]:
    """Haal STAFF taken op gegeven een dictionary van agents.

    Args:
        agents: Dictionary die agent keys mapt naar Agent instances.
                Verwachte keys: ceo, group_cio, group_cro, group_coo,
                group_cfo, compliance, security, prime, data, people
        mode: Taak mode - "trading" voor actieve trading operaties (standaard),
              "full" voor alle taken inclusief interviews/compensation/90day

    Returns:
        Lijst van STAFF taken.
    """
    tasks = []

    if "ceo" in agents:
        # CEO always gets core trading tasks
        tasks.extend(get_ceo_tasks(agents["ceo"]))

        # Only add extra tasks in full mode
        if mode == "full":
            tasks.extend(get_ceo_interview_tasks(agents["ceo"]))
            tasks.extend(get_ceo_compensation_tasks(agents["ceo"]))
            tasks.extend(get_ceo_90day_tasks(agents["ceo"]))

    # In trading mode zijn alleen CEO taken belangrijk - CEO delegeert naar desks
    # Andere STAFF agents ondersteunen maar hebben geen primaire taken tijdens trading
    if mode == "full":
        if "group_cio" in agents:
            tasks.extend(get_group_cio_tasks(agents["group_cio"]))
        if "group_cro" in agents:
            tasks.extend(get_group_cro_tasks(agents["group_cro"]))
        if "group_coo" in agents:
            tasks.extend(get_group_coo_tasks(agents["group_coo"]))
        if "group_cfo" in agents:
            tasks.extend(get_group_cfo_tasks(agents["group_cfo"]))
        if "compliance" in agents:
            tasks.extend(get_compliance_tasks(agents["compliance"]))
        if "security" in agents:
            tasks.extend(get_security_tasks(agents["security"]))
        if "prime" in agents:
            tasks.extend(get_prime_tasks(agents["prime"]))
        if "data" in agents:
            tasks.extend(get_data_tasks(agents["data"]))
        if "people" in agents:
            tasks.extend(get_people_tasks(agents["people"]))

    return tasks
