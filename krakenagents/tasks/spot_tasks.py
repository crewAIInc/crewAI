"""Spot desk agent taken voor QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_spot_leadership_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot leadership agents (01-04)."""
    tasks = []

    if "cio" in agents:
        cio = agents["cio"]
        tasks.extend([
            # TASK-A01-01
            create_task(
                description=(
                    "Definieer tradable universe, exposure caps en allocatie per strategie. "
                    "Documenteer huidige allocaties en eventuele wijzigingen."
                ),
                expected_output="Universe en allocatie document met exposure limieten.",
                agent=cio,
            ),
            # TASK-A01-02
            create_task(
                description=(
                    "Stel risico budgetten in per pod: systematisch, discretionair, arb, event, intraday. "
                    "Zorg dat budgetten aansluiten bij totaal portfolio risico."
                ),
                expected_output="Risico budget allocatie per pod met rationale.",
                agent=cio,
            ),
            # TASK-A01-03
            create_task(
                description=(
                    "Voer maandelijkse allocatie en kill/schaal beslissingen uit. "
                    "Baseer beslissingen op performance data en markt condities."
                ),
                expected_output="Maandelijks allocatie beslissingsrapport.",
                agent=cio,
            ),
            # TASK-A01-04
            create_task(
                description=(
                    "Verhoog allocatie naar high-conviction strategieën: scale winning strategieën "
                    "(within risk limits) naar capture extra alpha."
                ),
                expected_output="High-conviction allocatie verhogingsrapport.",
                agent=cio,
            ),
        ])

    if "head_trading" in agents:
        head = agents["head_trading"]
        tasks.extend([
            # TASK-A02-01
            create_task(
                description=(
                    "Dagelijkse desk briefing: focus lijst, levels, events, risico mode. "
                    "Communiceer key informatie naar alle traders."
                ),
                expected_output="Dagelijkse briefing document met alle key informatie.",
                agent=head,
            ),
            # TASK-A02-02
            create_task(
                description=(
                    "Monitor playbook discipline en trade kwaliteit. "
                    "Identificeer en adresseer overtrading of regelovertredingen."
                ),
                expected_output="Trade kwaliteit rapport met eventuele problemen gemarkeerd.",
                agent=head,
            ),
            # TASK-A02-03
            create_task(
                description=(
                    "Voer post-trade reviews uit en handhaaf journaling. "
                    "Identificeer patronen voor verbetering."
                ),
                expected_output="Post-trade review samenvatting met verbetergebieden.",
                agent=head,
            ),
            # TASK-A02-04
            create_task(
                description=(
                    "Laat traders agressief inzetten op A-setup trades (binnen limieten) "
                    "en minimaliseer tijd op marginale kansen."
                ),
                expected_output="A-setup trade focus rapport met trader performance.",
                agent=head,
            ),
        ])

    if "cro" in agents:
        cro = agents["cro"]
        tasks.extend([
            # TASK-A03-01
            create_task(
                description=(
                    "Ontwerp risico framework: exposure caps, liquiditeit tiers, max drawdown, escalaties. "
                    "Documenteer alle limieten en drempels."
                ),
                expected_output="Spot risico framework document.",
                agent=cro,
            ),
            # TASK-A03-02
            create_task(
                description=(
                    "Real-time monitoring en alerts. Handhaaf risico reducties bij drempels."
                ),
                expected_output="Risico monitoring rapport met eventuele ondernomen acties.",
                agent=cro,
            ),
            # TASK-A03-03
            create_task(
                description=(
                    "Keur nieuwe spot strategieën. Voer pre-mortem en failure mode analyse uit."
                ),
                expected_output="Strategie goedkeuring met risico analyse.",
                agent=cro,
            ),
            # TASK-A03-04
            create_task(
                description=(
                    "Sta tijdelijk hoger risico toe voor uitzonderlijke kansen (binnen afgesproken extra marges) "
                    "om buitengewone winsten mogelijk te maken zonder het risico framework te breken."
                ),
                expected_output="Uitzonderlijke kans risico goedkeuringsrapport.",
                agent=cro,
            ),
        ])

    if "coo" in agents:
        coo = agents["coo"]
        tasks.extend([
            # TASK-A04-01
            create_task(
                description=(
                    "Stel dagelijkse reconciliation, goedkeuringen en incident runbooks op. "
                    "Documenteer alle operationele procedures."
                ),
                expected_output="Operationele procedures documentatie.",
                agent=coo,
            ),
            # TASK-A04-02
            create_task(
                description=(
                    "Beheer operationele SLA's met exchanges en custody. "
                    "Track performance tegen SLA's."
                ),
                expected_output="SLA performance rapport.",
                agent=coo,
            ),
            # TASK-A04-03
            create_task(
                description=(
                    "Onderhoud audit trail en handhaaf scheiding van functies. "
                    "Zorg dat alle acties correct gelogd worden."
                ),
                expected_output="Audit trail en scheiding van functies rapport.",
                agent=coo,
            ),
            # TASK-A04-04
            create_task(
                description=(
                    "Versnel onboarding van nieuwe venues/assets tijdens kansen: "
                    "zorg voor snelle account/goedkeuring setup zonder controle regels te overtreden."
                ),
                expected_output="Fast-track onboarding rapport.",
                agent=coo,
            ),
        ])

    return tasks


def get_spot_systematic_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot systematic agents (05, 10)."""
    tasks = []

    if "systematic_head" in agents:
        agent = agents["systematic_head"]
        tasks.extend([
            # TASK-A05-01
            create_task(
                description=(
                    "Ontwerp en onderhoud signaal library voor systematic strategieën. "
                    "Inclusief trend, momentum en mean reversion signalen."
                ),
                expected_output="Signaal library documentatie met performance metrieken.",
                agent=agent,
            ),
            # TASK-A05-02
            create_task(
                description=(
                    "Schrijf strategie specs voor dev team: regels, data, risico, executie aannames. "
                    "Zorg voor duidelijke documentatie voor implementatie."
                ),
                expected_output="Strategie specificatie document.",
                agent=agent,
            ),
            # TASK-A05-03
            create_task(
                description=(
                    "Maandelijkse model review: drift detectie + kill/schaal voorstellen. "
                    "Identificeer strategieën losing edge."
                ),
                expected_output="Model review rapport met aanbevelingen.",
                agent=agent,
            ),
            # TASK-A05-04
            create_task(
                description=(
                    "Gebruik AI/ML en alternatieve data (sentiment, macro) om nieuwe signalen te vinden. "
                    "Valideer rigoureus en pilot voor extra alpha."
                ),
                expected_output="Nieuwe signaal ontdekking rapport met validatie resultaten.",
                agent=agent,
            ),
        ])

    if "systematic_operator" in agents:
        agent = agents["systematic_operator"]
        tasks.extend([
            # TASK-A10-01
            create_task(
                description=(
                    "Voer signalen uit, check data kwaliteit, voer rebalances uit. "
                    "Log alle afwijkingen en fixes."
                ),
                expected_output="Dagelijkse systematische operaties log.",
                agent=agent,
            ),
            # TASK-A10-02
            create_task(
                description=(
                    "Pauzeer strategie bij anomalieën per SOP en rapporteer aan Agent 05/03. "
                    "Documenteer alle pauzes met rationale."
                ),
                expected_output="Anomalie pauze log met escalaties.",
                agent=agent,
            ),
            # TASK-A10-03
            create_task(
                description=(
                    "Onderhoud afwijking en fix logboek. "
                    "Track alle operationele problemen en resoluties."
                ),
                expected_output="Afwijking en fix logboek.",
                agent=agent,
            ),
            # TASK-A10-04
            create_task(
                description=(
                    "Lever continue feedback aan quant devs over executie fricties of data fouten "
                    "zodat modellen/strategieën verbeterd kunnen worden voor meer winst."
                ),
                expected_output="Feedback rapport aan quant team.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_discretionary_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot discretionary/swing agents (06, 13, 14)."""
    tasks = []

    if "discretionary_head" in agents:
        agent = agents["discretionary_head"]
        tasks.extend([
            # TASK-A06-01
            create_task(
                description=(
                    "Bouw thesis-gedreven trades (dagen-weken) met strikte invalidatie. "
                    "Documenteer entry, exit en stop levels."
                ),
                expected_output="Trade thesis document met entry/exit criteria.",
                agent=agent,
            ),
            # TASK-A06-02
            create_task(
                description=(
                    "Integreer research: tokenomics, unlocks, flows, catalysts. "
                    "Gebruik data om trade beslissingen te informeren."
                ),
                expected_output="Research integratie rapport.",
                agent=agent,
            ),
            # TASK-A06-03
            create_task(
                description=(
                    "Beheer positie management: scale in/out, trailing stops, winst bescherming. "
                    "Documenteer alle positie wijzigingen."
                ),
                expected_output="Positie management log.",
                agent=agent,
            ),
            # TASK-A06-04
            create_task(
                description=(
                    "Jaag op niche tokens/narratieven vroeg en neem kleine posities (bewust hoog risico) "
                    "voor potentieel buitensporige winsten als thesis uitkomt. Exit direct bij falen."
                ),
                expected_output="Niche token kansen rapport.",
                agent=agent,
            ),
        ])

    if "swing_trader_i" in agents:
        agent = agents["swing_trader_i"]
        tasks.extend([
            # TASK-A13-01
            create_task(
                description=(
                    "Plan trend continuation/pullback trades met invalidation levels. "
                    "Documenteer setup criteria."
                ),
                expected_output="Trend trade plan.",
                agent=agent,
            ),
            # TASK-A13-02
            create_task(
                description=(
                    "Combine levels met flows/volume (no indicator blindness). "
                    "Use multiple confirmation signals."
                ),
                expected_output="Multi-signal confluence report.",
                agent=agent,
            ),
            # TASK-A13-03
            create_task(
                description=(
                    "Bouw scenario trade plans: base/bull/bear cases. "
                    "Prepare voor different outcomes."
                ),
                expected_output="Scenario trade plan document.",
                agent=agent,
            ),
            # TASK-A13-04
            create_task(
                description=(
                    "Let winners run: increase position of widen trailing stop when trade is convincingly winning "
                    "om trend winst te maximaliseren (handhaaf stop discipline)."
                ),
                expected_output="Winner optimization report.",
                agent=agent,
            ),
        ])

    if "swing_trader_ii" in agents:
        agent = agents["swing_trader_ii"]
        tasks.extend([
            # TASK-A14-01
            create_task(
                description=(
                    "Bouw theme baskets + sector rotation (L2/AI/DeFi) within liquidity tiers. "
                    "Documenteer basket composition."
                ),
                expected_output="Theme basket composition document.",
                agent=agent,
            ),
            # TASK-A14-02
            create_task(
                description=(
                    "Onderhoud strict sizing (never outsized in illiquid assets). "
                    "Documenteer sizing rationale."
                ),
                expected_output="Position sizing report.",
                agent=agent,
            ),
            # TASK-A14-03
            create_task(
                description=(
                    "Plan unlock/supply events met research. "
                    "Prepare positions ahead of events."
                ),
                expected_output="Event preparation plan.",
                agent=agent,
            ),
            # TASK-A14-04
            create_task(
                description=(
                    "Allocate limited capital naar emerging alts (micro-caps of new sector) "
                    "for potential high gains. Strict exit if liquidity drops."
                ),
                expected_output="Emerging alt allocation report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_arbitrage_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot arbitrage agents (07, 11, 12)."""
    tasks = []

    if "arb_head" in agents:
        agent = agents["arb_head"]
        tasks.extend([
            # TASK-A07-01
            create_task(
                description=(
                    "Run cross-exchange spreads, triangular arb, stablecoin dislocations (within policy). "
                    "Monitor all opportunities."
                ),
                expected_output="Arbitrage opportunity report met capacity analysis.",
                agent=agent,
            ),
            # TASK-A07-02
            create_task(
                description=(
                    "Definieer venue filters: withdrawal reliability, limits, liquidity. "
                    "Onderhoud approved venue list."
                ),
                expected_output="Venue filter criteria document.",
                agent=agent,
            ),
            # TASK-A07-03
            create_task(
                description=(
                    "Capacity management: prevent edge erosion through scale/costs. "
                    "Monitor capacity utilization."
                ),
                expected_output="Capacity management report.",
                agent=agent,
            ),
            # TASK-A07-04
            create_task(
                description=(
                    "Explore arbitrage op new/illiquid markets (including DEX if possible) "
                    "with limited capital naar profit before competitors."
                ),
                expected_output="New market arb exploration report.",
                agent=agent,
            ),
        ])

    if "arb_trader_i" in agents:
        agent = agents["arb_trader_i"]
        tasks.extend([
            # TASK-A11-01
            create_task(
                description=(
                    "Scan spreads en execute legs according naar execution policy. "
                    "Documenteer all executions."
                ),
                expected_output="Spread execution log.",
                agent=agent,
            ),
            # TASK-A11-02
            create_task(
                description=(
                    "Monitor venue limits + settlement windows. "
                    "Zorg timely settlement."
                ),
                expected_output="Venue limit monitoring report.",
                agent=agent,
            ),
            # TASK-A11-03
            create_task(
                description=(
                    "Report capacity + frictions (fees, slippage, downtime). "
                    "Identificeer bottlenecks."
                ),
                expected_output="Capacity en friction report.",
                agent=agent,
            ),
            # TASK-A11-04
            create_task(
                description=(
                    "Scale successful arb trades: increase volume op stable spreads "
                    "and expand naar new asset pairs if performance is consistent."
                ),
                expected_output="Arb scaling report.",
                agent=agent,
            ),
        ])

    if "arb_trader_ii" in agents:
        agent = agents["arb_trader_ii"]
        tasks.extend([
            # TASK-A12-01
            create_task(
                description=(
                    "Identificeer en execute triangular opportunities within strict limits. "
                    "Documenteer all opportunities."
                ),
                expected_output="Triangular arb execution log.",
                agent=agent,
            ),
            # TASK-A12-02
            create_task(
                description=(
                    "Trade stablecoin spreads met predefined depeg rules. "
                    "Monitor stablecoin health."
                ),
                expected_output="Stablecoin spread trading log.",
                agent=agent,
            ),
            # TASK-A12-03
            create_task(
                description=(
                    "Monitor settlement risk + venue health met ops. "
                    "Escalate issues immediately."
                ),
                expected_output="Settlement en venue health report.",
                agent=agent,
            ),
            # TASK-A12-04
            create_task(
                description=(
                    "Play stablecoin depeg situations opportunistically (quick in/out voor recovery) "
                    "and experiment met triangular arb op new pairings where liquidity increases."
                ),
                expected_output="Opportunistic depeg trading report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_research_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot research agents (08, 20-24)."""
    tasks = []

    if "research_head" in agents:
        agent = agents["research_head"]
        tasks.extend([
            # TASK-A08-01
            create_task(
                description=(
                    "Bouw catalyst calendar: unlocks, upgrades, listings, treasury moves, governance. "
                    "Track all upcoming events."
                ),
                expected_output="Catalyst calendar en watchlist met scores.",
                agent=agent,
            ),
            # TASK-A08-02
            create_task(
                description=(
                    "Produce watchlists met tradeability score (liquidity, supply risk, narrative, flows). "
                    "Rank opportunities."
                ),
                expected_output="Ranked watchlist met tradeability scores.",
                agent=agent,
            ),
            # TASK-A08-03
            create_task(
                description=(
                    "Publish 'opportunity docket' + real-time alerts met impact assessment. "
                    "Keep desk informed."
                ),
                expected_output="Opportunity docket publication.",
                agent=agent,
            ),
            # TASK-A08-04
            create_task(
                description=(
                    "Monitor social media & dev community voor hype (trending Twitter/Reddit, GitHub activity) "
                    "and alert trading team op early signals."
                ),
                expected_output="Social/dev community monitoring report.",
                agent=agent,
            ),
        ])

    if "onchain_analyst" in agents:
        agent = agents["onchain_analyst"]
        tasks.extend([
            # TASK-A20-01
            create_task(
                description=(
                    "Bouw dashboards: inflow/outflow, whale deposits, cohort behavior. "
                    "Visualize on-chain data."
                ),
                expected_output="On-chain analytics dashboard.",
                agent=agent,
            ),
            # TASK-A20-02
            create_task(
                description=(
                    "Creëer alerts met context (noise vs signal). "
                    "Filter meaningful signals."
                ),
                expected_output="Contextual alert system.",
                agent=agent,
            ),
            # TASK-A20-03
            create_task(
                description=(
                    "Run post-mortems: when signal failed en why. "
                    "Learn van misses."
                ),
                expected_output="Post-mortem analysis report.",
                agent=agent,
            ),
            # TASK-A20-04
            create_task(
                description=(
                    "Convert on-chain signals directly naar trade actions: e.g., large whale deposit -> "
                    "warn voor short, large stablecoin burn -> signal voor potential rally."
                ),
                expected_output="On-chain signal naar trade action mapping.",
                agent=agent,
            ),
        ])

    if "tokenomics_analyst" in agents:
        agent = agents["tokenomics_analyst"]
        tasks.extend([
            # TASK-A21-01
            create_task(
                description=(
                    "Bouw supply shock calendar met impact scores (unlock vs liquidity). "
                    "Track all supply events."
                ),
                expected_output="Supply shock calendar met impact scores.",
                agent=agent,
            ),
            # TASK-A21-02
            create_task(
                description=(
                    "Identificeer mechanical flows: vesting dumps, emissions pressure. "
                    "Predict supply changes."
                ),
                expected_output="Mechanical flows analysis.",
                agent=agent,
            ),
            # TASK-A21-03
            create_task(
                description=(
                    "Warn voor governance/treasury risks. "
                    "Flag potential negative events."
                ),
                expected_output="Governance/treasury risk report.",
                agent=agent,
            ),
            # TASK-A21-04
            create_task(
                description=(
                    "Hunt tokens met extreme tokenomics events (large unlocks, buybacks) coming "
                    "and advise short/long strategieën voor extra alpha."
                ),
                expected_output="Tokenomics event opportunity report.",
                agent=agent,
            ),
        ])

    if "quant_analyst" in agents:
        agent = agents["quant_analyst"]
        tasks.extend([
            # TASK-A22-01
            create_task(
                description=(
                    "Run ad-hoc studies: reactions naar unlocks/listings. "
                    "Analyseer historical patterns."
                ),
                expected_output="Ad-hoc quantitative study.",
                agent=agent,
            ),
            # TASK-A22-02
            create_task(
                description=(
                    "Onderhoud watchlist scoring + sector dashboards. "
                    "Keep metrics updated."
                ),
                expected_output="Watchlist scoring update.",
                agent=agent,
            ),
            # TASK-A22-03
            create_task(
                description=(
                    "Data QA: outliers + venue inconsistencies. "
                    "Zorg data quality."
                ),
                expected_output="Data QA report.",
                agent=agent,
            ),
            # TASK-A22-04
            create_task(
                description=(
                    "Bouw sentiment/trend indexes van alt-data (Twitter volume, Google trends) "
                    "to detect potential price triggers. Share met traders."
                ),
                expected_output="Sentiment/trend index report.",
                agent=agent,
            ),
        ])

    if "news_analyst" in agents:
        agent = agents["news_analyst"]
        tasks.extend([
            # TASK-A23-01
            create_task(
                description=(
                    "Breaking alerts: listings, hacks, outages, regulatory headlines. "
                    "Immediate notification."
                ),
                expected_output="Breaking news alert system.",
                agent=agent,
            ),
            # TASK-A23-02
            create_task(
                description=(
                    "Impact assessment: coins, second-order effects, risks. "
                    "Analyseer implications."
                ),
                expected_output="News impact assessment.",
                agent=agent,
            ),
            # TASK-A23-03
            create_task(
                description=(
                    "Communicate in short, action-oriented bullets. "
                    "Clear en concise updates."
                ),
                expected_output="Action-oriented news summary.",
                agent=agent,
            ),
            # TASK-A23-04
            create_task(
                description=(
                    "Monitor crowd sentiment (Twitter/Reddit/Telegram) voor extreme mood. "
                    "Give contrarian of trend-following advice at manic of panic signals."
                ),
                expected_output="Crowd sentiment analysis.",
                agent=agent,
            ),
        ])

    if "macro_analyst" in agents:
        agent = agents["macro_analyst"]
        tasks.extend([
            # TASK-A24-01
            create_task(
                description=(
                    "Definieer regime labels + guidance (normal/reduced risk). "
                    "Set risk mode recommendations."
                ),
                expected_output="Regime label definition.",
                agent=agent,
            ),
            # TASK-A24-02
            create_task(
                description=(
                    "Bouw calendar: macro events + weekend liquidity risk. "
                    "Track risk periods."
                ),
                expected_output="Macro event calendar.",
                agent=agent,
            ),
            # TASK-A24-03
            create_task(
                description=(
                    "Write scenarios: 'if X, then reduce exposures'. "
                    "Prepare contingency plans."
                ),
                expected_output="Scenario playbook.",
                agent=agent,
            ),
            # TASK-A24-04
            create_task(
                description=(
                    "Communicate clearly when 'risk-on' vs 'risk-off': give desks green light naar go full "
                    "in favorable macro climate, en brake when macro turns against. Update immediately after events."
                ),
                expected_output="Risk-on/off communication.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_execution_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot execution agents (09, 16, 17)."""
    tasks = []

    if "execution_head" in agents:
        agent = agents["execution_head"]
        tasks.extend([
            # TASK-A09-01
            create_task(
                description=(
                    "Set up execution KPIs: implementation shortfall, reject rate, adverse selection. "
                    "Track en improve performance."
                ),
                expected_output="Execution KPI report met improvement actions.",
                agent=agent,
            ),
            # TASK-A09-02
            create_task(
                description=(
                    "Definieer maker/taker policy + routing rules + large order playbooks. "
                    "Documenteer execution procedures."
                ),
                expected_output="Execution policy document.",
                agent=agent,
            ),
            # TASK-A09-03
            create_task(
                description=(
                    "Continuously improve fills/costs across venues. "
                    "Optimaliseer execution quality."
                ),
                expected_output="Fill quality improvement report.",
                agent=agent,
            ),
            # TASK-A09-04
            create_task(
                description=(
                    "Integreer automated execution algos (TWAP/VWAP) en explore dark liquidity sources "
                    "to execute large orders quietly."
                ),
                expected_output="Execution algo integration report.",
                agent=agent,
            ),
        ])

    if "intraday_trader_i" in agents:
        agent = agents["intraday_trader_i"]
        tasks.extend([
            # TASK-A16-01
            create_task(
                description=(
                    "Play setups: breakout validation, absorption, liquidity walls (liquid only). "
                    "Focus op high-probability setups."
                ),
                expected_output="Intraday setup execution log.",
                agent=agent,
            ),
            # TASK-A16-02
            create_task(
                description=(
                    "Journal met setup tags + execution notes. "
                    "Documenteer all trades."
                ),
                expected_output="Trade journal met tags.",
                agent=agent,
            ),
            # TASK-A16-03
            create_task(
                description=(
                    "Onderhoud stop discipline (no moving stops outside playbook). "
                    "Strict risk management."
                ),
                expected_output="Stop discipline compliance report.",
                agent=agent,
            ),
            # TASK-A16-04
            create_task(
                description=(
                    "Increase position size slightly wanneer 'in the zone' en markt trend jouw kant opgaat "
                    "om extra PnL te maken (maar behoud dagelijkse loss limit)."
                ),
                expected_output="Position sizing optimization report.",
                agent=agent,
            ),
        ])

    if "intraday_trader_ii" in agents:
        agent = agents["intraday_trader_ii"]
        tasks.extend([
            # TASK-A17-01
            create_task(
                description=(
                    "Work within no-trade windows (thin liquidity). "
                    "Avoid low-liquidity periods."
                ),
                expected_output="No-trade window compliance.",
                agent=agent,
            ),
            # TASK-A17-02
            create_task(
                description=(
                    "Coördineer larger entries/exits met execution. "
                    "Work met execution team."
                ),
                expected_output="Coördineerd execution log.",
                agent=agent,
            ),
            # TASK-A17-03
            create_task(
                description=(
                    "Dagelijkse self-review + desk review met Agent 02. "
                    "Continuous improvement."
                ),
                expected_output="Dagelijkse review document.",
                agent=agent,
            ),
            # TASK-A17-04
            create_task(
                description=(
                    "Go full in during macro events intraday (e.g., CPI, FOMC) when liquidity is high "
                    "for clear moves. Avoid overtrading after the spike."
                ),
                expected_output="Macro event intraday trading report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_event_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot event-driven agent (15)."""
    tasks = []

    if "event_trader" in agents:
        agent = agents["event_trader"]
        tasks.extend([
            # TASK-A15-01
            create_task(
                description=(
                    "Write pre-event plan: entry, invalidation, hedge/exit rules. "
                    "Documenteer complete event strategy."
                ),
                expected_output="Event trading playbook.",
                agent=agent,
            ),
            # TASK-A15-02
            create_task(
                description=(
                    "Beheer post-event: 'sell the news' + volatility regime. "
                    "Handle event aftermath."
                ),
                expected_output="Post-event management report.",
                agent=agent,
            ),
            # TASK-A15-03
            create_task(
                description=(
                    "Use news/on-chain alerts voor confirmation. "
                    "Valideer event thesis."
                ),
                expected_output="Event confirmation checklist.",
                agent=agent,
            ),
            # TASK-A15-04
            create_task(
                description=(
                    "Sometimes pre-position voor big events (with small risk) when own analysis differs "
                    "from consensus, voor potential outsized gain. Exit immediately op failure."
                ),
                expected_output="Contrarian event positioning report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_mm_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot market making agent (18)."""
    tasks = []

    if "mm_supervisor" in agents:
        agent = agents["mm_supervisor"]
        tasks.extend([
            # TASK-A18-01
            create_task(
                description=(
                    "Set quoting rules: spreads, inventory bands, stop rules. "
                    "Definieer MM parameters."
                ),
                expected_output="Market making rules en inventory report.",
                agent=agent,
            ),
            # TASK-A18-02
            create_task(
                description=(
                    "Monitor inventory; force flattening op regime change. "
                    "Beheer inventory risk."
                ),
                expected_output="Inventory monitoring report.",
                agent=agent,
            ),
            # TASK-A18-03
            create_task(
                description=(
                    "Evalueer PnL source: spread capture vs adverse selection. "
                    "Analyseer MM profitability."
                ),
                expected_output="MM PnL attribution report.",
                agent=agent,
            ),
            # TASK-A18-04
            create_task(
                description=(
                    "Focus op volatile liquid pairs met wide spreads voor more spread capture. "
                    "Reduce inventory quickly op spikes naar avoid slippage. Maximize MM PnL."
                ),
                expected_output="MM optimization report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_risk_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot risk agents (19, 25-27)."""
    tasks = []

    if "inventory_coordinator" in agents:
        agent = agents["inventory_coordinator"]
        tasks.extend([
            # TASK-A19-01
            create_task(
                description=(
                    "Dagelijkse inventory checks: concentrations, liquidity tiers, exit readiness. "
                    "Monitor all positions."
                ),
                expected_output="Inventory en exposure report.",
                agent=agent,
            ),
            # TASK-A19-02
            create_task(
                description=(
                    "Start inter-desk hedge request if hedge only via futures possible "
                    "(without trading futures yourself)."
                ),
                expected_output="Inter-desk hedge request log.",
                agent=agent,
            ),
            # TASK-A19-03
            create_task(
                description=(
                    "Signal mismatch between exposure en regime. "
                    "Alert op misalignments."
                ),
                expected_output="Exposure/regime mismatch alert.",
                agent=agent,
            ),
            # TASK-A19-04
            create_task(
                description=(
                    "Don't hedge too early: let limited overexposure ride when market is favorable. "
                    "Hedge only when risk-asymmetry increases, voor better risk/reward."
                ),
                expected_output="Hedging timing optimization report.",
                agent=agent,
            ),
        ])

    if "risk_analyst" in agents:
        agent = agents["risk_analyst"]
        tasks.extend([
            # TASK-A25-01
            create_task(
                description=(
                    "Measure exposure per coin/sector/venue; correlation clusters; liquidity tiers. "
                    "Comprehensive risk metrics."
                ),
                expected_output="Risk exposure analysis.",
                agent=agent,
            ),
            # TASK-A25-02
            create_task(
                description=(
                    "Generate alerts + escalations op threshold breaches. "
                    "Immediate notification."
                ),
                expected_output="Threshold breach alert system.",
                agent=agent,
            ),
            # TASK-A25-03
            create_task(
                description=(
                    "Produce dagelijkse risk pack voor CIO/CRO/Head Trading. "
                    "Comprehensive risk summary."
                ),
                expected_output="Dagelijkse risk pack.",
                agent=agent,
            ),
            # TASK-A25-04
            create_task(
                description=(
                    "Rapporteer ook upside scenarios: laat zien waar extra risico kan leiden naar veel winst "
                    "so CIO sees balance between risk en reward."
                ),
                expected_output="Upside scenario analysis.",
                agent=agent,
            ),
        ])

    if "venue_risk" in agents:
        agent = agents["venue_risk"]
        tasks.extend([
            # TASK-A26-01
            create_task(
                description=(
                    "Onderhoud venue scorecards (API stability, withdrawals, legal/ops signals). "
                    "Track venue health."
                ),
                expected_output="Venue scorecard update.",
                agent=agent,
            ),
            # TASK-A26-02
            create_task(
                description=(
                    "Set limits per venue + triggers voor exposure reduction. "
                    "Definieer venue risk limits."
                ),
                expected_output="Venue limit setting.",
                agent=agent,
            ),
            # TASK-A26-03
            create_task(
                description=(
                    "Evalueer new venues before approved status. "
                    "Due diligence op new venues."
                ),
                expected_output="New venue evaluation report.",
                agent=agent,
            ),
            # TASK-A26-04
            create_task(
                description=(
                    "Explore new venues cautiously: put small capital when arb edge is large, "
                    "with strict exposure limit en intensive monitoring, naar gain extra profit without big risks."
                ),
                expected_output="New venue exploration report.",
                agent=agent,
            ),
        ])

    if "strategy_risk" in agents:
        agent = agents["strategy_risk"]
        tasks.extend([
            # TASK-A27-01
            create_task(
                description=(
                    "Check overfitting, regime dependency, execution assumptions. "
                    "Valideer strategy robustness."
                ),
                expected_output="Strategy validation report.",
                agent=agent,
            ),
            # TASK-A27-02
            create_task(
                description=(
                    "Definieer kill criteria + monitoring metrics before live. "
                    "Set strategy guardrails."
                ),
                expected_output="Kill criteria definition.",
                agent=agent,
            ),
            # TASK-A27-03
            create_task(
                description=(
                    "Post-launch review: is it working as intended? "
                    "Monitor strategy performance."
                ),
                expected_output="Post-launch strategy review.",
                agent=agent,
            ),
            # TASK-A27-04
            create_task(
                description=(
                    "Think along met quants: find ways naar bring risky but promising strategieën live "
                    "in controlled manner (hedges, lower allocation) instead of rejecting them outright."
                ),
                expected_output="Controlled strategy launch plan.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_operations_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Spot operations agents (28-32)."""
    tasks = []

    if "controller" in agents:
        agent = agents["controller"]
        tasks.extend([
            # TASK-A28-01
            create_task(
                description=(
                    "Dagelijkse NAV/PnL; fees; slippage accounting; resolve breaks. "
                    "Accurate financial tracking."
                ),
                expected_output="Dagelijkse NAV/PnL report.",
                agent=agent,
            ),
            # TASK-A28-02
            create_task(
                description=(
                    "Performance attribution per strategy/pod. "
                    "Detailed performance breakdown."
                ),
                expected_output="Performance attribution report.",
                agent=agent,
            ),
            # TASK-A28-03
            create_task(
                description=(
                    "Report deviations directly naar COO/CRO. "
                    "Immediate escalation."
                ),
                expected_output="Deviation escalation report.",
                agent=agent,
            ),
            # TASK-A28-04
            create_task(
                description=(
                    "Analyseer which strategieën/pods generate real alpha vs just add volatility. "
                    "Advise CIO voor reallocation toward high-performers."
                ),
                expected_output="Alpha generation analysis.",
                agent=agent,
            ),
        ])

    if "treasury" in agents:
        agent = agents["treasury"]
        tasks.extend([
            # TASK-A29-01
            create_task(
                description=(
                    "Beheer buffers per venue; optimize idle vs safety. "
                    "Balanceer liquidity en returns."
                ),
                expected_output="Buffer management report.",
                agent=agent,
            ),
            # TASK-A29-02
            create_task(
                description=(
                    "Funds movement via 4-eyes + whitelists. "
                    "Secure fund transfers."
                ),
                expected_output="Fund movement log.",
                agent=agent,
            ),
            # TASK-A29-03
            create_task(
                description=(
                    "Stress planning: weekend, depeg, venue issues. "
                    "Prepare voor contingencies."
                ),
                expected_output="Stress scenario plan.",
                agent=agent,
            ),
            # TASK-A29-04
            create_task(
                description=(
                    "Put idle funds naar work: lend/stake temporary surpluses (with CRO approval) "
                    "to earn extra yield while not needed voor trading."
                ),
                expected_output="Idle fund optimization report.",
                agent=agent,
            ),
        ])

    if "security" in agents:
        agent = agents["security"]
        tasks.extend([
            # TASK-A30-01
            create_task(
                description=(
                    "Access control, key management, whitelists, device policies. "
                    "Comprehensive security controls."
                ),
                expected_output="Security controls report.",
                agent=agent,
            ),
            # TASK-A30-02
            create_task(
                description=(
                    "Incident runbooks: phishing, compromise, withdrawal anomalies. "
                    "Response procedures."
                ),
                expected_output="Incident runbook.",
                agent=agent,
            ),
            # TASK-A30-03
            create_task(
                description=(
                    "Venue security reviews + account hardening. "
                    "External security assessment."
                ),
                expected_output="Venue security review.",
                agent=agent,
            ),
            # TASK-A30-04
            create_task(
                description=(
                    "Minimize security friction: pre-approve whitelists, set up API keys upfront. "
                    "Maak mogelijk fast fund transfers without compromising security."
                ),
                expected_output="Security friction reduction report.",
                agent=agent,
            ),
        ])

    if "compliance" in agents:
        agent = agents["compliance"]
        tasks.extend([
            # TASK-A31-01
            create_task(
                description=(
                    "Policies: restricted list, personal trading, comms archiving. "
                    "Compliance framework."
                ),
                expected_output="Compliance policy document.",
                agent=agent,
            ),
            # TASK-A31-02
            create_task(
                description=(
                    "Surveillance procedures + escalation op suspicious behavior. "
                    "Monitoring en enforcement."
                ),
                expected_output="Surveillance procedure report.",
                agent=agent,
            ),
            # TASK-A31-03
            create_task(
                description=(
                    "Training + periodic checks. "
                    "Compliance education."
                ),
                expected_output="Training completion report.",
                agent=agent,
            ),
            # TASK-A31-04
            create_task(
                description=(
                    "Track regulations en market developments closely. Update restricted lists/policies "
                    "immediately op changes so aggressive trades don't lead naar violations."
                ),
                expected_output="Regulatory update tracking.",
                agent=agent,
            ),
        ])

    if "ops" in agents:
        agent = agents["ops"]
        tasks.extend([
            # TASK-A32-01
            create_task(
                description=(
                    "Trade capture, confirmations, settlement monitoring. "
                    "Operational workflow."
                ),
                expected_output="Trade operations log.",
                agent=agent,
            ),
            # TASK-A32-02
            create_task(
                description=(
                    "Coördineer met controller/treasury op breaks. "
                    "Cross-team coordination."
                ),
                expected_output="Break coordination report.",
                agent=agent,
            ),
            # TASK-A32-03
            create_task(
                description=(
                    "Operational readiness voor new coins/venues. "
                    "Onboarding preparation."
                ),
                expected_output="Operational readiness assessment.",
                agent=agent,
            ),
            # TASK-A32-04
            create_task(
                description=(
                    "Prepare ops voor new opportunities: before listings of new exchange launches "
                    "hebben al accounts/funding/procedures klaar zodat traders onmiddellijk kunnen handelen."
                ),
                expected_output="Opportunity readiness report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal alle Spot desk taken op gegeven een dictionary van agents.

    Args:
        agents: Dictionary die agent keys mapt naar Agent instances.

    Returns:
        Lijst van alle Spot desk taken.
    """
    tasks = []
    tasks.extend(get_spot_leadership_tasks(agents))
    tasks.extend(get_spot_systematic_tasks(agents))
    tasks.extend(get_spot_discretionary_tasks(agents))
    tasks.extend(get_spot_arbitrage_tasks(agents))
    tasks.extend(get_spot_research_tasks(agents))
    tasks.extend(get_spot_execution_tasks(agents))
    tasks.extend(get_spot_event_tasks(agents))
    tasks.extend(get_spot_mm_tasks(agents))
    tasks.extend(get_spot_risk_tasks(agents))
    tasks.extend(get_spot_operations_tasks(agents))
    return tasks
