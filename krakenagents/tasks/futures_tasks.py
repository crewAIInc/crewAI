"""Futures desk agent taken voor QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_futures_leadership_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures leadership agents (33-36)."""
    tasks = []

    if "cio" in agents:
        cio = agents["cio"]
        tasks.extend([
            # TASK-A33-01
            create_task(
                description=(
                    "Allocatie over futures strategieën (funding/carry, intraday, systematisch, curve). "
                    "Overweeg leverage en funding kosten."
                ),
                expected_output="Futures universe en allocatie document.",
                agent=cio,
            ),
            # TASK-A33-02
            create_task(
                description=(
                    "Bepaal leverage per regime + max netto/bruto exposure. "
                    "Stel leverage limieten in per pod."
                ),
                expected_output="Risico budget allocatie met leverage limieten.",
                agent=cio,
            ),
            # TASK-A33-03
            create_task(
                description=(
                    "Kill/schaal beslissingen gebaseerd op risico-gecorrigeerde performance. "
                    "Factor in funding regime en basis condities."
                ),
                expected_output="Maandelijks futures allocatie beslissingsrapport.",
                agent=cio,
            ),
            # TASK-A33-04
            create_task(
                description=(
                    "Sta leverage boven normaal maximum toe voor uitzonderlijke edge (alleen met CRO goedkeuring) "
                    "om mega-kansen te vangen terwijl margin buffers gegarandeerd blijven."
                ),
                expected_output="Uitzonderlijke leverage goedkeuringsrapport.",
                agent=cio,
            ),
        ])

    if "head_trading" in agents:
        head = agents["head_trading"]
        tasks.extend([
            # TASK-A34-01
            create_task(
                description=(
                    "Dagelijks plan: focus instrumenten, event vensters, risico mode. "
                    "Coördineer met spot desk op cross-desk kansen."
                ),
                expected_output="Dagelijks futures briefing document.",
                agent=head,
            ),
            # TASK-A34-02
            create_task(
                description=(
                    "Monitor intraday risico controles: max verlies, max leverage. "
                    "Speciale focus op leverage gebruik."
                ),
                expected_output="Trade discipline rapport met leverage analyse.",
                agent=head,
            ),
            # TASK-A34-03
            create_task(
                description=(
                    "Review cultuur + fout reductie. "
                    "Voer post-trade reviews uit en handhaaf journaling."
                ),
                expected_output="Review cultuur rapport.",
                agent=head,
            ),
            # TASK-A34-04
            create_task(
                description=(
                    "Stimuleer grotere positie op high-conviction trades (binnen margin limieten) en leer ervan. "
                    "Zorg dat margin stress test gedaan is voor zo'n push."
                ),
                expected_output="High-conviction trade rapport.",
                agent=head,
            ),
        ])

    if "cro" in agents:
        cro = agents["cro"]
        tasks.extend([
            # TASK-A35-01
            create_task(
                description=(
                    "Ontwerp risico framework: leverage caps, liquidatie buffers, drawdown limieten "
                    "(en optionele gamma-achtige exposure). Documenteereer alle limieten."
                ),
                expected_output="Futures risico framework met margin controles.",
                agent=cro,
            ),
            # TASK-A35-02
            create_task(
                description=(
                    "Real-time monitoring: margin, liquidatie afstand, venue risico. "
                    "Handhaaf positie reducties bij drempels."
                ),
                expected_output="Margin en liquidatie risico rapport.",
                agent=cro,
            ),
            # TASK-A35-03
            create_task(
                description=(
                    "Keur nieuwe derivaten strategieën goed. "
                    "Voer pre-mortem en failure mode analyse uit."
                ),
                expected_output="Strategie goedkeuring met risico analyse.",
                agent=cro,
            ),
            # TASK-A35-04
            create_task(
                description=(
                    "Verhoog risico limieten wanneer performance het toestaat: met consistent profits under current limit, "
                    "consider higher limits (Board approval) om potentieel te verhogen zonder liquidatie gevaar."
                ),
                expected_output="Risico limiet verhogingsvoorstel.",
                agent=cro,
            ),
        ])

    if "coo" in agents:
        coo = agents["coo"]
        tasks.extend([
            # TASK-A36-01
            create_task(
                description=(
                    "Dagelijkse reconciliation van posities, funding, fees. "
                    "Zorg voor 24/7 dekking voor margin events."
                ),
                expected_output="Futures operationele procedures.",
                agent=coo,
            ),
            # TASK-A36-02
            create_task(
                description=(
                    "Incident management voor venue uitval/margin wijzigingen. "
                    "Coördineer snelle respons."
                ),
                expected_output="Incident management rapport.",
                agent=coo,
            ),
            # TASK-A36-03
            create_task(
                description=(
                    "Proces controle + audit trail. "
                    "Zorg dat alle acties correct gelogd worden."
                ),
                expected_output="Proces controle rapport.",
                agent=coo,
            ),
            # TASK-A36-04
            create_task(
                description=(
                    "Zorg dat operaties real-time blijven tijdens extreme bewegingen (liquidatie cascades): "
                    "snelle funding aanpassingen, margin top-ups van treasury, etc., vlekkeloos uitgevoerd."
                ),
                expected_output="Extreme beweging ops gereedheid rapport.",
                agent=coo,
            ),
        ])

    return tasks


def get_futures_systematic_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures systematic agents (37, 42)."""
    tasks = []

    if "systematic_head" in agents:
        agent = agents["systematic_head"]
        tasks.extend([
            # TASK-A37-01
            create_task(
                description=(
                    "Ontwerp signals (trend/mean reversion/carry filters) voor perps/futures. "
                    "Include funding cost assumptions."
                ),
                expected_output="Futures signal library documentation.",
                agent=agent,
            ),
            # TASK-A37-02
            create_task(
                description=(
                    "Specify strategies voor dev team; monitor live drift. "
                    "Zorg clear documentation."
                ),
                expected_output="Strategie specificatie document.",
                agent=agent,
            ),
            # TASK-A37-03
            create_task(
                description=(
                    "Maandelijkse model reviews + performance attribution. "
                    "Identificeer strategies losing edge."
                ),
                expected_output="Model review rapport.",
                agent=agent,
            ),
            # TASK-A37-04
            create_task(
                description=(
                    "Research advanced quant (AI, deep learning op orderbook data) voor new high-alpha strategies. "
                    "Strict testing en phased rollout if successful."
                ),
                expected_output="Advanced quant research report.",
                agent=agent,
            ),
        ])

    if "systematic_operator" in agents:
        agent = agents["systematic_operator"]
        tasks.extend([
            # TASK-A42-01
            create_task(
                description=(
                    "Voer uit signals; monitor exceptions; pause op anomalies. "
                    "Track funding payments en their impact."
                ),
                expected_output="Dagelijkse futures systematic operations log.",
                agent=agent,
            ),
            # TASK-A42-02
            create_task(
                description=(
                    "Log deviations; coordinate met Agent 37/35. "
                    "Documenteer all operational issues."
                ),
                expected_output="Deviation log met escalations.",
                agent=agent,
            ),
            # TASK-A42-03
            create_task(
                description=(
                    "Voorkom human override outside protocol. "
                    "Onderhoud systematic discipline."
                ),
                expected_output="Protocol compliance report.",
                agent=agent,
            ),
            # TASK-A42-04
            create_task(
                description=(
                    "Optimaliseer capital utilization: rapporteer aan Agent 33 wanneer een strategie idle is of underused "
                    "so capital can be temporarily deployed elsewhere voor more return."
                ),
                expected_output="Capital utilization optimization report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_carry_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures carry agents (38, 43, 44)."""
    tasks = []

    if "carry_head" in agents:
        agent = agents["carry_head"]
        tasks.extend([
            # TASK-A38-01
            create_task(
                description=(
                    "Funding capture models + cross-venue funding dislocations + basis trades. "
                    "Monitor funding rate opportunities."
                ),
                expected_output="Funding strategy documentation.",
                agent=agent,
            ),
            # TASK-A38-02
            create_task(
                description=(
                    "Capacity & unwind playbooks; prevent stranded hedges. "
                    "Documenteer exit procedures."
                ),
                expected_output="Capacity en unwind playbook.",
                agent=agent,
            ),
            # TASK-A38-03
            create_task(
                description=(
                    "Coördineer met treasury voor collateral efficiency. "
                    "Optimaliseer margin usage."
                ),
                expected_output="Collateral efficiency report.",
                agent=agent,
            ),
            # TASK-A38-04
            create_task(
                description=(
                    "Exploit extreme funding dislocations maximally: at very high positive funding rate "
                    "(shorts get paid) go aggressively short (within margin limits) en hedge via spot/options if needed."
                ),
                expected_output="Extreme funding exploitation report.",
                agent=agent,
            ),
        ])

    if "carry_trader_i" in agents:
        agent = agents["carry_trader_i"]
        tasks.extend([
            # TASK-A43-01
            create_task(
                description=(
                    "Voer uit funding strategies within leverage/margin policy. "
                    "Documenteer all positions."
                ),
                expected_output="Funding strategy execution log.",
                agent=agent,
            ),
            # TASK-A43-02
            create_task(
                description=(
                    "Monitor funding regime shifts + unwind triggers. "
                    "Alert op significant changes."
                ),
                expected_output="Funding regime monitoring report.",
                agent=agent,
            ),
            # TASK-A43-03
            create_task(
                description=(
                    "Report capacity + cost breakdown. "
                    "Track funding income en costs."
                ),
                expected_output="Capacity en cost breakdown.",
                agent=agent,
            ),
            # TASK-A43-04
            create_task(
                description=(
                    "Scale naar max size op extreme funding spreads (within risk appetite) naar maximize carry. "
                    "Zorg extra margin buffers naar prevent liquidations."
                ),
                expected_output="Carry maximization report.",
                agent=agent,
            ),
        ])

    if "carry_trader_ii" in agents:
        agent = agents["carry_trader_ii"]
        tasks.extend([
            # TASK-A44-01
            create_task(
                description=(
                    "Capture spreads between venues met strict venue limits. "
                    "Voer uit cross-venue carry."
                ),
                expected_output="Cross-venue carry execution log.",
                agent=agent,
            ),
            # TASK-A44-02
            create_task(
                description=(
                    "Unwind procedures op venue health deterioration. "
                    "Rapid exit when needed."
                ),
                expected_output="Venue health unwind procedure.",
                agent=agent,
            ),
            # TASK-A44-03
            create_task(
                description=(
                    "Voorkom stranded legs via predefined triggers. "
                    "Avoid asymmetric positions."
                ),
                expected_output="Stranded leg prevention report.",
                agent=agent,
            ),
            # TASK-A44-04
            create_task(
                description=(
                    "Explore arbitrage op new futures venues (under strict limit) naar benefit van inefficiencies "
                    "before others do. Monitor risk continuously."
                ),
                expected_output="New venue arbitrage exploration report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_microstructure_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures microstructure agents (39, 45-47)."""
    tasks = []

    if "microstructure_head" in agents:
        agent = agents["microstructure_head"]
        tasks.extend([
            # TASK-A39-01
            create_task(
                description=(
                    "Ontwikkel playbooks: breakout validation, absorption, stop runs, liquidity cliffs. "
                    "Ontwerp microstructure trading strategies."
                ),
                expected_output="Microstructure trading playbook.",
                agent=agent,
            ),
            # TASK-A39-02
            create_task(
                description=(
                    "Coach intraday traders; enforce no-trade regimes. "
                    "Onderhoud trading discipline."
                ),
                expected_output="Intraday coaching report.",
                agent=agent,
            ),
            # TASK-A39-03
            create_task(
                description=(
                    "Work met execution op order types + timing. "
                    "Optimaliseer execution quality."
                ),
                expected_output="Execution optimization report.",
                agent=agent,
            ),
            # TASK-A39-04
            create_task(
                description=(
                    "Hunt extreme orderflow moments (massive stop-hunts, large buyers/sellers) en design "
                    "microstructure trades naar profit van these (e.g., quick scalps during panic)."
                ),
                expected_output="Extreme orderflow opportunity report.",
                agent=agent,
            ),
        ])

    if "intraday_trader_i" in agents:
        agent = agents["intraday_trader_i"]
        tasks.extend([
            # TASK-A45-01
            create_task(
                description=(
                    "Play only high-liquidity setups; enforce max trades/day + max loss/day. "
                    "Strict discipline."
                ),
                expected_output="Intraday setup execution log.",
                agent=agent,
            ),
            # TASK-A45-02
            create_task(
                description=(
                    "Journal met setup tags + screenshots/notes. "
                    "Documenteer all trades."
                ),
                expected_output="Trade journal met evidence.",
                agent=agent,
            ),
            # TASK-A45-03
            create_task(
                description=(
                    "Coördineer timing met execution during stress. "
                    "Work met execution team."
                ),
                expected_output="Stress coordination log.",
                agent=agent,
            ),
            # TASK-A45-04
            create_task(
                description=(
                    "Gebruik 'house money': met vroege winst op de dag, trade iets agressiever "
                    "with part of that profit later (maintain discipline) voor higher PnL."
                ),
                expected_output="House money trading report.",
                agent=agent,
            ),
        ])

    if "intraday_trader_ii" in agents:
        agent = agents["intraday_trader_ii"]
        tasks.extend([
            # TASK-A46-01
            create_task(
                description=(
                    "Same discipline as Agent 45: max loss, max trades, liquid setups. "
                    "Focus op ETH-specific opportunities."
                ),
                expected_output="ETH intraday execution log.",
                agent=agent,
            ),
            # TASK-A46-02
            create_task(
                description=(
                    "Monitor correlation exposure vs BTC. "
                    "Track ETH/BTC relationship."
                ),
                expected_output="ETH/BTC correlation report.",
                agent=agent,
            ),
            # TASK-A46-03
            create_task(
                description=(
                    "Journal + review. "
                    "Continuous improvement."
                ),
                expected_output="ETH trading journal.",
                agent=agent,
            ),
            # TASK-A46-04
            create_task(
                description=(
                    "Play unique ETH drivers: op news where ETH moves independently of BTC, increase ETH position "
                    "for extra alpha (optionally partially hedged via BTC naar temper directional risk)."
                ),
                expected_output="ETH-specific opportunity report.",
                agent=agent,
            ),
        ])

    if "orderflow_trader" in agents:
        agent = agents["orderflow_trader"]
        tasks.extend([
            # TASK-A47-01
            create_task(
                description=(
                    "Play playbooks voor headline spikes, liquidation cascades, gap moves. "
                    "React naar orderflow events."
                ),
                expected_output="Orderflow event trading log.",
                agent=agent,
            ),
            # TASK-A47-02
            create_task(
                description=(
                    "Work closely met news analyst + CRO during volatility. "
                    "Coördineer op fast-moving situations."
                ),
                expected_output="Volatility coordination report.",
                agent=agent,
            ),
            # TASK-A47-03
            create_task(
                description=(
                    "Quick exits op regime flips; prevent gambling mode. "
                    "Onderhoud discipline."
                ),
                expected_output="Regime flip response log.",
                agent=agent,
            ),
            # TASK-A47-04
            create_task(
                description=(
                    "Prepare pre-set orders voor macro releases (e.g., stop buys/sells just outside current spread) "
                    "to react immediately op surprise. Minimize latency."
                ),
                expected_output="Macro event pre-positioning report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_research_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures research agents (40, 52-56)."""
    tasks = []

    if "research_head" in agents:
        agent = agents["research_head"]
        tasks.extend([
            # TASK-A40-01
            create_task(
                description=(
                    "Dashboards: funding extremes, OI changes, liquidation clusters. "
                    "Visualize derivatives data."
                ),
                expected_output="Futures research report met funding forecasts.",
                agent=agent,
            ),
            # TASK-A40-02
            create_task(
                description=(
                    "Analyseer event impact op perps: funding/OI behavior. "
                    "Track market structure changes."
                ),
                expected_output="Event impact analysis.",
                agent=agent,
            ),
            # TASK-A40-03
            create_task(
                description=(
                    "Alerts + wekelijkse docket naar CIO/Head Trading. "
                    "Keep leadership informed."
                ),
                expected_output="Wekelijkse research docket.",
                agent=agent,
            ),
            # TASK-A40-04
            create_task(
                description=(
                    "Use options market insights (OI, put/call ratios) as supplement: if options show extreme bias "
                    "vs futures metrics, signal arbitrage of contrarian trade opportunities."
                ),
                expected_output="Options/futures divergence report.",
                agent=agent,
            ),
        ])

    if "data_analyst" in agents:
        agent = agents["data_analyst"]
        tasks.extend([
            # TASK-A52-01
            create_task(
                description=(
                    "Bouw funding/OI dashboards + liquidation clusters + anomaly alerts. "
                    "Comprehensive derivatives data."
                ),
                expected_output="Derivatives data dashboard.",
                agent=agent,
            ),
            # TASK-A52-02
            create_task(
                description=(
                    "Data QA: venue differences + outliers. "
                    "Zorg data quality."
                ),
                expected_output="Data quality report.",
                agent=agent,
            ),
            # TASK-A52-03
            create_task(
                description=(
                    "Ondersteun desk situational awareness. "
                    "Keep traders informed."
                ),
                expected_output="Situational awareness report.",
                agent=agent,
            ),
            # TASK-A52-04
            create_task(
                description=(
                    "Integreer options data (OI, put/call ratios) in analyses voor fuller picture of market expectations, "
                    "even though we don't trade options directly."
                ),
                expected_output="Options data integration report.",
                agent=agent,
            ),
        ])

    if "macro_analyst" in agents:
        agent = agents["macro_analyst"]
        tasks.extend([
            # TASK-A53-01
            create_task(
                description=(
                    "Give risk mode advice (normal/reduced/defensive) based op macro calendar. "
                    "Set leverage guidance."
                ),
                expected_output="Risk mode recommendation.",
                agent=agent,
            ),
            # TASK-A53-02
            create_task(
                description=(
                    "Scenario planning around macro events. "
                    "Prepare voor different outcomes."
                ),
                expected_output="Macro scenario plan.",
                agent=agent,
            ),
            # TASK-A53-03
            create_task(
                description=(
                    "Communicate leverage discipline per regime. "
                    "Clear guidance op position sizing."
                ),
                expected_output="Leverage discipline communication.",
                agent=agent,
            ),
            # TASK-A53-04
            create_task(
                description=(
                    "Warn timely when high leverage is (ir)responsible: signal periods naar be more careful "
                    "(e.g., around Fed decisions) vs moments met macro tailwind naar increase risk."
                ),
                expected_output="Leverage timing advisory.",
                agent=agent,
            ),
        ])

    if "news_analyst" in agents:
        agent = agents["news_analyst"]
        tasks.extend([
            # TASK-A54-01
            create_task(
                description=(
                    "Alerts: 'what does this mean voor perps, funding, OI, liquidations?' "
                    "Translate news naar derivatives impact."
                ),
                expected_output="News naar derivatives impact alert.",
                agent=agent,
            ),
            # TASK-A54-02
            create_task(
                description=(
                    "Coördineer met Head Trading + CRO op headline risk. "
                    "Rapid response coordination."
                ),
                expected_output="Headline risk coordination report.",
                agent=agent,
            ),
            # TASK-A54-03
            create_task(
                description=(
                    "Ondersteun safer reaction trading. "
                    "Help traders navigate news-driven moves."
                ),
                expected_output="Reaction trading support.",
                agent=agent,
            ),
            # TASK-A54-04
            create_task(
                description=(
                    "Monitor traditional markets (S&P, USD, yields) too en translate large moves there naar crypto context. "
                    "Warn team when cross-asset signals point naar 'risk-off' of 'risk-on'."
                ),
                expected_output="Cross-asset signal report.",
                agent=agent,
            ),
        ])

    if "flow_analyst" in agents:
        agent = agents["flow_analyst"]
        tasks.extend([
            # TASK-A55-01
            create_task(
                description=(
                    "Alerts: large deposits, stablecoin mint/burn, whale moves. "
                    "Track significant flows."
                ),
                expected_output="Flow alert system.",
                agent=agent,
            ),
            # TASK-A55-02
            create_task(
                description=(
                    "Context: wanneer is flow een leading indicator vs ruis? "
                    "Filter signal van noise."
                ),
                expected_output="Flow signal analysis.",
                agent=agent,
            ),
            # TASK-A55-03
            create_task(
                description=(
                    "Timing input voor risk reduction/entries. "
                    "Lever actionable timing signals."
                ),
                expected_output="Flow timing advisory.",
                agent=agent,
            ),
            # TASK-A55-04
            create_task(
                description=(
                    "Use stablecoin on-chain data (large mints/burns) naar predict liquidity changes. "
                    "Warn if e.g. much stablecoins go naar exchanges (can affect funding rates)."
                ),
                expected_output="Stablecoin flow impact report.",
                agent=agent,
            ),
        ])

    if "quant_analyst" in agents:
        agent = agents["quant_analyst"]
        tasks.extend([
            # TASK-A56-01
            create_task(
                description=(
                    "Test new ideas (execution-realistic); write research memos. "
                    "Valideer strategy concepts."
                ),
                expected_output="Strategy research memo.",
                agent=agent,
            ),
            # TASK-A56-02
            create_task(
                description=(
                    "Post-launch evaluation: edge maintained of declining? "
                    "Monitor strategy health."
                ),
                expected_output="Post-launch strategy evaluation.",
                agent=agent,
            ),
            # TASK-A56-03
            create_task(
                description=(
                    "Continuous strategy improvement met evidence. "
                    "Data-driven optimization."
                ),
                expected_output="Strategy improvement report.",
                agent=agent,
            ),
            # TASK-A56-04
            create_task(
                description=(
                    "Follow external quant developments (papers, forums) voor fresh ideas. "
                    "Test promising concepts op our data naar discover new edge."
                ),
                expected_output="External quant research report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_execution_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures execution agents (41, 51)."""
    tasks = []

    if "execution_head" in agents:
        agent = agents["execution_head"]
        tasks.extend([
            # TASK-A41-01
            create_task(
                description=(
                    "KPIs: slippage, reject rate, adverse selection, fill quality. "
                    "Track en improve execution."
                ),
                expected_output="Futures execution KPI report.",
                agent=agent,
            ),
            # TASK-A41-02
            create_task(
                description=(
                    "Routing + ordertype policy per strategy. "
                    "Documenteer execution procedures."
                ),
                expected_output="Execution policy document.",
                agent=agent,
            ),
            # TASK-A41-03
            create_task(
                description=(
                    "Stress execution playbooks: rapid unwind without panic. "
                    "Prepare voor emergency situations."
                ),
                expected_output="Stress execution playbook.",
                agent=agent,
            ),
            # TASK-A41-04
            create_task(
                description=(
                    "Try new execution channels: use RFQ/dark pool voor large orders, apply maker-only vs taker "
                    "smartly per market condition, minimize fees using reduce-only where possible."
                ),
                expected_output="Execution channel optimization report.",
                agent=agent,
            ),
        ])

    if "unwind_specialist" in agents:
        agent = agents["unwind_specialist"]
        tasks.extend([
            # TASK-A51-01
            create_task(
                description=(
                    "Voer uit desk-wide risk reduction op kill-switch/drawdown. "
                    "Emergency position reduction."
                ),
                expected_output="Unwind execution plan en report.",
                agent=agent,
            ),
            # TASK-A51-02
            create_task(
                description=(
                    "Beheer hedge overlays: reduce net exposure, reduce liquidation risk. "
                    "Protective hedging."
                ),
                expected_output="Hedge overlay management report.",
                agent=agent,
            ),
            # TASK-A51-03
            create_task(
                description=(
                    "Coördineer met treasury/margin specialist. "
                    "Cross-team coordination."
                ),
                expected_output="Treasury coordination log.",
                agent=agent,
            ),
            # TASK-A51-04
            create_task(
                description=(
                    "Practice periodic crisis-unwind drills: simulate complete portfolio unwind naar test processes "
                    "and prepare traders voor real emergency situations."
                ),
                expected_output="Crisis unwind drill report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_swing_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures swing agents (48-50)."""
    tasks = []

    if "swing_head" in agents:
        agent = agents["swing_head"]
        tasks.extend([
            # TASK-A48-01
            create_task(
                description=(
                    "Regime-aware sizing + clear invalidation. "
                    "Adjust position size naar market conditions."
                ),
                expected_output="Directional trade thesis met leverage plan.",
                agent=agent,
            ),
            # TASK-A48-02
            create_task(
                description=(
                    "Reduce risk around events. "
                    "Lower exposure before major events."
                ),
                expected_output="Event risk reduction report.",
                agent=agent,
            ),
            # TASK-A48-03
            create_task(
                description=(
                    "Work met macro/regime input. "
                    "Align trades met macro view."
                ),
                expected_output="Macro-aligned trade plan.",
                agent=agent,
            ),
            # TASK-A48-04
            create_task(
                description=(
                    "Pyramid into strong trends: add naar winning position op pullbacks met built-up profit "
                    "(update stop), increasing profit potential without extra own-capital risk."
                ),
                expected_output="Trend pyramiding report.",
                agent=agent,
            ),
        ])

    if "swing_btc" in agents:
        agent = agents["swing_btc"]
        tasks.extend([
            # TASK-A49-01
            create_task(
                description=(
                    "Play only confirmed range regimes; avoid counter-trend trades. "
                    "Strict setup criteria."
                ),
                expected_output="BTC swing execution log.",
                agent=agent,
            ),
            # TASK-A49-02
            create_task(
                description=(
                    "Strict risk/reward checks before entry. "
                    "Only high-quality setups."
                ),
                expected_output="Risk/reward analysis.",
                agent=agent,
            ),
            # TASK-A49-03
            create_task(
                description=(
                    "Onderhoud stops/targets discipline. "
                    "No moving stops outside plan."
                ),
                expected_output="Discipline compliance report.",
                agent=agent,
            ),
            # TASK-A49-04
            create_task(
                description=(
                    "At clear range extremes take slightly larger position than normal (still small enough). "
                    "Split position naar take partial profit en let rest mean revert."
                ),
                expected_output="Range extreme positioning report.",
                agent=agent,
            ),
        ])

    if "curve_trader" in agents:
        agent = agents["curve_trader"]
        tasks.extend([
            # TASK-A50-01
            create_task(
                description=(
                    "Identificeer curve anomalies; execute spreads met low slippage. "
                    "Futures curve trading."
                ),
                expected_output="Curve spread execution log.",
                agent=agent,
            ),
            # TASK-A50-02
            create_task(
                description=(
                    "Stress test: funding spikes/volatility shocks impact. "
                    "Model adverse scenarios."
                ),
                expected_output="Curve stress test report.",
                agent=agent,
            ),
            # TASK-A50-03
            create_task(
                description=(
                    "Definieer capacity + exit plans. "
                    "Documenteer position limits en exits."
                ),
                expected_output="Capacity en exit planning.",
                agent=agent,
            ),
            # TASK-A50-04
            create_task(
                description=(
                    "Exploit new contracts: if new futures (extra expiries) launch met dislocations, "
                    "test those spreads small-scale en scale up if liquidity proves adequate."
                ),
                expected_output="New contract opportunity report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_risk_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures risk agents (57-59)."""
    tasks = []

    if "risk_monitor" in agents:
        agent = agents["risk_monitor"]
        tasks.extend([
            # TASK-A57-01
            create_task(
                description=(
                    "Monitor net/gross exposure, leverage, concentration, scenario shocks. "
                    "Comprehensive risk monitoring."
                ),
                expected_output="Margin en liquidation monitoring report.",
                agent=agent,
            ),
            # TASK-A57-02
            create_task(
                description=(
                    "Escalate at thresholds; advise risk scaling. "
                    "Immediate alerts."
                ),
                expected_output="Threshold escalation report.",
                agent=agent,
            ),
            # TASK-A57-03
            create_task(
                description=(
                    "Ondersteun risk-adjusted allocation input. "
                    "Lever data voor allocation decisions."
                ),
                expected_output="Risk-adjusted allocation support.",
                agent=agent,
            ),
            # TASK-A57-04
            create_task(
                description=(
                    "Proactively signal room voor more risk: if exposures are low en market conditions favorable, "
                    "suggest increasing leverage/positions voor more profit potential (with CRO approval)."
                ),
                expected_output="Risk capacity advisory.",
                agent=agent,
            ),
        ])

    if "margin_analyst" in agents:
        agent = agents["margin_analyst"]
        tasks.extend([
            # TASK-A58-01
            create_task(
                description=(
                    "Definieer collateral buffers + liquidation distance targets. "
                    "Set margin safety levels."
                ),
                expected_output="Margin optimization report.",
                agent=agent,
            ),
            # TASK-A58-02
            create_task(
                description=(
                    "Monitor margin changes per venue; initiate collateral moves. "
                    "Active margin management."
                ),
                expected_output="Margin monitoring report.",
                agent=agent,
            ),
            # TASK-A58-03
            create_task(
                description=(
                    "Run war-room procedures during extreme moves. "
                    "Crisis management."
                ),
                expected_output="War-room procedure report.",
                agent=agent,
            ),
            # TASK-A58-04
            create_task(
                description=(
                    "Simulate flash crashes & venue outages: test if collaterals are sufficient en procedures work. "
                    "Adjust collateral buffers of war-room procedures based op simulation findings."
                ),
                expected_output="Flash crash simulation report.",
                agent=agent,
            ),
        ])

    if "liquidation_specialist" in agents:
        agent = agents["liquidation_specialist"]
        tasks.extend([
            # TASK-A59-01
            create_task(
                description=(
                    "Pre-live review + kill criteria definition. "
                    "Strategy validation before launch."
                ),
                expected_output="Liquidation risk analysis.",
                agent=agent,
            ),
            # TASK-A59-02
            create_task(
                description=(
                    "Post-launch review op drift + regime dependency. "
                    "Ongoing strategy monitoring."
                ),
                expected_output="Post-launch review report.",
                agent=agent,
            ),
            # TASK-A59-03
            create_task(
                description=(
                    "Onderhoud audit trail of approvals. "
                    "Documenteer all decisions."
                ),
                expected_output="Approval audit trail.",
                agent=agent,
            ),
            # TASK-A59-04
            create_task(
                description=(
                    "Keep eye op potential: don't blindly block high-risk strategy, but require mitigations "
                    "(e.g., smaller size, extra hedges) zodat veelbelovende ideeën getest kunnen worden zonder ongecontroleerd risico."
                ),
                expected_output="Controlled risk approval framework.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_operations_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal taken op voor Futures operations agents (60-64)."""
    tasks = []

    if "controller" in agents:
        agent = agents["controller"]
        tasks.extend([
            # TASK-A60-01
            create_task(
                description=(
                    "Reconcile positions, funding, fees; PnL attribution per pod. "
                    "Accurate financial tracking."
                ),
                expected_output="Dagelijkse position reconciliation.",
                agent=agent,
            ),
            # TASK-A60-02
            create_task(
                description=(
                    "Signal breaks directly naar COO/CRO. "
                    "Immediate escalation."
                ),
                expected_output="Break escalation report.",
                agent=agent,
            ),
            # TASK-A60-03
            create_task(
                description=(
                    "Monitor funding misbookings/venue issues. "
                    "Data quality monitoring."
                ),
                expected_output="Funding en venue issue log.",
                agent=agent,
            ),
            # TASK-A60-04
            create_task(
                description=(
                    "Dissect PnL dagelijkse: which part came van price movement vs funding vs fees. "
                    "Signaleer opmerkelijke kosten (zoals onverwacht hoge fees) zodat team kan handelen (e.g., tier upgrade)."
                ),
                expected_output="PnL attribution analysis.",
                agent=agent,
            ),
        ])

    if "treasury" in agents:
        agent = agents["treasury"]
        tasks.extend([
            # TASK-A61-01
            create_task(
                description=(
                    "Beheer buffers per venue; optimize idle vs safety. "
                    "Balanceer liquidity en returns."
                ),
                expected_output="Venue buffer management.",
                agent=agent,
            ),
            # TASK-A61-02
            create_task(
                description=(
                    "Coördineer collateral moves via approvals. "
                    "Secure fund transfers."
                ),
                expected_output="Collateral movement log.",
                agent=agent,
            ),
            # TASK-A61-03
            create_task(
                description=(
                    "Prepare voor weekend risk + funding spikes. "
                    "Weekend contingency planning."
                ),
                expected_output="Weekend risk preparation.",
                agent=agent,
            ),
            # TASK-A61-04
            create_task(
                description=(
                    "Give idle collateral a useful role: if possible put surplus margin naar work "
                    "(e.g., via low-risk earn/lending) naar earn something op unused funds (CRO approval needed)."
                ),
                expected_output="Idle collateral yield report.",
                agent=agent,
            ),
        ])

    if "security" in agents:
        agent = agents["security"]
        tasks.extend([
            # TASK-A62-01
            create_task(
                description=(
                    "Role-based access, key rotation, device policies. "
                    "Comprehensive security controls."
                ),
                expected_output="Security controls report.",
                agent=agent,
            ),
            # TASK-A62-02
            create_task(
                description=(
                    "Incident runbooks: compromise, abnormal orders, suspicious API activity. "
                    "Response procedures."
                ),
                expected_output="Incident runbook.",
                agent=agent,
            ),
            # TASK-A62-03
            create_task(
                description=(
                    "Voorkom fat-finger via permissions + controls. "
                    "Error prevention."
                ),
                expected_output="Fat-finger prevention report.",
                agent=agent,
            ),
            # TASK-A62-04
            create_task(
                description=(
                    "Adapt API key management voor high-frequency: use segmented keys per strategy met limited rights "
                    "and frequent rotation, so potential compromise causes minimal damage."
                ),
                expected_output="API key security report.",
                agent=agent,
            ),
        ])

    if "compliance" in agents:
        agent = agents["compliance"]
        tasks.extend([
            # TASK-A63-01
            create_task(
                description=(
                    "Policies + training; surveillance + escalation. "
                    "Compliance framework."
                ),
                expected_output="Compliance policy document.",
                agent=agent,
            ),
            # TASK-A63-02
            create_task(
                description=(
                    "Monitor compliance met venue rules + internal leverage policy. "
                    "Rule enforcement."
                ),
                expected_output="Rule compliance report.",
                agent=agent,
            ),
            # TASK-A63-03
            create_task(
                description=(
                    "Zorg documentation/recordkeeping. "
                    "Audit trail maintenance."
                ),
                expected_output="Documenteeration compliance.",
                agent=agent,
            ),
            # TASK-A63-04
            create_task(
                description=(
                    "Watch aggressive trades vs market rules: prevent intensive strategies van crossing "
                    "market abuse lines (spoofing, squeezes) en intervene immediately op doubt."
                ),
                expected_output="Market abuse prevention report.",
                agent=agent,
            ),
        ])

    if "ops" in agents:
        agent = agents["ops"]
        tasks.extend([
            # TASK-A64-01
            create_task(
                description=(
                    "Funding/fees processing; resolve breaks; coordinate met controller/treasury. "
                    "Operational workflow."
                ),
                expected_output="Operational processing log.",
                agent=agent,
            ),
            # TASK-A64-02
            create_task(
                description=(
                    "Ondersteun during volatility spikes + venue incidents. "
                    "Crisis support."
                ),
                expected_output="Volatility support report.",
                agent=agent,
            ),
            # TASK-A64-03
            create_task(
                description=(
                    "Zorg trading can continue during ops friction. "
                    "Business continuity."
                ),
                expected_output="Business continuity report.",
                agent=agent,
            ),
            # TASK-A64-04
            create_task(
                description=(
                    "Automate monitoring: detect every deviation (missing funding posting, trade break) immediately "
                    "so you resolve it before it affects PnL of trader focus."
                ),
                expected_output="Automated monitoring report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Haal alle Futures desk taken op gegeven een dictionary van agents.

    Args:
        agents: Dictionary die agent keys mapt naar Agent instances.

    Returns:
        Lijst van alle Futures desk taken.
    """
    tasks = []
    tasks.extend(get_futures_leadership_tasks(agents))
    tasks.extend(get_futures_systematic_tasks(agents))
    tasks.extend(get_futures_carry_tasks(agents))
    tasks.extend(get_futures_microstructure_tasks(agents))
    tasks.extend(get_futures_research_tasks(agents))
    tasks.extend(get_futures_execution_tasks(agents))
    tasks.extend(get_futures_swing_tasks(agents))
    tasks.extend(get_futures_risk_tasks(agents))
    tasks.extend(get_futures_operations_tasks(agents))
    return tasks
