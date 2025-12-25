"""Spot desk agent tasks for QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_spot_leadership_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot leadership agents (01-04)."""
    tasks = []

    if "cio" in agents:
        cio = agents["cio"]
        tasks.extend([
            # TASK-A01-01
            create_task(
                description=(
                    "Define tradable universe, exposure caps, and allocation per strategy. "
                    "Document current allocations and any changes."
                ),
                expected_output="Universe and allocation document with exposure limits.",
                agent=cio,
            ),
            # TASK-A01-02
            create_task(
                description=(
                    "Set risk budgets per pod: systematic, discretionary, arb, event, intraday. "
                    "Ensure budgets align with overall portfolio risk."
                ),
                expected_output="Risk budget allocation per pod with rationale.",
                agent=cio,
            ),
            # TASK-A01-03
            create_task(
                description=(
                    "Run monthly allocation and kill/scale decisions. "
                    "Base decisions on performance data and market conditions."
                ),
                expected_output="Monthly allocation decision report.",
                agent=cio,
            ),
            # TASK-A01-04
            create_task(
                description=(
                    "Increase allocation to high-conviction strategies: scale winning strategies "
                    "(within risk limits) to capture extra alpha."
                ),
                expected_output="High-conviction allocation increase report.",
                agent=cio,
            ),
        ])

    if "head_trading" in agents:
        head = agents["head_trading"]
        tasks.extend([
            # TASK-A02-01
            create_task(
                description=(
                    "Daily desk briefing: focus list, levels, events, risk mode. "
                    "Communicate key information to all traders."
                ),
                expected_output="Daily briefing document with all key information.",
                agent=head,
            ),
            # TASK-A02-02
            create_task(
                description=(
                    "Monitor playbook discipline and trade quality. "
                    "Identify and address overtrading or rule violations."
                ),
                expected_output="Trade quality report with any issues flagged.",
                agent=head,
            ),
            # TASK-A02-03
            create_task(
                description=(
                    "Conduct post-trade reviews and enforce journaling. "
                    "Identify patterns for improvement."
                ),
                expected_output="Post-trade review summary with improvement areas.",
                agent=head,
            ),
            # TASK-A02-04
            create_task(
                description=(
                    "Let traders bet aggressively on A-setup trades (within limits) "
                    "and minimize time on marginal opportunities."
                ),
                expected_output="A-setup trade focus report with trader performance.",
                agent=head,
            ),
        ])

    if "cro" in agents:
        cro = agents["cro"]
        tasks.extend([
            # TASK-A03-01
            create_task(
                description=(
                    "Design risk framework: exposure caps, liquidity tiers, max drawdown, escalations. "
                    "Document all limits and thresholds."
                ),
                expected_output="Spot risk framework document.",
                agent=cro,
            ),
            # TASK-A03-02
            create_task(
                description=(
                    "Real-time monitoring and alerts. Enforce risk reductions at thresholds."
                ),
                expected_output="Risk monitoring report with any actions taken.",
                agent=cro,
            ),
            # TASK-A03-03
            create_task(
                description=(
                    "Sign off new spot strategies. Conduct pre-mortem and failure mode analysis."
                ),
                expected_output="Strategy sign-off with risk analysis.",
                agent=cro,
            ),
            # TASK-A03-04
            create_task(
                description=(
                    "Allow temporarily higher risk for exceptional opportunities (within agreed extra margins) "
                    "to facilitate extraordinary gains without breaking the risk framework."
                ),
                expected_output="Exceptional opportunity risk approval report.",
                agent=cro,
            ),
        ])

    if "coo" in agents:
        coo = agents["coo"]
        tasks.extend([
            # TASK-A04-01
            create_task(
                description=(
                    "Set up daily reconciliation, approvals, and incident runbooks. "
                    "Document all operational procedures."
                ),
                expected_output="Operational procedures documentation.",
                agent=coo,
            ),
            # TASK-A04-02
            create_task(
                description=(
                    "Manage operational SLAs with exchanges and custody. "
                    "Track performance against SLAs."
                ),
                expected_output="SLA performance report.",
                agent=coo,
            ),
            # TASK-A04-03
            create_task(
                description=(
                    "Maintain audit trail and enforce separation of duties. "
                    "Ensure all actions are properly logged."
                ),
                expected_output="Audit trail and separation of duties report.",
                agent=coo,
            ),
            # TASK-A04-04
            create_task(
                description=(
                    "Accelerate onboarding of new venues/assets during opportunities: "
                    "ensure fast account/approval setup without violating control rules."
                ),
                expected_output="Fast-track onboarding report.",
                agent=coo,
            ),
        ])

    return tasks


def get_spot_systematic_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot systematic agents (05, 10)."""
    tasks = []

    if "systematic_head" in agents:
        agent = agents["systematic_head"]
        tasks.extend([
            # TASK-A05-01
            create_task(
                description=(
                    "Design and maintain signal library for systematic strategies. "
                    "Include trend, momentum, and mean reversion signals."
                ),
                expected_output="Signal library documentation with performance metrics.",
                agent=agent,
            ),
            # TASK-A05-02
            create_task(
                description=(
                    "Write strategy specs for dev team: rules, data, risk, execution assumptions. "
                    "Ensure clear documentation for implementation."
                ),
                expected_output="Strategy specification document.",
                agent=agent,
            ),
            # TASK-A05-03
            create_task(
                description=(
                    "Monthly model review: drift detection + kill/scale proposals. "
                    "Identify strategies losing edge."
                ),
                expected_output="Model review report with recommendations.",
                agent=agent,
            ),
            # TASK-A05-04
            create_task(
                description=(
                    "Use AI/ML and alternative data (sentiment, macro) to find new signals. "
                    "Validate rigorously and pilot for extra alpha."
                ),
                expected_output="New signal discovery report with validation results.",
                agent=agent,
            ),
        ])

    if "systematic_operator" in agents:
        agent = agents["systematic_operator"]
        tasks.extend([
            # TASK-A10-01
            create_task(
                description=(
                    "Run signals, check data quality, execute rebalances. "
                    "Log all deviations and fixes."
                ),
                expected_output="Daily systematic operations log.",
                agent=agent,
            ),
            # TASK-A10-02
            create_task(
                description=(
                    "Pause strategy on anomalies per SOP and report to Agent 05/03. "
                    "Document all pauses with rationale."
                ),
                expected_output="Anomaly pause log with escalations.",
                agent=agent,
            ),
            # TASK-A10-03
            create_task(
                description=(
                    "Maintain deviation and fix logbook. "
                    "Track all operational issues and resolutions."
                ),
                expected_output="Deviation and fix logbook.",
                agent=agent,
            ),
            # TASK-A10-04
            create_task(
                description=(
                    "Provide continuous feedback to quant devs on execution frictions or data errors "
                    "so models/strategies can be improved for more profit."
                ),
                expected_output="Feedback report to quant team.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_discretionary_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot discretionary/swing agents (06, 13, 14)."""
    tasks = []

    if "discretionary_head" in agents:
        agent = agents["discretionary_head"]
        tasks.extend([
            # TASK-A06-01
            create_task(
                description=(
                    "Build thesis-driven trades (days-weeks) with strict invalidation. "
                    "Document entry, exit, and stop levels."
                ),
                expected_output="Trade thesis document with entry/exit criteria.",
                agent=agent,
            ),
            # TASK-A06-02
            create_task(
                description=(
                    "Integrate research: tokenomics, unlocks, flows, catalysts. "
                    "Use data to inform trade decisions."
                ),
                expected_output="Research integration report.",
                agent=agent,
            ),
            # TASK-A06-03
            create_task(
                description=(
                    "Manage position management: scale in/out, trailing stops, profit protection. "
                    "Document all position changes."
                ),
                expected_output="Position management log.",
                agent=agent,
            ),
            # TASK-A06-04
            create_task(
                description=(
                    "Hunt niche tokens/narratives early and take small positions (conscious high risk) "
                    "for potential outsized gains if thesis plays out. Exit direct on failure."
                ),
                expected_output="Niche token opportunity report.",
                agent=agent,
            ),
        ])

    if "swing_trader_i" in agents:
        agent = agents["swing_trader_i"]
        tasks.extend([
            # TASK-A13-01
            create_task(
                description=(
                    "Plan trend continuation/pullback trades with invalidation levels. "
                    "Document setup criteria."
                ),
                expected_output="Trend trade plan.",
                agent=agent,
            ),
            # TASK-A13-02
            create_task(
                description=(
                    "Combine levels with flows/volume (no indicator blindness). "
                    "Use multiple confirmation signals."
                ),
                expected_output="Multi-signal confluence report.",
                agent=agent,
            ),
            # TASK-A13-03
            create_task(
                description=(
                    "Build scenario trade plans: base/bull/bear cases. "
                    "Prepare for different outcomes."
                ),
                expected_output="Scenario trade plan document.",
                agent=agent,
            ),
            # TASK-A13-04
            create_task(
                description=(
                    "Let winners run: increase position or widen trailing stop when trade is convincingly winning "
                    "to maximize trend profit (maintain stop discipline)."
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
                    "Build theme baskets + sector rotation (L2/AI/DeFi) within liquidity tiers. "
                    "Document basket composition."
                ),
                expected_output="Theme basket composition document.",
                agent=agent,
            ),
            # TASK-A14-02
            create_task(
                description=(
                    "Maintain strict sizing (never outsized in illiquid assets). "
                    "Document sizing rationale."
                ),
                expected_output="Position sizing report.",
                agent=agent,
            ),
            # TASK-A14-03
            create_task(
                description=(
                    "Plan unlock/supply events with research. "
                    "Prepare positions ahead of events."
                ),
                expected_output="Event preparation plan.",
                agent=agent,
            ),
            # TASK-A14-04
            create_task(
                description=(
                    "Allocate limited capital to emerging alts (micro-caps or new sector) "
                    "for potential high gains. Strict exit if liquidity drops."
                ),
                expected_output="Emerging alt allocation report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_arbitrage_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot arbitrage agents (07, 11, 12)."""
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
                expected_output="Arbitrage opportunity report with capacity analysis.",
                agent=agent,
            ),
            # TASK-A07-02
            create_task(
                description=(
                    "Define venue filters: withdrawal reliability, limits, liquidity. "
                    "Maintain approved venue list."
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
                    "Explore arbitrage on new/illiquid markets (including DEX if possible) "
                    "with limited capital to profit before competitors."
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
                    "Scan spreads and execute legs according to execution policy. "
                    "Document all executions."
                ),
                expected_output="Spread execution log.",
                agent=agent,
            ),
            # TASK-A11-02
            create_task(
                description=(
                    "Monitor venue limits + settlement windows. "
                    "Ensure timely settlement."
                ),
                expected_output="Venue limit monitoring report.",
                agent=agent,
            ),
            # TASK-A11-03
            create_task(
                description=(
                    "Report capacity + frictions (fees, slippage, downtime). "
                    "Identify bottlenecks."
                ),
                expected_output="Capacity and friction report.",
                agent=agent,
            ),
            # TASK-A11-04
            create_task(
                description=(
                    "Scale successful arb trades: increase volume on stable spreads "
                    "and expand to new asset pairs if performance is consistent."
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
                    "Identify and execute triangular opportunities within strict limits. "
                    "Document all opportunities."
                ),
                expected_output="Triangular arb execution log.",
                agent=agent,
            ),
            # TASK-A12-02
            create_task(
                description=(
                    "Trade stablecoin spreads with predefined depeg rules. "
                    "Monitor stablecoin health."
                ),
                expected_output="Stablecoin spread trading log.",
                agent=agent,
            ),
            # TASK-A12-03
            create_task(
                description=(
                    "Monitor settlement risk + venue health with ops. "
                    "Escalate issues immediately."
                ),
                expected_output="Settlement and venue health report.",
                agent=agent,
            ),
            # TASK-A12-04
            create_task(
                description=(
                    "Play stablecoin depeg situations opportunistically (quick in/out for recovery) "
                    "and experiment with triangular arb on new pairings where liquidity increases."
                ),
                expected_output="Opportunistic depeg trading report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_research_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot research agents (08, 20-24)."""
    tasks = []

    if "research_head" in agents:
        agent = agents["research_head"]
        tasks.extend([
            # TASK-A08-01
            create_task(
                description=(
                    "Build catalyst calendar: unlocks, upgrades, listings, treasury moves, governance. "
                    "Track all upcoming events."
                ),
                expected_output="Catalyst calendar and watchlist with scores.",
                agent=agent,
            ),
            # TASK-A08-02
            create_task(
                description=(
                    "Produce watchlists with tradeability score (liquidity, supply risk, narrative, flows). "
                    "Rank opportunities."
                ),
                expected_output="Ranked watchlist with tradeability scores.",
                agent=agent,
            ),
            # TASK-A08-03
            create_task(
                description=(
                    "Publish 'opportunity docket' + real-time alerts with impact assessment. "
                    "Keep desk informed."
                ),
                expected_output="Opportunity docket publication.",
                agent=agent,
            ),
            # TASK-A08-04
            create_task(
                description=(
                    "Monitor social media & dev community for hype (trending Twitter/Reddit, GitHub activity) "
                    "and alert trading team on early signals."
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
                    "Build dashboards: inflow/outflow, whale deposits, cohort behavior. "
                    "Visualize on-chain data."
                ),
                expected_output="On-chain analytics dashboard.",
                agent=agent,
            ),
            # TASK-A20-02
            create_task(
                description=(
                    "Create alerts with context (noise vs signal). "
                    "Filter meaningful signals."
                ),
                expected_output="Contextual alert system.",
                agent=agent,
            ),
            # TASK-A20-03
            create_task(
                description=(
                    "Run post-mortems: when signal failed and why. "
                    "Learn from misses."
                ),
                expected_output="Post-mortem analysis report.",
                agent=agent,
            ),
            # TASK-A20-04
            create_task(
                description=(
                    "Convert on-chain signals directly to trade actions: e.g., large whale deposit -> "
                    "warn for short, large stablecoin burn -> signal for potential rally."
                ),
                expected_output="On-chain signal to trade action mapping.",
                agent=agent,
            ),
        ])

    if "tokenomics_analyst" in agents:
        agent = agents["tokenomics_analyst"]
        tasks.extend([
            # TASK-A21-01
            create_task(
                description=(
                    "Build supply shock calendar with impact scores (unlock vs liquidity). "
                    "Track all supply events."
                ),
                expected_output="Supply shock calendar with impact scores.",
                agent=agent,
            ),
            # TASK-A21-02
            create_task(
                description=(
                    "Identify mechanical flows: vesting dumps, emissions pressure. "
                    "Predict supply changes."
                ),
                expected_output="Mechanical flows analysis.",
                agent=agent,
            ),
            # TASK-A21-03
            create_task(
                description=(
                    "Warn for governance/treasury risks. "
                    "Flag potential negative events."
                ),
                expected_output="Governance/treasury risk report.",
                agent=agent,
            ),
            # TASK-A21-04
            create_task(
                description=(
                    "Hunt tokens with extreme tokenomics events (large unlocks, buybacks) coming "
                    "and advise short/long strategies for extra alpha."
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
                    "Run ad-hoc studies: reactions to unlocks/listings. "
                    "Analyze historical patterns."
                ),
                expected_output="Ad-hoc quantitative study.",
                agent=agent,
            ),
            # TASK-A22-02
            create_task(
                description=(
                    "Maintain watchlist scoring + sector dashboards. "
                    "Keep metrics updated."
                ),
                expected_output="Watchlist scoring update.",
                agent=agent,
            ),
            # TASK-A22-03
            create_task(
                description=(
                    "Data QA: outliers + venue inconsistencies. "
                    "Ensure data quality."
                ),
                expected_output="Data QA report.",
                agent=agent,
            ),
            # TASK-A22-04
            create_task(
                description=(
                    "Build sentiment/trend indexes from alt-data (Twitter volume, Google trends) "
                    "to detect potential price triggers. Share with traders."
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
                    "Analyze implications."
                ),
                expected_output="News impact assessment.",
                agent=agent,
            ),
            # TASK-A23-03
            create_task(
                description=(
                    "Communicate in short, action-oriented bullets. "
                    "Clear and concise updates."
                ),
                expected_output="Action-oriented news summary.",
                agent=agent,
            ),
            # TASK-A23-04
            create_task(
                description=(
                    "Monitor crowd sentiment (Twitter/Reddit/Telegram) for extreme mood. "
                    "Give contrarian or trend-following advice at manic or panic signals."
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
                    "Define regime labels + guidance (normal/reduced risk). "
                    "Set risk mode recommendations."
                ),
                expected_output="Regime label definition.",
                agent=agent,
            ),
            # TASK-A24-02
            create_task(
                description=(
                    "Build calendar: macro events + weekend liquidity risk. "
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
                    "Communicate clearly when 'risk-on' vs 'risk-off': give desks green light to go full "
                    "in favorable macro climate, and brake when macro turns against. Update immediately after events."
                ),
                expected_output="Risk-on/off communication.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_execution_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot execution agents (09, 16, 17)."""
    tasks = []

    if "execution_head" in agents:
        agent = agents["execution_head"]
        tasks.extend([
            # TASK-A09-01
            create_task(
                description=(
                    "Set up execution KPIs: implementation shortfall, reject rate, adverse selection. "
                    "Track and improve performance."
                ),
                expected_output="Execution KPI report with improvement actions.",
                agent=agent,
            ),
            # TASK-A09-02
            create_task(
                description=(
                    "Define maker/taker policy + routing rules + large order playbooks. "
                    "Document execution procedures."
                ),
                expected_output="Execution policy document.",
                agent=agent,
            ),
            # TASK-A09-03
            create_task(
                description=(
                    "Continuously improve fills/costs across venues. "
                    "Optimize execution quality."
                ),
                expected_output="Fill quality improvement report.",
                agent=agent,
            ),
            # TASK-A09-04
            create_task(
                description=(
                    "Integrate automated execution algos (TWAP/VWAP) and explore dark liquidity sources "
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
                    "Focus on high-probability setups."
                ),
                expected_output="Intraday setup execution log.",
                agent=agent,
            ),
            # TASK-A16-02
            create_task(
                description=(
                    "Journal with setup tags + execution notes. "
                    "Document all trades."
                ),
                expected_output="Trade journal with tags.",
                agent=agent,
            ),
            # TASK-A16-03
            create_task(
                description=(
                    "Maintain stop discipline (no moving stops outside playbook). "
                    "Strict risk management."
                ),
                expected_output="Stop discipline compliance report.",
                agent=agent,
            ),
            # TASK-A16-04
            create_task(
                description=(
                    "Increase position size slightly when 'in the zone' and market trend goes your way "
                    "to make extra PnL (but keep daily loss limit)."
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
                    "Coordinate larger entries/exits with execution. "
                    "Work with execution team."
                ),
                expected_output="Coordinated execution log.",
                agent=agent,
            ),
            # TASK-A17-03
            create_task(
                description=(
                    "Daily self-review + desk review with Agent 02. "
                    "Continuous improvement."
                ),
                expected_output="Daily review document.",
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
    """Get tasks for Spot event-driven agent (15)."""
    tasks = []

    if "event_trader" in agents:
        agent = agents["event_trader"]
        tasks.extend([
            # TASK-A15-01
            create_task(
                description=(
                    "Write pre-event plan: entry, invalidation, hedge/exit rules. "
                    "Document complete event strategy."
                ),
                expected_output="Event trading playbook.",
                agent=agent,
            ),
            # TASK-A15-02
            create_task(
                description=(
                    "Manage post-event: 'sell the news' + volatility regime. "
                    "Handle event aftermath."
                ),
                expected_output="Post-event management report.",
                agent=agent,
            ),
            # TASK-A15-03
            create_task(
                description=(
                    "Use news/on-chain alerts for confirmation. "
                    "Validate event thesis."
                ),
                expected_output="Event confirmation checklist.",
                agent=agent,
            ),
            # TASK-A15-04
            create_task(
                description=(
                    "Sometimes pre-position for big events (with small risk) when own analysis differs "
                    "from consensus, for potential outsized gain. Exit immediately on failure."
                ),
                expected_output="Contrarian event positioning report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_mm_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot market making agent (18)."""
    tasks = []

    if "mm_supervisor" in agents:
        agent = agents["mm_supervisor"]
        tasks.extend([
            # TASK-A18-01
            create_task(
                description=(
                    "Set quoting rules: spreads, inventory bands, stop rules. "
                    "Define MM parameters."
                ),
                expected_output="Market making rules and inventory report.",
                agent=agent,
            ),
            # TASK-A18-02
            create_task(
                description=(
                    "Monitor inventory; force flattening on regime change. "
                    "Manage inventory risk."
                ),
                expected_output="Inventory monitoring report.",
                agent=agent,
            ),
            # TASK-A18-03
            create_task(
                description=(
                    "Evaluate PnL source: spread capture vs adverse selection. "
                    "Analyze MM profitability."
                ),
                expected_output="MM PnL attribution report.",
                agent=agent,
            ),
            # TASK-A18-04
            create_task(
                description=(
                    "Focus on volatile liquid pairs with wide spreads for more spread capture. "
                    "Reduce inventory quickly on spikes to avoid slippage. Maximize MM PnL."
                ),
                expected_output="MM optimization report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_risk_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot risk agents (19, 25-27)."""
    tasks = []

    if "inventory_coordinator" in agents:
        agent = agents["inventory_coordinator"]
        tasks.extend([
            # TASK-A19-01
            create_task(
                description=(
                    "Daily inventory checks: concentrations, liquidity tiers, exit readiness. "
                    "Monitor all positions."
                ),
                expected_output="Inventory and exposure report.",
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
                    "Signal mismatch between exposure and regime. "
                    "Alert on misalignments."
                ),
                expected_output="Exposure/regime mismatch alert.",
                agent=agent,
            ),
            # TASK-A19-04
            create_task(
                description=(
                    "Don't hedge too early: let limited overexposure ride when market is favorable. "
                    "Hedge only when risk-asymmetry increases, for better risk/reward."
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
                    "Generate alerts + escalations on threshold breaches. "
                    "Immediate notification."
                ),
                expected_output="Threshold breach alert system.",
                agent=agent,
            ),
            # TASK-A25-03
            create_task(
                description=(
                    "Produce daily risk pack for CIO/CRO/Head Trading. "
                    "Comprehensive risk summary."
                ),
                expected_output="Daily risk pack.",
                agent=agent,
            ),
            # TASK-A25-04
            create_task(
                description=(
                    "Report upside scenarios too: show where taking extra risk can lead to much profit "
                    "so CIO sees balance between risk and reward."
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
                    "Maintain venue scorecards (API stability, withdrawals, legal/ops signals). "
                    "Track venue health."
                ),
                expected_output="Venue scorecard update.",
                agent=agent,
            ),
            # TASK-A26-02
            create_task(
                description=(
                    "Set limits per venue + triggers for exposure reduction. "
                    "Define venue risk limits."
                ),
                expected_output="Venue limit setting.",
                agent=agent,
            ),
            # TASK-A26-03
            create_task(
                description=(
                    "Evaluate new venues before approved status. "
                    "Due diligence on new venues."
                ),
                expected_output="New venue evaluation report.",
                agent=agent,
            ),
            # TASK-A26-04
            create_task(
                description=(
                    "Explore new venues cautiously: put small capital when arb edge is large, "
                    "with strict exposure limit and intensive monitoring, to gain extra profit without big risks."
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
                    "Validate strategy robustness."
                ),
                expected_output="Strategy validation report.",
                agent=agent,
            ),
            # TASK-A27-02
            create_task(
                description=(
                    "Define kill criteria + monitoring metrics before live. "
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
                    "Think along with quants: find ways to bring risky but promising strategies live "
                    "in controlled manner (hedges, lower allocation) instead of rejecting them outright."
                ),
                expected_output="Controlled strategy launch plan.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_operations_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Spot operations agents (28-32)."""
    tasks = []

    if "controller" in agents:
        agent = agents["controller"]
        tasks.extend([
            # TASK-A28-01
            create_task(
                description=(
                    "Daily NAV/PnL; fees; slippage accounting; resolve breaks. "
                    "Accurate financial tracking."
                ),
                expected_output="Daily NAV/PnL report.",
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
                    "Report deviations directly to COO/CRO. "
                    "Immediate escalation."
                ),
                expected_output="Deviation escalation report.",
                agent=agent,
            ),
            # TASK-A28-04
            create_task(
                description=(
                    "Analyze which strategies/pods generate real alpha vs just add volatility. "
                    "Advise CIO for reallocation toward high-performers."
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
                    "Manage buffers per venue; optimize idle vs safety. "
                    "Balance liquidity and returns."
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
                    "Prepare for contingencies."
                ),
                expected_output="Stress scenario plan.",
                agent=agent,
            ),
            # TASK-A29-04
            create_task(
                description=(
                    "Put idle funds to work: lend/stake temporary surpluses (with CRO approval) "
                    "to earn extra yield while not needed for trading."
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
                    "Enable fast fund transfers without compromising security."
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
                    "Surveillance procedures + escalation on suspicious behavior. "
                    "Monitoring and enforcement."
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
                    "Track regulations and market developments closely. Update restricted lists/policies "
                    "immediately on changes so aggressive trades don't lead to violations."
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
                    "Coordinate with controller/treasury on breaks. "
                    "Cross-team coordination."
                ),
                expected_output="Break coordination report.",
                agent=agent,
            ),
            # TASK-A32-03
            create_task(
                description=(
                    "Operational readiness for new coins/venues. "
                    "Onboarding preparation."
                ),
                expected_output="Operational readiness assessment.",
                agent=agent,
            ),
            # TASK-A32-04
            create_task(
                description=(
                    "Prepare ops for new opportunities: before listings or new exchange launches "
                    "already have accounts/funding/procedures ready so traders can act immediately."
                ),
                expected_output="Opportunity readiness report.",
                agent=agent,
            ),
        ])

    return tasks


def get_spot_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get all Spot desk tasks given a dictionary of agents.

    Args:
        agents: Dictionary mapping agent keys to Agent instances.

    Returns:
        List of all Spot desk tasks.
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
