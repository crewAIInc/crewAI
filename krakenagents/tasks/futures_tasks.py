"""Futures desk agent tasks for QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_futures_leadership_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures leadership agents (33-36)."""
    tasks = []

    if "cio" in agents:
        cio = agents["cio"]
        tasks.extend([
            # TASK-A33-01
            create_task(
                description=(
                    "Allocation over futures strategies (funding/carry, intraday, systematic, curve). "
                    "Consider leverage and funding costs."
                ),
                expected_output="Futures universe and allocation document.",
                agent=cio,
            ),
            # TASK-A33-02
            create_task(
                description=(
                    "Determine leverage per regime + max net/gross exposure. "
                    "Set leverage limits per pod."
                ),
                expected_output="Risk budget allocation with leverage limits.",
                agent=cio,
            ),
            # TASK-A33-03
            create_task(
                description=(
                    "Kill/scale decisions based on risk-adjusted performance. "
                    "Factor in funding regime and basis conditions."
                ),
                expected_output="Monthly futures allocation decision report.",
                agent=cio,
            ),
            # TASK-A33-04
            create_task(
                description=(
                    "Allow leverage above normal maximum for exceptional edge (only with CRO approval) "
                    "to capture mega-opportunities while margin buffers remain guaranteed."
                ),
                expected_output="Exceptional leverage approval report.",
                agent=cio,
            ),
        ])

    if "head_trading" in agents:
        head = agents["head_trading"]
        tasks.extend([
            # TASK-A34-01
            create_task(
                description=(
                    "Daily plan: focus instruments, event windows, risk mode. "
                    "Coordinate with spot desk on cross-desk opportunities."
                ),
                expected_output="Daily futures briefing document.",
                agent=head,
            ),
            # TASK-A34-02
            create_task(
                description=(
                    "Monitor intraday risk controls: max loss, max leverage. "
                    "Special focus on leverage usage."
                ),
                expected_output="Trade discipline report with leverage analysis.",
                agent=head,
            ),
            # TASK-A34-03
            create_task(
                description=(
                    "Review culture + error reduction. "
                    "Conduct post-trade reviews and enforce journaling."
                ),
                expected_output="Review culture report.",
                agent=head,
            ),
            # TASK-A34-04
            create_task(
                description=(
                    "Stimulate larger position on high-conviction trades (within margin limits) and learn from it. "
                    "Ensure margin stress test is done before such push."
                ),
                expected_output="High-conviction trade report.",
                agent=head,
            ),
        ])

    if "cro" in agents:
        cro = agents["cro"]
        tasks.extend([
            # TASK-A35-01
            create_task(
                description=(
                    "Design risk framework: leverage caps, liquidation buffers, drawdown limits "
                    "(and optional gamma-like exposure). Document all limits."
                ),
                expected_output="Futures risk framework with margin controls.",
                agent=cro,
            ),
            # TASK-A35-02
            create_task(
                description=(
                    "Real-time monitoring: margin, liquidation distance, venue risk. "
                    "Enforce position reductions at thresholds."
                ),
                expected_output="Margin and liquidation risk report.",
                agent=cro,
            ),
            # TASK-A35-03
            create_task(
                description=(
                    "Sign off new derivatives strategies. "
                    "Conduct pre-mortem and failure mode analysis."
                ),
                expected_output="Strategy sign-off with risk analysis.",
                agent=cro,
            ),
            # TASK-A35-04
            create_task(
                description=(
                    "Increase risk limits when performance allows: with consistent profits under current limit, "
                    "consider higher limits (Board approval) to increase potential without liquidation danger."
                ),
                expected_output="Risk limit increase proposal.",
                agent=cro,
            ),
        ])

    if "coo" in agents:
        coo = agents["coo"]
        tasks.extend([
            # TASK-A36-01
            create_task(
                description=(
                    "Daily reconciliation of positions, funding, fees. "
                    "Ensure 24/7 coverage for margin events."
                ),
                expected_output="Futures operational procedures.",
                agent=coo,
            ),
            # TASK-A36-02
            create_task(
                description=(
                    "Incident management for venue outages/margin changes. "
                    "Coordinate rapid response."
                ),
                expected_output="Incident management report.",
                agent=coo,
            ),
            # TASK-A36-03
            create_task(
                description=(
                    "Process control + audit trail. "
                    "Ensure all actions are properly logged."
                ),
                expected_output="Process control report.",
                agent=coo,
            ),
            # TASK-A36-04
            create_task(
                description=(
                    "Ensure operations remain real-time during extreme moves (liquidation cascades): "
                    "fast funding adjustments, margin top-ups from treasury, etc., executed flawlessly."
                ),
                expected_output="Extreme move ops readiness report.",
                agent=coo,
            ),
        ])

    return tasks


def get_futures_systematic_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures systematic agents (37, 42)."""
    tasks = []

    if "systematic_head" in agents:
        agent = agents["systematic_head"]
        tasks.extend([
            # TASK-A37-01
            create_task(
                description=(
                    "Design signals (trend/mean reversion/carry filters) for perps/futures. "
                    "Include funding cost assumptions."
                ),
                expected_output="Futures signal library documentation.",
                agent=agent,
            ),
            # TASK-A37-02
            create_task(
                description=(
                    "Specify strategies for dev team; monitor live drift. "
                    "Ensure clear documentation."
                ),
                expected_output="Strategy specification document.",
                agent=agent,
            ),
            # TASK-A37-03
            create_task(
                description=(
                    "Monthly model reviews + performance attribution. "
                    "Identify strategies losing edge."
                ),
                expected_output="Model review report.",
                agent=agent,
            ),
            # TASK-A37-04
            create_task(
                description=(
                    "Research advanced quant (AI, deep learning on orderbook data) for new high-alpha strategies. "
                    "Strict testing and phased rollout if successful."
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
                    "Execute signals; monitor exceptions; pause on anomalies. "
                    "Track funding payments and their impact."
                ),
                expected_output="Daily futures systematic operations log.",
                agent=agent,
            ),
            # TASK-A42-02
            create_task(
                description=(
                    "Log deviations; coordinate with Agent 37/35. "
                    "Document all operational issues."
                ),
                expected_output="Deviation log with escalations.",
                agent=agent,
            ),
            # TASK-A42-03
            create_task(
                description=(
                    "Prevent human override outside protocol. "
                    "Maintain systematic discipline."
                ),
                expected_output="Protocol compliance report.",
                agent=agent,
            ),
            # TASK-A42-04
            create_task(
                description=(
                    "Optimize capital utilization: report to Agent 33 when a strategy is idle or underused "
                    "so capital can be temporarily deployed elsewhere for more return."
                ),
                expected_output="Capital utilization optimization report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_carry_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures carry agents (38, 43, 44)."""
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
                    "Document exit procedures."
                ),
                expected_output="Capacity and unwind playbook.",
                agent=agent,
            ),
            # TASK-A38-03
            create_task(
                description=(
                    "Coordinate with treasury for collateral efficiency. "
                    "Optimize margin usage."
                ),
                expected_output="Collateral efficiency report.",
                agent=agent,
            ),
            # TASK-A38-04
            create_task(
                description=(
                    "Exploit extreme funding dislocations maximally: at very high positive funding rate "
                    "(shorts get paid) go aggressively short (within margin limits) and hedge via spot/options if needed."
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
                    "Execute funding strategies within leverage/margin policy. "
                    "Document all positions."
                ),
                expected_output="Funding strategy execution log.",
                agent=agent,
            ),
            # TASK-A43-02
            create_task(
                description=(
                    "Monitor funding regime shifts + unwind triggers. "
                    "Alert on significant changes."
                ),
                expected_output="Funding regime monitoring report.",
                agent=agent,
            ),
            # TASK-A43-03
            create_task(
                description=(
                    "Report capacity + cost breakdown. "
                    "Track funding income and costs."
                ),
                expected_output="Capacity and cost breakdown.",
                agent=agent,
            ),
            # TASK-A43-04
            create_task(
                description=(
                    "Scale to max size on extreme funding spreads (within risk appetite) to maximize carry. "
                    "Ensure extra margin buffers to prevent liquidations."
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
                    "Capture spreads between venues with strict venue limits. "
                    "Execute cross-venue carry."
                ),
                expected_output="Cross-venue carry execution log.",
                agent=agent,
            ),
            # TASK-A44-02
            create_task(
                description=(
                    "Unwind procedures on venue health deterioration. "
                    "Rapid exit when needed."
                ),
                expected_output="Venue health unwind procedure.",
                agent=agent,
            ),
            # TASK-A44-03
            create_task(
                description=(
                    "Prevent stranded legs via predefined triggers. "
                    "Avoid asymmetric positions."
                ),
                expected_output="Stranded leg prevention report.",
                agent=agent,
            ),
            # TASK-A44-04
            create_task(
                description=(
                    "Explore arbitrage on new futures venues (under strict limit) to benefit from inefficiencies "
                    "before others do. Monitor risk continuously."
                ),
                expected_output="New venue arbitrage exploration report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_microstructure_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures microstructure agents (39, 45-47)."""
    tasks = []

    if "microstructure_head" in agents:
        agent = agents["microstructure_head"]
        tasks.extend([
            # TASK-A39-01
            create_task(
                description=(
                    "Develop playbooks: breakout validation, absorption, stop runs, liquidity cliffs. "
                    "Design microstructure trading strategies."
                ),
                expected_output="Microstructure trading playbook.",
                agent=agent,
            ),
            # TASK-A39-02
            create_task(
                description=(
                    "Coach intraday traders; enforce no-trade regimes. "
                    "Maintain trading discipline."
                ),
                expected_output="Intraday coaching report.",
                agent=agent,
            ),
            # TASK-A39-03
            create_task(
                description=(
                    "Work with execution on order types + timing. "
                    "Optimize execution quality."
                ),
                expected_output="Execution optimization report.",
                agent=agent,
            ),
            # TASK-A39-04
            create_task(
                description=(
                    "Hunt extreme orderflow moments (massive stop-hunts, large buyers/sellers) and design "
                    "microstructure trades to profit from these (e.g., quick scalps during panic)."
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
                    "Journal with setup tags + screenshots/notes. "
                    "Document all trades."
                ),
                expected_output="Trade journal with evidence.",
                agent=agent,
            ),
            # TASK-A45-03
            create_task(
                description=(
                    "Coordinate timing with execution during stress. "
                    "Work with execution team."
                ),
                expected_output="Stress coordination log.",
                agent=agent,
            ),
            # TASK-A45-04
            create_task(
                description=(
                    "Use 'house money': with early profit on the day, trade slightly more aggressively "
                    "with part of that profit later (maintain discipline) for higher PnL."
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
                    "Focus on ETH-specific opportunities."
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
                    "Play unique ETH drivers: on news where ETH moves independently of BTC, increase ETH position "
                    "for extra alpha (optionally partially hedged via BTC to temper directional risk)."
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
                    "Play playbooks for headline spikes, liquidation cascades, gap moves. "
                    "React to orderflow events."
                ),
                expected_output="Orderflow event trading log.",
                agent=agent,
            ),
            # TASK-A47-02
            create_task(
                description=(
                    "Work closely with news analyst + CRO during volatility. "
                    "Coordinate on fast-moving situations."
                ),
                expected_output="Volatility coordination report.",
                agent=agent,
            ),
            # TASK-A47-03
            create_task(
                description=(
                    "Quick exits on regime flips; prevent gambling mode. "
                    "Maintain discipline."
                ),
                expected_output="Regime flip response log.",
                agent=agent,
            ),
            # TASK-A47-04
            create_task(
                description=(
                    "Prepare pre-set orders for macro releases (e.g., stop buys/sells just outside current spread) "
                    "to react immediately on surprise. Minimize latency."
                ),
                expected_output="Macro event pre-positioning report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_research_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures research agents (40, 52-56)."""
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
                expected_output="Futures research report with funding forecasts.",
                agent=agent,
            ),
            # TASK-A40-02
            create_task(
                description=(
                    "Analyze event impact on perps: funding/OI behavior. "
                    "Track market structure changes."
                ),
                expected_output="Event impact analysis.",
                agent=agent,
            ),
            # TASK-A40-03
            create_task(
                description=(
                    "Alerts + weekly docket to CIO/Head Trading. "
                    "Keep leadership informed."
                ),
                expected_output="Weekly research docket.",
                agent=agent,
            ),
            # TASK-A40-04
            create_task(
                description=(
                    "Use options market insights (OI, put/call ratios) as supplement: if options show extreme bias "
                    "vs futures metrics, signal arbitrage or contrarian trade opportunities."
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
                    "Build funding/OI dashboards + liquidation clusters + anomaly alerts. "
                    "Comprehensive derivatives data."
                ),
                expected_output="Derivatives data dashboard.",
                agent=agent,
            ),
            # TASK-A52-02
            create_task(
                description=(
                    "Data QA: venue differences + outliers. "
                    "Ensure data quality."
                ),
                expected_output="Data quality report.",
                agent=agent,
            ),
            # TASK-A52-03
            create_task(
                description=(
                    "Support desk situational awareness. "
                    "Keep traders informed."
                ),
                expected_output="Situational awareness report.",
                agent=agent,
            ),
            # TASK-A52-04
            create_task(
                description=(
                    "Integrate options data (OI, put/call ratios) in analyses for fuller picture of market expectations, "
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
                    "Give risk mode advice (normal/reduced/defensive) based on macro calendar. "
                    "Set leverage guidance."
                ),
                expected_output="Risk mode recommendation.",
                agent=agent,
            ),
            # TASK-A53-02
            create_task(
                description=(
                    "Scenario planning around macro events. "
                    "Prepare for different outcomes."
                ),
                expected_output="Macro scenario plan.",
                agent=agent,
            ),
            # TASK-A53-03
            create_task(
                description=(
                    "Communicate leverage discipline per regime. "
                    "Clear guidance on position sizing."
                ),
                expected_output="Leverage discipline communication.",
                agent=agent,
            ),
            # TASK-A53-04
            create_task(
                description=(
                    "Warn timely when high leverage is (ir)responsible: signal periods to be more careful "
                    "(e.g., around Fed decisions) vs moments with macro tailwind to increase risk."
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
                    "Alerts: 'what does this mean for perps, funding, OI, liquidations?' "
                    "Translate news to derivatives impact."
                ),
                expected_output="News to derivatives impact alert.",
                agent=agent,
            ),
            # TASK-A54-02
            create_task(
                description=(
                    "Coordinate with Head Trading + CRO on headline risk. "
                    "Rapid response coordination."
                ),
                expected_output="Headline risk coordination report.",
                agent=agent,
            ),
            # TASK-A54-03
            create_task(
                description=(
                    "Support safer reaction trading. "
                    "Help traders navigate news-driven moves."
                ),
                expected_output="Reaction trading support.",
                agent=agent,
            ),
            # TASK-A54-04
            create_task(
                description=(
                    "Monitor traditional markets (S&P, USD, yields) too and translate large moves there to crypto context. "
                    "Warn team when cross-asset signals point to 'risk-off' or 'risk-on'."
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
                    "Context: when is flow a leading indicator vs noise? "
                    "Filter signal from noise."
                ),
                expected_output="Flow signal analysis.",
                agent=agent,
            ),
            # TASK-A55-03
            create_task(
                description=(
                    "Timing input for risk reduction/entries. "
                    "Provide actionable timing signals."
                ),
                expected_output="Flow timing advisory.",
                agent=agent,
            ),
            # TASK-A55-04
            create_task(
                description=(
                    "Use stablecoin on-chain data (large mints/burns) to predict liquidity changes. "
                    "Warn if e.g. much stablecoins go to exchanges (can affect funding rates)."
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
                    "Validate strategy concepts."
                ),
                expected_output="Strategy research memo.",
                agent=agent,
            ),
            # TASK-A56-02
            create_task(
                description=(
                    "Post-launch evaluation: edge maintained or declining? "
                    "Monitor strategy health."
                ),
                expected_output="Post-launch strategy evaluation.",
                agent=agent,
            ),
            # TASK-A56-03
            create_task(
                description=(
                    "Continuous strategy improvement with evidence. "
                    "Data-driven optimization."
                ),
                expected_output="Strategy improvement report.",
                agent=agent,
            ),
            # TASK-A56-04
            create_task(
                description=(
                    "Follow external quant developments (papers, forums) for fresh ideas. "
                    "Test promising concepts on our data to discover new edge."
                ),
                expected_output="External quant research report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_execution_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures execution agents (41, 51)."""
    tasks = []

    if "execution_head" in agents:
        agent = agents["execution_head"]
        tasks.extend([
            # TASK-A41-01
            create_task(
                description=(
                    "KPIs: slippage, reject rate, adverse selection, fill quality. "
                    "Track and improve execution."
                ),
                expected_output="Futures execution KPI report.",
                agent=agent,
            ),
            # TASK-A41-02
            create_task(
                description=(
                    "Routing + ordertype policy per strategy. "
                    "Document execution procedures."
                ),
                expected_output="Execution policy document.",
                agent=agent,
            ),
            # TASK-A41-03
            create_task(
                description=(
                    "Stress execution playbooks: rapid unwind without panic. "
                    "Prepare for emergency situations."
                ),
                expected_output="Stress execution playbook.",
                agent=agent,
            ),
            # TASK-A41-04
            create_task(
                description=(
                    "Try new execution channels: use RFQ/dark pool for large orders, apply maker-only vs taker "
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
                    "Execute desk-wide risk reduction on kill-switch/drawdown. "
                    "Emergency position reduction."
                ),
                expected_output="Unwind execution plan and report.",
                agent=agent,
            ),
            # TASK-A51-02
            create_task(
                description=(
                    "Manage hedge overlays: reduce net exposure, reduce liquidation risk. "
                    "Protective hedging."
                ),
                expected_output="Hedge overlay management report.",
                agent=agent,
            ),
            # TASK-A51-03
            create_task(
                description=(
                    "Coordinate with treasury/margin specialist. "
                    "Cross-team coordination."
                ),
                expected_output="Treasury coordination log.",
                agent=agent,
            ),
            # TASK-A51-04
            create_task(
                description=(
                    "Practice periodic crisis-unwind drills: simulate complete portfolio unwind to test processes "
                    "and prepare traders for real emergency situations."
                ),
                expected_output="Crisis unwind drill report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_swing_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures swing agents (48-50)."""
    tasks = []

    if "swing_head" in agents:
        agent = agents["swing_head"]
        tasks.extend([
            # TASK-A48-01
            create_task(
                description=(
                    "Regime-aware sizing + clear invalidation. "
                    "Adjust position size to market conditions."
                ),
                expected_output="Directional trade thesis with leverage plan.",
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
                    "Work with macro/regime input. "
                    "Align trades with macro view."
                ),
                expected_output="Macro-aligned trade plan.",
                agent=agent,
            ),
            # TASK-A48-04
            create_task(
                description=(
                    "Pyramid into strong trends: add to winning position on pullbacks with built-up profit "
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
                    "Maintain stops/targets discipline. "
                    "No moving stops outside plan."
                ),
                expected_output="Discipline compliance report.",
                agent=agent,
            ),
            # TASK-A49-04
            create_task(
                description=(
                    "At clear range extremes take slightly larger position than normal (still small enough). "
                    "Split position to take partial profit and let rest mean revert."
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
                    "Identify curve anomalies; execute spreads with low slippage. "
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
                    "Define capacity + exit plans. "
                    "Document position limits and exits."
                ),
                expected_output="Capacity and exit planning.",
                agent=agent,
            ),
            # TASK-A50-04
            create_task(
                description=(
                    "Exploit new contracts: if new futures (extra expiries) launch with dislocations, "
                    "test those spreads small-scale and scale up if liquidity proves adequate."
                ),
                expected_output="New contract opportunity report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_risk_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures risk agents (57-59)."""
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
                expected_output="Margin and liquidation monitoring report.",
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
                    "Support risk-adjusted allocation input. "
                    "Provide data for allocation decisions."
                ),
                expected_output="Risk-adjusted allocation support.",
                agent=agent,
            ),
            # TASK-A57-04
            create_task(
                description=(
                    "Proactively signal room for more risk: if exposures are low and market conditions favorable, "
                    "suggest increasing leverage/positions for more profit potential (with CRO approval)."
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
                    "Define collateral buffers + liquidation distance targets. "
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
                    "Simulate flash crashes & venue outages: test if collaterals are sufficient and procedures work. "
                    "Adjust collateral buffers or war-room procedures based on simulation findings."
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
                    "Post-launch review on drift + regime dependency. "
                    "Ongoing strategy monitoring."
                ),
                expected_output="Post-launch review report.",
                agent=agent,
            ),
            # TASK-A59-03
            create_task(
                description=(
                    "Maintain audit trail of approvals. "
                    "Document all decisions."
                ),
                expected_output="Approval audit trail.",
                agent=agent,
            ),
            # TASK-A59-04
            create_task(
                description=(
                    "Keep eye on potential: don't blindly block high-risk strategy, but require mitigations "
                    "(e.g., smaller size, extra hedges) so promising ideas can be tested without uncontrolled risk."
                ),
                expected_output="Controlled risk approval framework.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_operations_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get tasks for Futures operations agents (60-64)."""
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
                expected_output="Daily position reconciliation.",
                agent=agent,
            ),
            # TASK-A60-02
            create_task(
                description=(
                    "Signal breaks directly to COO/CRO. "
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
                expected_output="Funding and venue issue log.",
                agent=agent,
            ),
            # TASK-A60-04
            create_task(
                description=(
                    "Dissect PnL daily: which part came from price movement vs funding vs fees. "
                    "Signal notable costs (like unexpectedly high fees) so team can act (e.g., tier upgrade)."
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
                    "Manage buffers per venue; optimize idle vs safety. "
                    "Balance liquidity and returns."
                ),
                expected_output="Venue buffer management.",
                agent=agent,
            ),
            # TASK-A61-02
            create_task(
                description=(
                    "Coordinate collateral moves via approvals. "
                    "Secure fund transfers."
                ),
                expected_output="Collateral movement log.",
                agent=agent,
            ),
            # TASK-A61-03
            create_task(
                description=(
                    "Prepare for weekend risk + funding spikes. "
                    "Weekend contingency planning."
                ),
                expected_output="Weekend risk preparation.",
                agent=agent,
            ),
            # TASK-A61-04
            create_task(
                description=(
                    "Give idle collateral a useful role: if possible put surplus margin to work "
                    "(e.g., via low-risk earn/lending) to earn something on unused funds (CRO approval needed)."
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
                    "Prevent fat-finger via permissions + controls. "
                    "Error prevention."
                ),
                expected_output="Fat-finger prevention report.",
                agent=agent,
            ),
            # TASK-A62-04
            create_task(
                description=(
                    "Adapt API key management for high-frequency: use segmented keys per strategy with limited rights "
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
                    "Monitor compliance with venue rules + internal leverage policy. "
                    "Rule enforcement."
                ),
                expected_output="Rule compliance report.",
                agent=agent,
            ),
            # TASK-A63-03
            create_task(
                description=(
                    "Ensure documentation/recordkeeping. "
                    "Audit trail maintenance."
                ),
                expected_output="Documentation compliance.",
                agent=agent,
            ),
            # TASK-A63-04
            create_task(
                description=(
                    "Watch aggressive trades vs market rules: prevent intensive strategies from crossing "
                    "market abuse lines (spoofing, squeezes) and intervene immediately on doubt."
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
                    "Funding/fees processing; resolve breaks; coordinate with controller/treasury. "
                    "Operational workflow."
                ),
                expected_output="Operational processing log.",
                agent=agent,
            ),
            # TASK-A64-02
            create_task(
                description=(
                    "Support during volatility spikes + venue incidents. "
                    "Crisis support."
                ),
                expected_output="Volatility support report.",
                agent=agent,
            ),
            # TASK-A64-03
            create_task(
                description=(
                    "Ensure trading can continue during ops friction. "
                    "Business continuity."
                ),
                expected_output="Business continuity report.",
                agent=agent,
            ),
            # TASK-A64-04
            create_task(
                description=(
                    "Automate monitoring: detect every deviation (missing funding posting, trade break) immediately "
                    "so you resolve it before it affects PnL or trader focus."
                ),
                expected_output="Automated monitoring report.",
                agent=agent,
            ),
        ])

    return tasks


def get_futures_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Get all Futures desk tasks given a dictionary of agents.

    Args:
        agents: Dictionary mapping agent keys to Agent instances.

    Returns:
        List of all Futures desk tasks.
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
