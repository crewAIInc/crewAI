"""STAFF agent tasks for QRI Trading Organization."""

from crewai import Agent, Task

from krakenagents.tasks.base import create_task


def get_ceo_tasks(ceo: Agent) -> list[Task]:
    """Get tasks for STAFF-00 CEO."""
    return [
        # TASK-CEO-00 — ACTIVATE TRADING DESKS (PRIORITY 1)
        create_task(
            description=(
                "CRITICAL FIRST ACTION: Activate the trading desks to start trading operations.\n\n"
                "You MUST execute these steps in order:\n\n"
                "STEP 1: Check desk status\n"
                "- Call the tool: get_desk_status\n"
                "- Input: 'all'\n"
                "- This shows if Spot and Futures desks are running\n\n"
                "STEP 2: Start Spot Desk trading\n"
                "- Call the tool: delegate_to_spot_desk\n"
                "- directive: 'Analyze BTC and ETH spot markets. Check current prices, order books, "
                "and recent trades. Identify any arbitrage or trading opportunities. Execute small "
                "test trades if conditions are favorable.'\n"
                "- priority: 'high'\n\n"
                "STEP 3: Start Futures Desk trading\n"
                "- Call the tool: delegate_to_futures_desk\n"
                "- directive: 'Analyze BTC and ETH perpetual futures. Check funding rates, open interest, "
                "and basis vs spot. Look for carry trade opportunities or directional trades. "
                "Execute positions if edge is identified.'\n"
                "- priority: 'high'\n\n"
                "DO NOT proceed to other tasks until both desks are activated."
            ),
            expected_output=(
                "Confirmation showing:\n"
                "1. Desk status check completed\n"
                "2. Spot Desk activated with trading directive\n"
                "3. Futures Desk activated with trading directive\n"
                "Both desks must be running before this task is complete."
            ),
            agent=ceo,
        ),
        # TASK-CEO-01 — Monitor trading operations
        create_task(
            description=(
                "Monitor active trading operations across both desks.\n"
                "1. Use get_desk_status with input 'all' to check both desks\n"
                "2. Review any alerts from the risk dashboard\n"
                "3. Ensure desks are executing their directives\n"
                "4. If a desk has stopped, restart it with delegate_to_spot_desk or delegate_to_futures_desk"
            ),
            expected_output="Trading operations status report with any interventions made.",
            agent=ceo,
        ),
    ]


def get_ceo_interview_tasks(ceo: Agent) -> list[Task]:
    """Get interview case tasks for STAFF-00 CEO."""
    return [
        # TASK-CEO-INT-01 — Weekend crash + venue problems
        create_task(
            description=(
                "Interview Case 1: Weekend crash + venue problems. "
                "Scenario: Major market crash during weekend with venue withdrawals frozen. "
                "Evaluate response: risk reduction, communication, contingency activation."
            ),
            expected_output="Case response document demonstrating crisis management approach.",
            agent=ceo,
        ),
        # TASK-CEO-INT-02 — Best strategy works, how to scale?
        create_task(
            description=(
                "Interview Case 2: Best strategy works, how to scale? "
                "Scenario: A strategy is delivering 3x expected returns. "
                "Evaluate: How to scale without degrading edge or increasing risk disproportionately."
            ),
            expected_output="Scaling decision framework with risk considerations.",
            agent=ceo,
        ),
        # TASK-CEO-INT-03 — Trader breaks rules but makes profit
        create_task(
            description=(
                "Interview Case 3: Trader breaks risk rules but makes profit. "
                "Scenario: Trader exceeded position limits but generated significant PnL. "
                "Evaluate: How to handle rule violations that resulted in profits."
            ),
            expected_output="Disciplinary framework that maintains risk culture while being fair.",
            agent=ceo,
        ),
    ]


def get_ceo_compensation_tasks(ceo: Agent) -> list[Task]:
    """Get compensation structure tasks for STAFF-00 CEO."""
    return [
        # TASK-CEO-COMP-01 — Risk-adjusted performance bonus
        create_task(
            description=(
                "Set bonus on risk-adjusted performance (not only PnL). "
                "Design compensation structure that rewards Sharpe ratio, not just returns. "
                "Include drawdown penalties in bonus calculation."
            ),
            expected_output="Risk-adjusted bonus calculation methodology.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-02 — Hard thresholds
        create_task(
            description=(
                "Implement hard thresholds for compensation. "
                "Define minimum performance hurdles before any bonus is paid. "
                "Set maximum exposure limits that trigger bonus clawback if breached."
            ),
            expected_output="Compensation threshold document with clear triggers.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-03 — Deferred bonus (skin in the game)
        create_task(
            description=(
                "Implement deferred bonus structure for skin in the game. "
                "Vest portion of bonus over 2-3 years. "
                "Allow clawback if subsequent losses occur."
            ),
            expected_output="Deferred compensation structure with vesting schedule.",
            agent=ceo,
        ),
        # TASK-CEO-COMP-04 — Bonus kicker for outsized performance
        create_task(
            description=(
                "Bonus kicker for outsized performance (extreme wins extra rewarded). "
                "Design accelerator for exceptional returns above threshold. "
                "Include deferral/clawback as protection against subsequent losses."
            ),
            expected_output="Performance kicker structure with protective measures.",
            agent=ceo,
        ),
    ]


def get_ceo_90day_tasks(ceo: Agent) -> list[Task]:
    """Get first 90 days deliverable tasks for STAFF-00 CEO."""
    return [
        # TASK-CEO-90D-01 (Day 1-30) — Complete board team
        create_task(
            description=(
                "Day 1-30: Complete board team (CIO/CRO/COO/CFO/Compliance/Security/etc.). "
                "Hire or confirm all STAFF positions. "
                "Ensure clear reporting lines and responsibilities."
            ),
            expected_output="Completed org chart with all STAFF positions filled.",
            agent=ceo,
        ),
        # TASK-CEO-90D-02 (Day 1-30) — Risk charter + kill-switch live
        create_task(
            description=(
                "Day 1-30: Finalize risk charter and kill-switch ladder. "
                "Document all risk limits, thresholds, and escalation procedures. "
                "Test kill-switch activation with dry runs."
            ),
            expected_output="Live risk charter document with tested kill-switch procedures.",
            agent=ceo,
        ),
        # TASK-CEO-90D-03 (Day 1-30) — Desk separation complete
        create_task(
            description=(
                "Day 1-30: Complete desk separation technically and operationally. "
                "Separate accounts, approvals, and reporting between Spot and Futures. "
                "Verify no cross-desk dependencies exist."
            ),
            expected_output="Desk separation audit report confirming complete segregation.",
            agent=ceo,
        ),
        # TASK-CEO-90D-04 (Day 31-60) — Strategy pipeline
        create_task(
            description=(
                "Day 31-60: Establish strategy pipeline: research → pilot → go/no-go. "
                "Define gates and criteria for each stage. "
                "Create standard templates for strategy proposals."
            ),
            expected_output="Strategy pipeline documentation with templates.",
            agent=ceo,
        ),
        # TASK-CEO-90D-05 (Day 31-60) — Execution KPIs + cost dashboards
        create_task(
            description=(
                "Day 31-60: Launch execution KPIs and cost dashboards. "
                "Track implementation shortfall, fees, slippage. "
                "Make dashboards accessible to all relevant personnel."
            ),
            expected_output="Live execution KPI dashboard with cost tracking.",
            agent=ceo,
        ),
        # TASK-CEO-90D-06 (Day 31-60) — First pilots with small risk
        create_task(
            description=(
                "Day 31-60: Launch first strategy pilots with small risk budgets. "
                "Start with most promising strategies from pipeline. "
                "Monitor closely and gather performance data."
            ),
            expected_output="Pilot launch report with initial performance data.",
            agent=ceo,
        ),
        # TASK-CEO-90D-07 (Day 61-90) — Scale what works, kill what doesn't
        create_task(
            description=(
                "Day 61-90: Scale up strategies that work; kill those that don't. "
                "Use pilot data to make informed decisions. "
                "Reallocate capital from failures to successes."
            ),
            expected_output="Kill/scale decision report with capital reallocation.",
            agent=ceo,
        ),
        # TASK-CEO-90D-08 (Day 61-90) — 24/7 escalation + incident drills
        create_task(
            description=(
                "Day 61-90: Establish 24/7 escalation coverage and conduct incident drills. "
                "Test response times and decision-making under pressure. "
                "Document lessons learned and improve procedures."
            ),
            expected_output="Incident drill report with escalation coverage confirmation.",
            agent=ceo,
        ),
        # TASK-CEO-90D-09 (Day 61-90) — Monthly allocation cycle operational
        create_task(
            description=(
                "Day 61-90: Monthly allocation cycle running as machine. "
                "Standardize proposal format and timeline. "
                "Ensure data-driven decisions with clear rationale."
            ),
            expected_output="First complete allocation cycle documentation.",
            agent=ceo,
        ),
        # TASK-CEO-90D-10 (Day 61-90) — High-performance culture
        create_task(
            description=(
                "Day 61-90: Establish high-performance culture: stimulate aggressive alpha initiatives "
                "(high risk, high reward) within risk limits, with discipline as foundation. "
                "Reward exceptional performance, penalize rule-breaking."
            ),
            expected_output="Culture statement with incentive alignment documentation.",
            agent=ceo,
        ),
    ]


def get_group_cio_tasks(group_cio: Agent) -> list[Task]:
    """Get tasks for STAFF-01 Group CIO."""
    return [
        # TASK-GCIO-01 — Collect desk proposals + create allocation proposal
        create_task(
            description=(
                "Collect desk allocation proposals and create Group allocation recommendation. "
                "Consider risk budgets, capacity, and performance metrics."
            ),
            expected_output="Consolidated allocation proposal with rationale for each desk.",
            agent=group_cio,
        ),
        # TASK-GCIO-02 — Monthly kill/scale review
        create_task(
            description=(
                "Run monthly kill/scale review based on data. Identify underperforming strategies "
                "to kill and successful strategies to scale."
            ),
            expected_output="Kill/scale recommendation report with supporting data analysis.",
            agent=group_cio,
        ),
        # TASK-GCIO-03 — Publish allocation decision
        create_task(
            description=(
                "Publish allocation decision with rationale and KPIs per desk/pod. "
                "Communicate clearly to all stakeholders. "
                "Set expectations for next review cycle."
            ),
            expected_output="Published allocation decision with KPIs.",
            agent=group_cio,
        ),
        # TASK-GCIO-04 — Scout high-alpha opportunities
        create_task(
            description=(
                "Scout high-alpha opportunities in new markets or strategies. "
                "Reserve pilot capital with CRO approval for testing potential. "
                "Propose new market entries or strategy innovations."
            ),
            expected_output="High-alpha opportunity report with pilot recommendations.",
            agent=group_cio,
        ),
    ]


def get_group_cro_tasks(group_cro: Agent) -> list[Task]:
    """Get tasks for STAFF-02 Group CRO."""
    return [
        # TASK-GCRO-01 — Define risk charter + hard rails
        create_task(
            description=(
                "Define and maintain Group risk charter with hard rails. "
                "Document limits, thresholds, and escalation procedures."
            ),
            expected_output="Updated risk charter document with all limits and procedures.",
            agent=group_cro,
        ),
        # TASK-GCRO-02 — Enforce veto process
        create_task(
            description=(
                "Enforce veto process on breaches. Review all breach reports "
                "and exercise veto authority where required."
            ),
            expected_output="Breach review report with veto decisions and rationale.",
            agent=group_cro,
        ),
        # TASK-GCRO-03 — Kill-switch ladder testing
        create_task(
            description=(
                "Test kill-switch ladder with drills and maintain audit trail. "
                "Ensure all kill-switch procedures are functional and documented."
            ),
            expected_output="Kill-switch drill report with test results and audit trail.",
            agent=group_cro,
        ),
        # TASK-GCRO-04 — Periodically review risk limits
        create_task(
            description=(
                "Periodically review risk limits: ensure they are high enough for aggressive trading "
                "within safe margins. Increase limits for proven success (with Board approval). "
                "Balance risk appetite with protection."
            ),
            expected_output="Risk limit review report with adjustment recommendations.",
            agent=group_cro,
        ),
    ]


def get_group_coo_tasks(group_coo: Agent) -> list[Task]:
    """Get tasks for STAFF-03 Group COO."""
    return [
        # TASK-GCOO-01 — Enforce Spot/Derivatives separation
        create_task(
            description=(
                "Enforce operational separation of Spot/Derivatives. "
                "Verify accounts, approvals, and reporting are properly segregated."
            ),
            expected_output="Separation compliance report with any gaps identified.",
            agent=group_coo,
        ),
        # TASK-GCOO-02 — Daily ops control cycle
        create_task(
            description=(
                "Run daily ops control cycle. Ensure all breaks are identified "
                "and resolved same day."
            ),
            expected_output="Daily ops control report with break resolution status.",
            agent=group_coo,
        ),
        # TASK-GCOO-03 — Incident response coordination
        create_task(
            description=(
                "Coordinate incident response for venue outages, settlement issues, "
                "and ops problems. Ensure proper communication and resolution."
            ),
            expected_output="Incident response report with actions taken and resolution.",
            agent=group_coo,
        ),
        # TASK-GCOO-04 — Scale ops for high volume
        create_task(
            description=(
                "Scale ops processes for high-volume: ensure during peak volatility "
                "reconciliations, settlements etc. run flawlessly without delay. "
                "Stress test operational capacity."
            ),
            expected_output="Ops scalability report with stress test results.",
            agent=group_coo,
        ),
    ]


def get_group_cfo_tasks(group_cfo: Agent) -> list[Task]:
    """Get tasks for STAFF-04 Group CFO."""
    return [
        # TASK-GCFO-01 — Daily NAV/PnL consolidation
        create_task(
            description=(
                "Daily NAV/PnL consolidation across spot and derivatives. "
                "Ensure accurate calculation of all positions."
            ),
            expected_output="Daily NAV/PnL report with consolidated figures.",
            agent=group_cfo,
        ),
        # TASK-GCFO-02 — Performance attribution
        create_task(
            description=(
                "Performance attribution per desk/pod/strategy. "
                "Identify top and weak performers."
            ),
            expected_output="Performance attribution report with rankings.",
            agent=group_cfo,
        ),
        # TASK-GCFO-03 — Cost dashboards + budget variances
        create_task(
            description=(
                "Cost analysis: monitor fees/funding/borrow and identify optimization opportunities. "
                "Track budget variances and flag deviations."
            ),
            expected_output="Cost analysis report with optimization recommendations.",
            agent=group_cfo,
        ),
        # TASK-GCFO-04 — Identify top vs weak strategies
        create_task(
            description=(
                "Identify top vs weak strategies: highlight which desks/pods deliver most alpha "
                "vs which underperform. Provide data for targeted allocation decisions."
            ),
            expected_output="Strategy performance ranking with allocation recommendations.",
            agent=group_cfo,
        ),
        # TASK-GCFO-05 — Optimize trading costs
        create_task(
            description=(
                "Optimize trading costs: monitor fees/funding/borrow and propose improvements "
                "(better fee tiers, less idle assets) to maximize net PnL."
            ),
            expected_output="Trading cost optimization report with specific recommendations.",
            agent=group_cfo,
        ),
    ]


def get_compliance_tasks(compliance: Agent) -> list[Task]:
    """Get tasks for STAFF-05 Compliance."""
    return [
        # TASK-COMPL-01 — Enforce compliance stops
        create_task(
            description=(
                "Enforce compliance stops: restricted list and market conduct rules. "
                "Block any prohibited trading activity."
            ),
            expected_output="Compliance enforcement report with any blocks or warnings.",
            agent=compliance,
        ),
        # TASK-COMPL-02 — Periodic training + policy checks
        create_task(
            description=(
                "Run periodic training and policy checks. "
                "Ensure all personnel are compliant with current policies."
            ),
            expected_output="Training and policy compliance report.",
            agent=compliance,
        ),
        # TASK-COMPL-03 — Incident/case escalation workflow
        create_task(
            description=(
                "Run incident/case escalation workflow. "
                "Document and track all compliance incidents. "
                "Escalate to appropriate parties based on severity."
            ),
            expected_output="Incident tracking report with escalation actions.",
            agent=compliance,
        ),
        # TASK-COMPL-04 — Advanced surveillance tools
        create_task(
            description=(
                "Deploy advanced tools (chain analytics for fund flows, AI surveillance) "
                "to detect violations early without slowing down trading. "
                "Balance oversight with operational efficiency."
            ),
            expected_output="Surveillance tool deployment report with detection metrics.",
            agent=compliance,
        ),
    ]


def get_security_tasks(security: Agent) -> list[Task]:
    """Get tasks for STAFF-06 Security."""
    return [
        # TASK-SEC-01 — Access control + key management
        create_task(
            description=(
                "Implement and maintain access control and key management policies. "
                "Ensure proper custody security."
            ),
            expected_output="Security policy implementation report.",
            agent=security,
        ),
        # TASK-SEC-02 — Withdrawal/whitelist approval flows
        create_task(
            description=(
                "Enforce withdrawal/whitelist approval flows with separation of duties. "
                "Verify all large transfers follow proper approval."
            ),
            expected_output="Approval flow audit report.",
            agent=security,
        ),
        # TASK-SEC-03 — Test incident runbooks
        create_task(
            description=(
                "Test incident runbooks: phishing, compromise, anomalies. "
                "Conduct regular drills and update procedures based on results. "
                "Ensure response times meet targets."
            ),
            expected_output="Incident drill results with procedure updates.",
            agent=security,
        ),
        # TASK-SEC-04 — Advanced custody (MPC wallets)
        create_task(
            description=(
                "Implement advanced custody (e.g., MPC wallets via Fireblocks/Copper) "
                "for secure storage AND fast access so trading opportunities aren't missed. "
                "Balance security with operational speed."
            ),
            expected_output="Advanced custody implementation report with performance metrics.",
            agent=security,
        ),
    ]


def get_prime_tasks(prime: Agent) -> list[Task]:
    """Get tasks for STAFF-07 Prime/Venues."""
    return [
        # TASK-PRIME-01 — Venue scorecards + limits
        create_task(
            description=(
                "Maintain venue scorecards and propose limits with CRO. "
                "Assess venue health, fees, and reliability."
            ),
            expected_output="Venue scorecard update with limit recommendations.",
            agent=prime,
        ),
        # TASK-PRIME-02 — Liquidity/fee optimization
        create_task(
            description=(
                "Optimize liquidity and fees per venue. "
                "Identify opportunities for better execution."
            ),
            expected_output="Liquidity/fee optimization report.",
            agent=prime,
        ),
        # TASK-PRIME-03 — Concentration risk mitigation
        create_task(
            description=(
                "Mitigate concentration risk with diversification plan. "
                "Ensure no single venue holds excessive exposure. "
                "Maintain backup venues for critical operations."
            ),
            expected_output="Diversification plan with concentration limits.",
            agent=prime,
        ),
        # TASK-PRIME-04 — Low-latency connectivity
        create_task(
            description=(
                "Invest in low-latency connectivity: co-location, dedicated lines to key venues. "
                "Minimize latency/slippage for competitive execution."
            ),
            expected_output="Connectivity improvement plan with latency metrics.",
            agent=prime,
        ),
        # TASK-PRIME-05 — Multi-venue relationships
        create_task(
            description=(
                "Maintain multi-venue relationships: prime brokers, OTC desks for deep liquidity "
                "and fast execution of large orders. "
                "Negotiate better terms where possible."
            ),
            expected_output="Venue relationship report with negotiation outcomes.",
            agent=prime,
        ),
    ]


def get_data_tasks(data: Agent) -> list[Task]:
    """Get tasks for STAFF-08 Data."""
    return [
        # TASK-DATA-01 — Core datasets + QA checks
        create_task(
            description=(
                "Set up core datasets with QA checks. "
                "Ensure data quality and completeness."
            ),
            expected_output="Dataset quality report with QA results.",
            agent=data,
        ),
        # TASK-DATA-02 — Standardize desk dashboards
        create_task(
            description=(
                "Standardize desk dashboards for risk/perf/execution/research. "
                "Ensure consistent metrics across desks."
            ),
            expected_output="Dashboard standardization report.",
            agent=data,
        ),
        # TASK-DATA-03 — Alerting on data corruption/outliers
        create_task(
            description=(
                "Implement alerting on data corruption and outliers. "
                "Detect and flag anomalies before they impact trading decisions. "
                "Maintain data integrity standards."
            ),
            expected_output="Data quality alerting system with detection metrics.",
            agent=data,
        ),
        # TASK-DATA-04 — Alternative data integration
        create_task(
            description=(
                "Introduce alternative data: social sentiment, search trends, developer metrics, etc. "
                "Integrate into dashboards for traders to gain edge."
            ),
            expected_output="Alternative data integration report with source list.",
            agent=data,
        ),
        # TASK-DATA-05 — ML experimentation
        create_task(
            description=(
                "Experiment with machine learning on data: predictive models, anomaly detection "
                "to find hidden alpha. Validate signals rigorously before use."
            ),
            expected_output="ML experimentation report with validated signal candidates.",
            agent=data,
        ),
    ]


def get_people_tasks(people: Agent) -> list[Task]:
    """Get tasks for STAFF-09 People."""
    return [
        # TASK-PEOPLE-01 — Hiring scorecards + interview cases
        create_task(
            description=(
                "Roll out hiring scorecards and interview cases (including CEO cases). "
                "Ensure consistent evaluation of candidates."
            ),
            expected_output="Hiring scorecard implementation report.",
            agent=people,
        ),
        # TASK-PEOPLE-02 — Bonus/clawback/deferral structure
        create_task(
            description=(
                "Implement bonus/clawback/deferral structure. "
                "Ensure compensation aligns with risk-adjusted performance."
            ),
            expected_output="Compensation structure implementation report.",
            agent=people,
        ),
        # TASK-PEOPLE-03 — Performance cycle
        create_task(
            description=(
                "Run performance cycle: metrics → feedback → consequences. "
                "Ensure timely reviews and actionable feedback. "
                "Handle underperformers decisively."
            ),
            expected_output="Performance cycle report with actions taken.",
            agent=people,
        ),
        # TASK-PEOPLE-04 — High-risk/high-reward culture
        create_task(
            description=(
                "Cultivate high-risk/high-reward culture: reward outsized wins (within rules) "
                "with faster promotion and larger bonus. "
                "Address rule-breakers immediately to maintain discipline."
            ),
            expected_output="Culture assessment report with incentive alignment.",
            agent=people,
        ),
    ]


def get_staff_tasks(agents: dict[str, Agent], mode: str = "trading") -> list[Task]:
    """Get STAFF tasks given a dictionary of agents.

    Args:
        agents: Dictionary mapping agent keys to Agent instances.
                Expected keys: ceo, group_cio, group_cro, group_coo,
                group_cfo, compliance, security, prime, data, people
        mode: Task mode - "trading" for active trading operations (default),
              "full" for all tasks including interviews/compensation/90day

    Returns:
        List of STAFF tasks.
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

    # In trading mode, only CEO tasks matter - CEO delegates to desks
    # Other STAFF agents support but don't have primary tasks during trading
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
