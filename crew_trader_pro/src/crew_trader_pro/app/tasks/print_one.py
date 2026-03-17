from datetime import datetime
import json
import uuid
from crew_trader_pro.app.celery_config import celery_app
from crew_trader_pro.app.tools.technical_analyst_tools import TechnicalAnalystTools
from crew_trader_pro.main import _build_default_learning_bridge
from crew_trader_pro.crew import CrewTraderPro


@celery_app.task(name="crew_trader_pro.app.tasks.print_one")
def print_one():
    # 1. 调用工具获取技术分析输入
    symbol = "ETH/USDT"
    request_id = str(uuid.uuid4())
    tools = TechnicalAnalystTools()
    learning_bridge = _build_default_learning_bridge()
    technical_input = tools.generate_input(
        symbol=symbol,
        request_id=request_id,
        learning_bridge=learning_bridge,
    )
    print(f"[Celery] ✅ 技术分析数据生成完成")

    # 2. 拆分数据给不同 Agent
    input_data = technical_input.model_dump(mode="json")
    current_price = input_data["market_snapshot"]["current_price"]

    inputs = {
        "symbol": symbol,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": current_price,
        # Agent 1: K线形态分析 — 只需要 K 线数据
        "market_snapshot": json.dumps(
            input_data["market_snapshot"],
            ensure_ascii=False, indent=2
        ),
        # Agent 2: 技术指标分析 — 只需要指标数据
        "technical_indicators": json.dumps(
            input_data["technical_indicators"],
            ensure_ascii=False, indent=2
        ),
        # Agent 4: 首席策略师 — 需要历史纠错数据
        "learning_bridge": json.dumps(
            learning_bridge.model_dump(mode="json"),
            ensure_ascii=False, indent=2
        ),
    }

    # 3. 启动 CrewAI 顺序执行 4 个任务
    print(f"[Celery] 🤖 正在启动交易决策分析流...")
    print(f"[Celery] 📊 Symbol: {symbol} | Price: {current_price} | Request: {request_id}")
    result = CrewTraderPro().crew().kickoff(inputs=inputs)
    print(f"[Celery] ✅ 交易决策分析完成")

    return {"symbol": symbol, "request_id": request_id, "analysis": str(result)}