from datetime import datetime
import uuid
from crew_trader_pro.app.celery_config import celery_app
from crew_trader_pro.app.tools.technical_analyst_tools import TechnicalAnalystTools
from crew_trader_pro.main import _build_default_learning_bridge



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
    print(f"[Local Run] ✅ 技术分析数据生成完成")

    # 2. 将结构化数据传入 Crew
    inputs = {
        "symbol": symbol,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "technical_input": technical_input.model_dump(mode="json"),
    }

    print("inputs:", inputs)  # ✅ 输出输入数据，方便调试
    print("test print one")

    # 3. 返回结果 → Celery 自动存入 Redis (key: celery-task-meta-<task_id>)
    return inputs