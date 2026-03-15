#!/usr/bin/env python
import subprocess  # ✅ 正确：直接使用标准库
import sys
import os
import warnings
import uvicorn
from fastapi import FastAPI
from datetime import datetime
from crew_trader_pro.crew import CrewTraderPro
from dotenv import load_dotenv
from crew_trader_pro.app.tools.technical_analyst_tools import TechnicalAnalystTools, BarStatus, EmaAlignment, SlopeState

load_dotenv()
# 过滤警告
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# --- 1. 初始化 FastAPI 实例 ---
app = FastAPI(title="CrewTraderPro_Server")

@app.get("/")
def health_check():
    self = TechnicalAnalystTools().generate_input()  # 测试工具类是否能正常初始化
    return {
        "status": "online",
        "service": "CrewTraderPro Trading Server",
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/manual_kickoff")
def manual_kickoff():
    """通过 API 手动触发一次分析流"""
    run()
    return {"message": "Crew execution triggered locally."}

# --- 2. 启动服务端的辅助方法 ---
def start_server():
    """一键启动：FastAPI + Celery Worker + Celery Beat"""
    
    # 确保 PYTHONPATH 包含 src
    os.environ["PYTHONPATH"] = "src"
    
    # 启动 Celery Worker 和 Beat 作为子进程  (苦力/执行员)：它盯着 Redis 队列。一旦看到信封，它就拆开并执行里面的代码（比如你写的打印 1 或者去抓取币安数据）。
    print("🐝 Starting Celery Worker...")
    worker_command = [
        sys.executable, "-m", "celery", 
        "-A", "crew_trader_pro.app.celery_config:celery_app", 
        "worker", 
        "--loglevel=info", 
        "--pool=solo"  # <--- 这一行是解决你报错的关键
    ]
    worker_proc = subprocess.Popen(worker_command)
    
    # 启动 Celery Beat 作为子进程 (钟表匠/调度员)：它负责定时发出信号（比如每 10 秒），告诉 Worker 去执行某个任务（比如 print_one）。它就像个闹钟，到了时间就响，Worker 就去执行任务。
    print("⏰ Starting Celery Beat...")
    beat_command = [
        sys.executable, "-m", "celery", 
        "-A", "crew_trader_pro.app.celery_config:celery_app", 
        "beat", 
        "--loglevel=info"
    ]
    beat_proc = subprocess.Popen(beat_command)

    try:
        print("🚀 Starting FastAPI Server...")
        uvicorn.run("crew_trader_pro.main:app", host="0.0.0.0", port=8000, reload=False) # 生产模式建议 reload=False
    finally:
        # 当你 Ctrl+C 关闭 FastAPI 时，自动杀掉后台的 Celery 进程
        print("\n🛑 Shutting down Celery processes...")
        worker_proc.terminate()
        beat_proc.terminate()
# def start_server():
#     """供 pyproject.toml 调用启动 FastAPI"""
#     uvicorn.run("crew_trader_pro.main:app", host="0.0.0.0", port=8000, reload=True)


# --- 3. 原有的 CrewAI 逻辑 (保留并微调) ---

def run():
    """
    Run the crew locally.
    """
    # 模拟输入，后续将被 MarketDataService 替换
    inputs = {
        'symbol': 'BTC/USDT',
        'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"\n[Local Run] 🚀 正在执行交易决策流...")
    try:
        # 这里实例化你的 Crew 类
        result = CrewTraderPro().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        # 在服务端模式下建议 logging 而不是 raise

def train():
    # ... 保留原逻辑 ...
    inputs = {'symbol': 'BTC/USDT', 'current_year': str(datetime.now().year)}
    try:
        CrewTraderPro().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training: {e}")

def replay():
    # ... 保留原逻辑 ...
    try:
        CrewTraderPro().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying: {e}")

def test():
    # ... 保留原逻辑 ...
    inputs = {'symbol': 'BTC/USDT', 'current_year': str(datetime.now().year)}
    try:
        CrewTraderPro().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = CrewTraderPro().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
