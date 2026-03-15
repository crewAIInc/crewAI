import os
from celery import Celery
from datetime import timedelta

# 初始化 Celery 实例
celery_app = Celery(
    "crew_trader_pro",
    # 如果环境变量不存在，默认使用本地 Redis
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# 配置定时任务
celery_app.conf.beat_schedule = {
    "print-1-every-10-seconds": {
        "task": "crew_trader_pro.app.tasks.print_one",
        "schedule": timedelta(seconds=10), # 每 10 秒打印一次，方便观察
    },
}

# 必须导入任务文件，否则 Worker 找不到任务定义
celery_app.conf.update(
    imports=['crew_trader_pro.app.tasks.print_one'],
    result_serializer='json',         # 结果用 JSON 序列化
    accept_content=['json'],          # 只接受 JSON
    result_expires=timedelta(hours=24),  # 结果在 Redis 中保留 24 小时
    task_track_started=True,          # 跟踪任务开始状态
)