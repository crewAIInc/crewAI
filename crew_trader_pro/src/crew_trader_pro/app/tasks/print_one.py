from crew_trader_pro.app.celery_config import celery_app

@celery_app.task(name="crew_trader_pro.app.tasks.print_one")
def print_one():
    print(1)