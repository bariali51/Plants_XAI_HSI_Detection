# analysis/tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger
import time

logger = get_task_logger(__name__)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
    retry_backoff=True,
)
def test_task(self):
    """
    Task تجريبية:
    - تحاكي عملًا طويلًا
    - تدعم retry تلقائي
    - تسجّل progress و logs
    """
    logger.info("Task started")

    for i in range(5):
        time.sleep(1)
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": 5}
        )

    logger.info("Task finished successfully")
    return {
        "status": "SUCCESS",
        "message": "Task completed successfully!"
    }


def analyze_hyperspectral_task():
    return None