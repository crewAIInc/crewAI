import threading
from typing import List, Any

class ParallelAgentExecutor:
    """
    Parallel Executor for multi-agent task completion.
    Optimizes CrewAI workflows by running non-dependent tasks concurrently.
    """
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def execute_tasks(self, tasks: List[Any]):
        threads = []
        for task in tasks:
            t = threading.Thread(target=task.execute)
            threads.append(t)
            t.start()
            if len(threads) >= self.max_workers:
                for thread in threads:
                    thread.join()
                threads = []
        for thread in threads:
            thread.join()
