from typing import Callable, Optional

from pydantic import BaseModel

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


class ConditionalTask(Task):
    condition: Optional[Callable[[BaseModel], bool]] = None

    def __init__(
        self,
        *args,
        condition: Optional[Callable[[BaseModel], bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.condition = condition

    def should_execute(self, context: TaskOutput) -> bool:
        print("TaskOutput", TaskOutput)
        if self.condition:
            return self.condition(context)
        return True
