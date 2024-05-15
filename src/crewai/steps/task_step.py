from typing import Callable, Dict, Any, List, Optional, Union

from crewai.steps.step import Step


class TaskStep(Step):
    def __init__(self, task: Union[Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]], Callable[[], List[Dict[str, Any]]]]):
        self.task = task

    def kickoff(self, inputs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if inputs is not None:
            return self.task(inputs)
        else:
            return self.task()
