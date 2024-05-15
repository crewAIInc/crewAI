from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator

from crewai.steps.step import Step


class Workflow(BaseModel):
    initial_inputs: List[Dict[str, Any]]
    steps: List[Step] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='before')
    def check_crew_or_task(cls, values):
        step = values.get('step')
        if step is None:
            raise ValueError("A Workflow must be initialized with a Step.")
        return values

    def add_step(self, step: Step, ) -> 'Workflow':
        self.steps.append(step)
        return self

    def run(self) -> List[Dict[str, Any]]:
        results = self.initial_inputs

        for step in self.steps:
            results = step.kickoff(results)

        self.results = results
        return self.results
