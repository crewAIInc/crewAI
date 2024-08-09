from typing import Callable, Dict

from pydantic import ConfigDict

from crewai.crew import Crew
from crewai.pipeline.pipeline import Pipeline
from crewai.routers.router import Router


# TODO: Could potentially remove. Need to check with @joao and @gui if this is needed for CrewAI+
def PipelineBase(cls):
    class WrappedClass(cls):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        is_pipeline_class: bool = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stages = []
            self._map_pipeline_components()

        def _get_all_functions(self):
            return {
                name: getattr(self, name)
                for name in dir(self)
                if callable(getattr(self, name))
            }

        def _filter_functions(
            self, functions: Dict[str, Callable], attribute: str
        ) -> Dict[str, Callable]:
            return {
                name: func
                for name, func in functions.items()
                if hasattr(func, attribute)
            }

        def _map_pipeline_components(self):
            all_functions = self._get_all_functions()
            crew_functions = self._filter_functions(all_functions, "is_crew")
            router_functions = self._filter_functions(all_functions, "is_router")

            for stage_attr in dir(self):
                stage = getattr(self, stage_attr)
                if isinstance(stage, (Crew, Router)):
                    self.stages.append(stage)
                elif callable(stage) and hasattr(stage, "is_crew"):
                    self.stages.append(crew_functions[stage_attr]())
                elif callable(stage) and hasattr(stage, "is_router"):
                    self.stages.append(router_functions[stage_attr]())
                elif isinstance(stage, list) and all(
                    isinstance(item, Crew) for item in stage
                ):
                    self.stages.append(stage)

        def build_pipeline(self) -> Pipeline:
            return Pipeline(stages=self.stages)

    return WrappedClass
