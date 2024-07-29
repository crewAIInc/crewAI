from typing import TYPE_CHECKING, List, Union

from crewai.crew import Crew

if TYPE_CHECKING:
    from crewai.routers.pipeline_router import PipelineRouter

PipelineStage = Union[Crew, "PipelineRouter", List[Crew]]
