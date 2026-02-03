from pydantic import BaseModel, model_validator
from typing_extensions import Self

from crewai.flow.flow_context import current_flow_id, current_flow_request_id


class FlowTrackable(BaseModel):
    """Mixin that tracks flow execution context for objects created within flows.

    When a Crew or Agent is instantiated inside a flow execution, this mixin
    automatically captures the flow ID and request ID from context variables,
    enabling proper tracking and association with the parent flow execution.
    """

    @model_validator(mode="after")
    def _set_flow_context(self) -> Self:
        request_id = current_flow_request_id.get()
        if request_id:
            self._request_id = request_id
            self._flow_id = current_flow_id.get()

        return self
