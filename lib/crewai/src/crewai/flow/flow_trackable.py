import inspect

from pydantic import BaseModel, Field, InstanceOf, model_validator

from crewai.flow import Flow


class FlowTrackable(BaseModel):
    """Mixin that tracks the Flow instance that instantiated the object, e.g. a
    Flow instance that created a Crew or Agent.

    Automatically finds and stores a reference to the parent Flow instance by
    inspecting the call stack.
    """

    parent_flow: InstanceOf[Flow] | None = Field(
        default=None,
        description="The parent flow of the instance, if it was created inside a flow.",
    )

    @model_validator(mode="after")
    def _set_parent_flow(self) -> "FlowTrackable":
        max_depth = 5
        frame = inspect.currentframe()

        try:
            if frame is None:
                return self

            frame = frame.f_back
            for _ in range(max_depth):
                if frame is None:
                    break

                candidate = frame.f_locals.get("self")
                if isinstance(candidate, Flow):
                    self.parent_flow = candidate
                    break

                frame = frame.f_back
        finally:
            del frame

        return self
