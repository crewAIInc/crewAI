from typing import Any, Callable, Dict, Tuple, Union

from pydantic import BaseModel, Field

from crewai.pipeline.pipeline import Pipeline

RouteType = Tuple[Callable[[Dict[str, Any]], bool], Pipeline]


class PipelineRouter(BaseModel):
    routes: Dict[str, RouteType] = Field(
        default_factory=dict,
        description="Dictionary of route names to (condition, pipeline) tuples",
    )
    default: Pipeline = Field(
        ..., description="Default pipeline if no conditions are met"
    )

    def __init__(self, *routes: Union[Tuple[str, RouteType], Pipeline], **data):
        routes_dict = {}
        default_pipeline = None

        for route in routes:
            if isinstance(route, tuple) and len(route) == 2:
                name, route_tuple = route
                if isinstance(route_tuple, tuple) and len(route_tuple) == 2:
                    condition, pipeline = route_tuple
                    routes_dict[name] = (condition, pipeline)
                else:
                    raise ValueError(f"Invalid route tuple structure: {route}")
            elif isinstance(route, Pipeline):
                if default_pipeline is not None:
                    raise ValueError("Only one default pipeline can be specified")
                default_pipeline = route
            else:
                raise ValueError(f"Invalid route type: {type(route)}")

        if default_pipeline is None:
            raise ValueError("A default pipeline must be specified")

        super().__init__(routes=routes_dict, default=default_pipeline, **data)

    def add_route(
        self, name: str, condition: Callable[[Dict[str, Any]], bool], pipeline: Pipeline
    ) -> "PipelineRouter":
        """
        Add a named route with its condition and corresponding pipeline to the router.

        Args:
            name: A unique name for this route
            condition: A function that takes the input dictionary and returns a boolean
            pipeline: The Pipeline to execute if the condition is met

        Returns:
            The PipelineRouter instance for method chaining
        """
        self.routes[name] = (condition, pipeline)
        return self

    def route(self, input_dict: Dict[str, Any]) -> Tuple[Pipeline, str]:
        """
        Evaluate the input against the conditions and return the appropriate pipeline.

        Args:
            input_dict: The input dictionary to be evaluated

        Returns:
            A tuple containing the next Pipeline to be executed and the name of the route taken
        """
        for name, (condition, pipeline) in self.routes.items():
            if condition(input_dict):
                return pipeline, name

        return self.default, "default"
