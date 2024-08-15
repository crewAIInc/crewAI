from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

from pydantic import BaseModel, Field, PrivateAttr


class Route(BaseModel):
    condition: Callable[[Dict[str, Any]], bool]
    pipeline: Any


class Router(BaseModel):
    routes: Dict[str, Route] = Field(
        default_factory=dict,
        description="Dictionary of route names to (condition, pipeline) tuples",
    )
    default: Any = Field(..., description="Default pipeline if no conditions are met")
    _route_types: Dict[str, type] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, routes: Dict[str, Route], default: Any, **data):
        super().__init__(routes=routes, default=default, **data)
        self._check_copyable(default)
        for name, route in routes.items():
            self._check_copyable(route.pipeline)
            self._route_types[name] = type(route.pipeline)

    @staticmethod
    def _check_copyable(obj: Any) -> None:
        if not hasattr(obj, "copy") or not callable(getattr(obj, "copy")):
            raise ValueError(f"Object of type {type(obj)} must have a 'copy' method")

    def add_route(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        pipeline: Any,
    ) -> "Router":
        """
        Add a named route with its condition and corresponding pipeline to the router.

        Args:
            name: A unique name for this route
            condition: A function that takes a dictionary input and returns a boolean
            pipeline: The Pipeline to execute if the condition is met

        Returns:
            The Router instance for method chaining
        """
        self._check_copyable(pipeline)
        self.routes[name] = Route(condition=condition, pipeline=pipeline)
        self._route_types[name] = type(pipeline)
        return self

    def route(self, input_data: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Evaluate the input against the conditions and return the appropriate pipeline.

        Args:
            input_data: The input dictionary to be evaluated

        Returns:
            A tuple containing the next Pipeline to be executed and the name of the route taken
        """
        for name, route in self.routes.items():
            if route.condition(input_data):
                return route.pipeline, name

        return self.default, "default"

    def copy(self) -> "Router":
        """Create a deep copy of the Router."""
        new_routes = {
            name: Route(
                condition=deepcopy(route.condition),
                pipeline=route.pipeline.copy(),
            )
            for name, route in self.routes.items()
        }
        new_default = self.default.copy()

        return Router(routes=new_routes, default=new_default)
