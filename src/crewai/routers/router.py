from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=Dict[str, Any])
U = TypeVar("U")


@dataclass
class Route(Generic[T, U]):
    condition: Callable[[T], bool]
    pipeline: U


class Router(BaseModel, Generic[T, U]):
    routes: Dict[str, Route[T, U]] = Field(
        default_factory=dict,
        description="Dictionary of route names to (condition, pipeline) tuples",
    )
    default: U = Field(..., description="Default pipeline if no conditions are met")

    def __init__(self, routes: Dict[str, Route[T, U]], default: U, **data):
        super().__init__(routes=routes, default=default, **data)

    def add_route(
        self,
        name: str,
        condition: Callable[[T], bool],
        pipeline: U,
    ) -> "Router[T, U]":
        """
        Add a named route with its condition and corresponding pipeline to the router.

        Args:
            name: A unique name for this route
            condition: A function that takes a dictionary input and returns a boolean
            pipeline: The Pipeline to execute if the condition is met

        Returns:
            The Router instance for method chaining
        """
        self.routes[name] = Route(condition=condition, pipeline=pipeline)
        return self

    def route(self, input_data: T) -> Tuple[U, str]:
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
