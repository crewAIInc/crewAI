from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, Optional, Type, Tuple, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model, validator
from pydantic.fields import FieldInfo
from pydantic import BaseModel as PydanticBaseModel

def _create_model_fields(fields: Dict[str, Tuple[Any, FieldInfo]]) -> Dict[str, Any]:
    """Helper function to create model fields with proper type hints."""
    return {name: (annotation, field) for name, (annotation, field) in fields.items()}

class BaseTool(BaseModel, ABC):
    """Base class for all tools."""

    class _ArgsSchemaPlaceholder(PydanticBaseModel):
        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)
    func: Optional[Callable] = None

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    args_schema: Type[PydanticBaseModel] = Field(default=_ArgsSchemaPlaceholder)
    """The schema for the arguments that the tool accepts."""
    description_updated: bool = False
    """Flag to check if the description has been updated."""
    cache_function: Callable = lambda _args=None, _result=None: True
    """Function that will be used to determine if the tool should be cached."""
    result_as_answer: bool = False
    """Flag to check if the tool should be the final agent answer."""

    @validator("args_schema", always=True, pre=True)
    def _default_args_schema(
        cls, v: Type[PydanticBaseModel]
    ) -> Type[PydanticBaseModel]:
        if not isinstance(v, cls._ArgsSchemaPlaceholder):
            return v

        return type(
            f"{cls.__name__}Schema",
            (PydanticBaseModel,),
            {
                "__annotations__": {
                    k: v for k, v in cls._run.__annotations__.items() if k != "return"
                },
            },
        )

    def model_post_init(self, __context: Any) -> None:
        self._generate_description()
        super().model_post_init(__context)

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Here goes the actual implementation of the tool."""

    def _set_args_schema(self) -> None:
        if self.args_schema is None:
            class_name = f"{self.__class__.__name__}Schema"
            self.args_schema = type(
                class_name,
                (PydanticBaseModel,),
                {
                    "__annotations__": {
                        k: v
                        for k, v in self._run.__annotations__.items()
                        if k != "return"
                    },
                },
            )

    def _generate_description(self) -> None:
        args_schema = {
            name: {
                "description": field.description,
                "type": BaseTool._get_arg_annotations(field.annotation),
            }
            for name, field in self.args_schema.model_fields.items()
        }

        self.description = f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nTool Description: {self.description}"

    @staticmethod
    def _get_arg_annotations(annotation: type[Any] | None) -> str:
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            return (
                annotation.__name__
                if hasattr(annotation, "__name__")
                else str(annotation)
            )

        if args:
            args_str = ", ".join(BaseTool._get_arg_annotations(arg) for arg in args)
            return f"{origin.__name__}[{args_str}]"

        return origin.__name__


class Tool(BaseTool):
    """Tool class that wraps a function."""

    func: Callable
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        if "func" not in kwargs:
            raise ValueError("Tool requires a 'func' argument")
        super().__init__(**kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(*args: Any) -> Any:
    """Decorator to create a tool from a function."""

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(f: Callable) -> Tool:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")
            if f.__annotations__ is None:
                raise ValueError("Function must have type annotations")

            class_name = "".join(tool_name.split()).title()
            args_schema = type(
                class_name,
                (PydanticBaseModel,),
                {
                    "__annotations__": {
                        k: v for k, v in f.__annotations__.items() if k != "return"
                    },
                },
            )

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
                args_schema=args_schema,
            )

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])
    raise ValueError("Invalid arguments")
