from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field


class PydanticSchemaParser(BaseModel):
    model: type[BaseModel] = Field(..., description="The Pydantic model to parse.")

    def get_schema(self) -> str:
        """Public method to get the schema of a Pydantic model.

        Returns:
            String representation of the model schema.
        """
        return "{\n" + self._get_model_schema(self.model) + "\n}"

    def _get_model_schema(self, model: type[BaseModel], depth: int = 0) -> str:
        """Recursively get the schema of a Pydantic model, handling nested models and lists.

        Args:
            model: The Pydantic model to process.
            depth: The current depth of recursion for indentation purposes.

        Returns:
            A string representation of the model schema.
        """
        indent: str = " " * 4 * depth
        lines: list[str] = [
            f"{indent}    {field_name}: {self._get_field_type_for_annotation(field.annotation, depth + 1)}"
            for field_name, field in model.model_fields.items()
        ]
        return ",\n".join(lines)

    def _format_list_type(self, list_item_type: Any, depth: int) -> str:
        """Format a List type, handling nested models if necessary.

        Args:
            list_item_type: The type of items in the list.
            depth: The current depth of recursion for indentation purposes.

        Returns:
            A string representation of the List type.
        """
        if isinstance(list_item_type, type) and issubclass(list_item_type, BaseModel):
            nested_schema = self._get_model_schema(list_item_type, depth + 1)
            nested_indent = " " * 4 * depth
            return f"List[\n{nested_indent}{{\n{nested_schema}\n{nested_indent}}}\n{nested_indent}]"
        return f"List[{list_item_type.__name__}]"

    def _format_union_type(self, field_type: Any, depth: int) -> str:
        """Format a Union type, handling Optional and nested types.

        Args:
            field_type: The Union type to format.
            depth: The current depth of recursion for indentation purposes.

        Returns:
            A string representation of the Union type.
        """
        args = get_args(field_type)
        if type(None) in args:
            # It's an Optional type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                inner_type = self._get_field_type_for_annotation(
                    non_none_args[0], depth
                )
                return f"Optional[{inner_type}]"
            # Union with None and multiple other types
            inner_types = ", ".join(
                self._get_field_type_for_annotation(arg, depth) for arg in non_none_args
            )
            return f"Optional[Union[{inner_types}]]"
        # General Union type
        inner_types = ", ".join(
            self._get_field_type_for_annotation(arg, depth) for arg in args
        )
        return f"Union[{inner_types}]"

    def _get_field_type_for_annotation(self, annotation: Any, depth: int) -> str:
        """Recursively get the string representation of a field's type annotation.

        Args:
            annotation: The type annotation to process.
            depth: The current depth of recursion for indentation purposes.

        Returns:
            A string representation of the type annotation.
        """
        origin: Any = get_origin(annotation)
        if origin is list:
            list_item_type = get_args(annotation)[0]
            return self._format_list_type(list_item_type, depth)
        if origin is dict:
            key_type, value_type = get_args(annotation)
            return f"Dict[{key_type.__name__}, {value_type.__name__}]"
        if origin is Union:
            return self._format_union_type(annotation, depth)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_schema = self._get_model_schema(annotation, depth)
            nested_indent = " " * 4 * depth
            return f"{annotation.__name__}\n{nested_indent}{{\n{nested_schema}\n{nested_indent}}}"
        return annotation.__name__
