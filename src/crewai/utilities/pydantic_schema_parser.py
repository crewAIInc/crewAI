from typing import Type, Union, get_args, get_origin

from pydantic import BaseModel


class PydanticSchemaParser(BaseModel):
    model: Type[BaseModel]

    def get_schema(self) -> str:
        """
        Public method to get the schema of a Pydantic model.

        :param model: The Pydantic model class to generate schema for.
        :return: String representation of the model schema.
        """
        return self._get_model_schema(self.model)

    def _get_model_schema(self, model, depth=0) -> str:
        indent = "    " * depth
        lines = [f"{indent}{{"]
        for field_name, field in model.model_fields.items():
            field_type_str = self._get_field_type(field, depth + 1)
            lines.append(f"{indent}    {field_name}: {field_type_str},")
        lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma from last item
        lines.append(f"{indent}}}")
        return "\n".join(lines)

    def _get_field_type(self, field, depth) -> str:
        field_type = field.annotation
        if get_origin(field_type) is list:
            list_item_type = get_args(field_type)[0]
            if isinstance(list_item_type, type) and issubclass(
                list_item_type, BaseModel
            ):
                nested_schema = self._get_model_schema(list_item_type, depth + 1)
                return f"List[\n{nested_schema}\n{' ' * 4 * depth}]"
            else:
                return f"List[{list_item_type.__name__}]"
        elif get_origin(field_type) is Union:
            union_args = get_args(field_type)
            if type(None) in union_args:
                non_none_type = next(arg for arg in union_args if arg is not type(None))
                return f"Optional[{self._get_field_type(field.__class__(annotation=non_none_type), depth)}]"
            else:
                return f"Union[{', '.join(arg.__name__ for arg in union_args)}]"
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return self._get_model_schema(field_type, depth)
        else:
            return getattr(field_type, "__name__", str(field_type))
