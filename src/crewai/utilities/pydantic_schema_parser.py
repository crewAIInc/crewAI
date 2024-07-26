from typing import Type, get_args, get_origin

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
        elif issubclass(field_type, BaseModel):
            return self._get_model_schema(field_type, depth)
        else:
            return field_type.__name__
