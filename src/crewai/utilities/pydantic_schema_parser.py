from typing import Type, get_args, get_origin

from pydantic import BaseModel


class PydanticSchemaParser(BaseModel):
    """This class is responsible for parsing the schema of a Pydantic model.
    It takes a Pydantic model as input and generates a string representation of its schema."""

    model: Type[BaseModel]  # The Pydantic model to parse

    def get_schema(self) -> str:
        """
        This is a public method that initiates the schema generation process for the Pydantic model.

        It calls the private method _get_model_schema with the model as an argument.
        :return: String representation of the model schema.
        """
        return self._get_model_schema(self.model)

    def _get_model_schema(self, model, depth=0) -> str:
        """
        This is a private method that generates the schema for a given Pydantic model.

        It iterates over the fields of the model, generates a string representation for each field type,
        and appends it to a list of lines. The lines are then joined into a single string.
        :param model: The Pydantic model to generate schema for.
        :param depth: The current depth of nested models. Used for indentation.
        :return: String representation of the model schema.
        """
        lines = []
        for field_name, field in model.model_fields.items():
            field_type_str = self._get_field_type(field, depth + 1)
            lines.append(f"{' ' * 4 * depth}- {field_name}: {field_type_str}")

        return "\n".join(lines)

    def _get_field_type(self, field, depth) -> str:
        """
        This is a private method that generates a string representation for a given field type.

        If the field type is a list, it checks the type of the list items. If the items are of a Pydantic model type,
        it generates a nested schema for the model. Otherwise, it simply returns the name of the item type.
        If the field type is a Pydantic model, it generates a nested schema for the model.
        Otherwise, it simply returns the name of the field type.
        :param field: The field to generate a type string for.
        :param depth: The current depth of nested models. Used for indentation.
        :return: String representation of the field type.
        """
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
            return f"\n{self._get_model_schema(field_type, depth)}"
        else:
            return field_type.__name__
