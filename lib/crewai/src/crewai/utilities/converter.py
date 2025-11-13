from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
import re
from typing import TYPE_CHECKING, Any, Final, TypedDict

from pydantic import BaseModel, ValidationError
from typing_extensions import Unpack

from crewai.agents.agent_builder.utilities.base_output_converter import OutputConverter
from crewai.utilities.i18n import get_i18n
from crewai.utilities.internal_instructor import InternalInstructor
from crewai.utilities.printer import Printer


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM

_JSON_PATTERN: Final[re.Pattern[str]] = re.compile(r"({.*})", re.DOTALL)
_I18N = get_i18n()


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        """Initialize the ConverterError with a message.

        Args:
            message: The error message.
            *args: Additional arguments for the base Exception class.
        """
        super().__init__(message, *args)
        self.message = message


class Converter(OutputConverter):
    """Class that converts text into either pydantic or json."""

    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        """Convert text to pydantic.

        Args:
            current_attempt: The current attempt number for conversion retries.

        Returns:
            A Pydantic BaseModel instance.

        Raises:
            ConverterError: If conversion fails after maximum attempts.
        """
        try:
            if self.llm.supports_function_calling():
                response = self.llm.call(
                    messages=[
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ],
                    response_model=self.model,
                )
                result = self.model.model_validate_json(response)
            else:
                response = self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
                try:
                    # Try to directly validate the response JSON
                    result = self.model.model_validate_json(response)
                except ValidationError:
                    # If direct validation fails, attempt to extract valid JSON
                    result = handle_partial_json(  # type: ignore[assignment]
                        result=response,
                        model=self.model,
                        is_json_output=False,
                        agent=None,
                    )
                    # Ensure result is a BaseModel instance
                    if not isinstance(result, BaseModel):
                        if isinstance(result, dict):
                            result = self.model.model_validate(result)
                        elif isinstance(result, str):
                            try:
                                result = self.model.model_validate_json(result)
                            except Exception as parse_err:
                                raise ConverterError(
                                    f"Failed to convert partial JSON result into Pydantic: {parse_err}"
                                ) from parse_err
                        else:
                            raise ConverterError(
                                "handle_partial_json returned an unexpected type."
                            ) from None
            return result
        except ValidationError as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to validation error: {e}"
            ) from e
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to error: {e}"
            ) from e

    def to_json(self, current_attempt: int = 1) -> str | ConverterError | Any:  # type: ignore[override]
        """Convert text to json.

        Args:
            current_attempt: The current attempt number for conversion retries.

        Returns:
            A JSON string or ConverterError if conversion fails.

        Raises:
            ConverterError: If conversion fails after maximum attempts.

        """
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_json()
            return json.dumps(
                self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
            )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_json(current_attempt + 1)
            return ConverterError(f"Failed to convert text into JSON, error: {e}.")

    def _create_instructor(self) -> InternalInstructor[Any]:
        """Create an instructor."""

        return InternalInstructor(
            llm=self.llm,
            model=self.model,
            content=self.text,
        )


def convert_to_model(
    result: str,
    output_pydantic: type[BaseModel] | None,
    output_json: type[BaseModel] | None,
    agent: Agent | BaseAgent | None = None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Convert a result string to a Pydantic model or JSON.

    Args:
        result: The result string to convert.
        output_pydantic: The Pydantic model class to convert to.
        output_json: The Pydantic model class to convert to JSON.
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.
    """
    model = output_pydantic or output_json
    if model is None:
        return result

    if converter_cls:
        return convert_with_instructions(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(
            result=escaped_result, model=model, is_json_output=bool(output_json)
        )
    except json.JSONDecodeError:
        return handle_partial_json(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    except ValidationError:
        return handle_partial_json(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    except Exception as e:
        Printer().print(
            content=f"Unexpected error during model conversion: {type(e).__name__}: {e}. Returning original result.",
            color="red",
        )
        return result


def validate_model(
    result: str, model: type[BaseModel], is_json_output: bool
) -> dict[str, Any] | BaseModel:
    """Validate and convert a JSON string to a Pydantic model or dict.

    Args:
        result: The JSON string to validate and convert.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).

    Returns:
        The converted result as a dict or BaseModel.
    """
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str,
    model: type[BaseModel],
    is_json_output: bool,
    agent: Agent | BaseAgent | None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Handle partial JSON in a result string and convert to Pydantic model or dict.

    Args:
        result: The result string to process.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.
    """
    match = _JSON_PATTERN.search(result)
    if match:
        try:
            exported_result = model.model_validate_json(match.group())
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except json.JSONDecodeError:
            pass
        except ValidationError:
            raise
        except Exception as e:
            Printer().print(
                content=f"Unexpected error during partial JSON handling: {type(e).__name__}: {e}. Attempting alternative conversion method.",
                color="red",
            )

    return convert_with_instructions(
        result=result,
        model=model,
        is_json_output=is_json_output,
        agent=agent,
        converter_cls=converter_cls,
    )


def convert_with_instructions(
    result: str,
    model: type[BaseModel],
    is_json_output: bool,
    agent: Agent | BaseAgent | None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Convert a result string to a Pydantic model or JSON using instructions.

    Args:
        result: The result string to convert.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.

    Raises:
        TypeError: If neither agent nor converter_cls is provided.

    Notes:
        - TODO: Fix llm typing issues, return llm should not be able to be str or None.
    """
    if agent is None:
        raise TypeError("Agent must be provided if converter_cls is not specified.")

    llm = getattr(agent, "function_calling_llm", None) or agent.llm

    if llm is None:
        raise ValueError("Agent must have a valid LLM instance for conversion")

    instructions = get_conversion_instructions(model=model, llm=llm)
    converter = create_converter(
        agent=agent,
        converter_cls=converter_cls,
        llm=llm,
        text=result,
        model=model,
        instructions=instructions,
    )
    exported_result = (
        converter.to_pydantic() if not is_json_output else converter.to_json()
    )

    if isinstance(exported_result, ConverterError):
        Printer().print(
            content=f"Failed to convert result to model: {exported_result}",
            color="red",
        )
        return result

    return exported_result


def get_conversion_instructions(
    model: type[BaseModel], llm: BaseLLM | LLM | str | Any
) -> str:
    """Generate conversion instructions based on the model and LLM capabilities.

    Args:
        model: A Pydantic model class.
        llm: The language model instance.

    Returns:

    """
    instructions = ""
    if (
        llm
        and not isinstance(llm, str)
        and hasattr(llm, "supports_function_calling")
        and llm.supports_function_calling()
    ):
        schema_dict = generate_model_description(model)
        schema = json.dumps(schema_dict, indent=2)
        formatted_task_instructions = _I18N.slice("formatted_task_instructions").format(
            output_format=schema
        )
        instructions += formatted_task_instructions
    else:
        model_description = generate_model_description(model)
        schema_json = json.dumps(model_description, indent=2)
        formatted_task_instructions = _I18N.slice("formatted_task_instructions").format(
            output_format=schema_json
        )
        instructions += formatted_task_instructions
    return instructions


class CreateConverterKwargs(TypedDict, total=False):
    """Keyword arguments for creating a converter.

    Attributes:
        llm: The language model instance.
        text: The text to convert.
        model: The Pydantic model class.
        instructions: The conversion instructions.
    """

    llm: BaseLLM | LLM | str
    text: str
    model: type[BaseModel]
    instructions: str


def create_converter(
    agent: Agent | BaseAgent | None = None,
    converter_cls: type[Converter] | None = None,
    *args: Any,
    **kwargs: Unpack[CreateConverterKwargs],
) -> Converter:
    """Create a converter instance based on the agent or provided class.

    Args:
        agent: The agent instance.
        converter_cls: The converter class to instantiate.
        *args: The positional arguments to pass to the converter.
        **kwargs: The keyword arguments to pass to the converter.

    Returns:
        An instance of the specified converter class.

    Raises:
        ValueError: If neither agent nor converter_cls is provided.
        AttributeError: If the agent does not have a 'get_output_converter' method.
        Exception: If no converter instance is created.

    """
    if agent and not converter_cls:
        if hasattr(agent, "get_output_converter"):
            converter = agent.get_output_converter(*args, **kwargs)
        else:
            raise AttributeError("Agent does not have a 'get_output_converter' method")
    elif converter_cls:
        converter = converter_cls(*args, **kwargs)
    else:
        raise ValueError("Either agent or converter_cls must be provided")

    if not converter:
        raise Exception("No output converter found or set.")

    return converter  # type: ignore[no-any-return]


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve all local $refs in the given JSON Schema using $defs as the source.

    This is needed because Pydantic generates $ref-based schemas that
    some consumers (e.g. LLMs, tool frameworks) don't handle well.

    Args:
        schema: JSON Schema dict that may contain "$refs" and "$defs".

    Returns:
        A new schema dictionary with all local $refs replaced by their definitions.
    """
    defs = schema.get("$defs", {})
    schema_copy = deepcopy(schema)

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                if def_name in defs:
                    return _resolve(deepcopy(defs[def_name]))
                raise KeyError(f"Definition '{def_name}' not found in $defs.")
            return {k: _resolve(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_resolve(i) for i in node]

        return node

    return _resolve(schema_copy)  # type: ignore[no-any-return]


def add_key_in_dict_recursively(
    d: dict[str, Any], key: str, value: Any, criteria: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    """Recursively adds a key/value pair to all nested dicts matching `criteria`."""
    if isinstance(d, dict):
        if criteria(d) and key not in d:
            d[key] = value
        for v in d.values():
            add_key_in_dict_recursively(v, key, value, criteria)
    elif isinstance(d, list):
        for i in d:
            add_key_in_dict_recursively(i, key, value, criteria)
    return d


def fix_discriminator_mappings(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace '#/$defs/...' references in discriminator.mapping with just the model name."""
    output = schema.get("properties", {}).get("output")
    if not output:
        return schema

    disc = output.get("discriminator")
    if not disc or "mapping" not in disc:
        return schema

    disc["mapping"] = {k: v.split("/")[-1] for k, v in disc["mapping"].items()}
    return schema


def add_const_to_oneof_variants(schema: dict[str, Any]) -> dict[str, Any]:
    """Add const fields to oneOf variants for discriminated unions.

    The json_schema_to_pydantic library requires each oneOf variant to have
    a const field for the discriminator property. This function adds those
    const fields based on the discriminator mapping.

    Args:
        schema: JSON Schema dict that may contain discriminated unions

    Returns:
        Modified schema with const fields added to oneOf variants
    """

    def _process_oneof(node: dict[str, Any]) -> dict[str, Any]:
        """Process a single node that might contain a oneOf with discriminator."""
        if not isinstance(node, dict):
            return node

        if "oneOf" in node and "discriminator" in node:
            discriminator = node["discriminator"]
            property_name = discriminator.get("propertyName")
            mapping = discriminator.get("mapping", {})

            if property_name and mapping:
                one_of_variants = node.get("oneOf", [])

                for variant in one_of_variants:
                    if isinstance(variant, dict) and "properties" in variant:
                        variant_title = variant.get("title", "")

                        matched_disc_value = None
                        for disc_value, schema_name in mapping.items():
                            if variant_title == schema_name or variant_title.endswith(
                                schema_name
                            ):
                                matched_disc_value = disc_value
                                break

                        if matched_disc_value is not None:
                            props = variant["properties"]
                            if property_name in props:
                                props[property_name]["const"] = matched_disc_value

        for key, value in node.items():
            if isinstance(value, dict):
                node[key] = _process_oneof(value)
            elif isinstance(value, list):
                node[key] = [
                    _process_oneof(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return node

    return _process_oneof(deepcopy(schema))


def convert_oneof_to_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert oneOf to anyOf for OpenAI compatibility.

    OpenAI's Structured Outputs support anyOf better than oneOf.
    This recursively converts all oneOf occurrences to anyOf.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Modified schema with anyOf instead of oneOf.
    """
    if isinstance(schema, dict):
        if "oneOf" in schema:
            schema["anyOf"] = schema.pop("oneOf")

        for value in schema.values():
            if isinstance(value, dict):
                convert_oneof_to_anyof(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_oneof_to_anyof(item)

    return schema


def ensure_all_properties_required(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure all properties are in the required array for OpenAI strict mode.

    OpenAI's strict structured outputs require all properties to be listed
    in the required array. This recursively updates all objects to include
    all their properties in required.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Modified schema with all properties marked as required.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema["properties"]
            if properties:
                schema["required"] = list(properties.keys())

        for value in schema.values():
            if isinstance(value, dict):
                ensure_all_properties_required(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        ensure_all_properties_required(item)

    return schema


def generate_model_description(model: type[BaseModel]) -> dict[str, Any]:
    """Generate JSON schema description of a Pydantic model.

    This function takes a Pydantic model class and returns its JSON schema,
    which includes full type information, discriminators, and all metadata.
    The schema is dereferenced to inline all $ref references for better LLM understanding.

    Args:
        model: A Pydantic model class.

    Returns:
        A JSON schema dictionary representation of the model.
    """

    json_schema = model.model_json_schema(ref_template="#/$defs/{model}")

    json_schema = add_key_in_dict_recursively(
        json_schema,
        key="additionalProperties",
        value=False,
        criteria=lambda d: d.get("type") == "object"
        and "additionalProperties" not in d,
    )

    json_schema = resolve_refs(json_schema)

    json_schema.pop("$defs", None)
    json_schema = fix_discriminator_mappings(json_schema)
    json_schema = convert_oneof_to_anyof(json_schema)
    json_schema = ensure_all_properties_required(json_schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": json_schema,
        },
    }
