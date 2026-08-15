"""Markdown skill rendering for Flow Definition authoring."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader
import yaml

from crewai.flow.expressions import (
    FLOW_TEMPLATE_EXPRESSION_EXAMPLES,
    FLOW_TEMPLATE_EXPRESSION_RULES,
)
from crewai.flow.flow_definition import FlowDefinition


SKIP_BY_MODEL: dict[str, str] = {
    "FlowScriptActionDefinition": "script_action",
    "FlowToolActionDefinition": "tool_action",
    "FlowExpressionActionDefinition": "expression_action",
    "FlowEachActionDefinition": "each",
    "FlowEachStepDefinition": "each",
    "FlowConfigDefinition": "config",
    "FlowHumanFeedbackDefinition": "hitl",
    "FlowPersistenceDefinition": "persistence",
}

FIELD_TYPE_OVERRIDES: dict[tuple[str, str], str] = {
    ("FlowDefinition", "state"): "[State](#json-schema-state-statetypejson_schema)",
    ("FlowDefinition", "methods"): "map of string to [Method](#method-methods)",
    ("FlowMethodDefinition", "do"): "[Action](#action)",
    ("FlowCrewActionDefinition", "with"): "inline crew definition",
    ("CrewAgentDefinition", "llm"): "string or inline LLM config",
    ("AgentDefinition", "llm"): "string or inline LLM config",
}

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ENVIRONMENT = Environment(  # noqa: S701 - renders trusted Markdown, not HTML.
    loader=FileSystemLoader(_TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render_skill_markdown(
    *,
    skips: Sequence[str] = (),
    examples_format: Literal["yaml", "json"] = "yaml",
) -> str:
    if examples_format not in ("yaml", "json"):
        raise ValueError("Flow skill examples_format must be 'yaml' or 'json'")

    skips_set = frozenset(skips)
    rendered = _ENVIRONMENT.get_template("flow_definition_skill.md.j2").render(
        template_context(skips_set, examples_format)
    )
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip() + "\n"


def template_context(
    skips: frozenset[str], examples_format: Literal["yaml", "json"] = "yaml"
) -> dict[str, Any]:
    return {
        "examples_format": examples_format,
        "example": render_flow_example(examples_format),
        "example_language": examples_format,
        "include_each_action": "each" not in skips,
        "include_conversational": "conversational" not in skips,
        "include_hitl": "hitl" not in skips,
        "include_non_linear_flows": "non_linear_flows" not in skips,
        "include_persistence": "persistence" not in skips,
        "include_expression_action": "expression_action" not in skips,
        "include_script_action": "script_action" not in skips,
        "include_tool_action": "tool_action" not in skips,
        "expression_contract_examples": FLOW_TEMPLATE_EXPRESSION_EXAMPLES[
            examples_format
        ],
        "expression_contract_rules": FLOW_TEMPLATE_EXPRESSION_RULES,
        "sections": FlowSkillReferenceExtractor(skips=skips).extract(),
    }


def render_flow_example(examples_format: Literal["yaml", "json"]) -> str:
    example_yaml = (_TEMPLATES_DIR / "flow_definition_example.yaml").read_text(
        encoding="utf-8"
    )
    if examples_format == "json":
        return json.dumps(yaml.safe_load(example_yaml), indent=2)
    return example_yaml.rstrip()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    section: str
    address: str = ""
    label: str = ""
    hidden: bool = False
    examples: bool = False
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def display_title(self) -> str:
        return self.label or MODEL_TITLES.get(self.name, self.section)

    @property
    def display_label(self) -> str:
        if not self.address:
            return self.display_title
        return f"{self.display_title} (`{self.address}`)"


MODEL_TITLES = {
    "FlowDefinition": "Flow Definition",
    "FlowDictStateDefinition": "Dict State",
    "FlowPydanticStateDefinition": "Pydantic State",
    "FlowJsonSchemaStateDefinition": "JSON Schema State",
    "FlowUnknownStateDefinition": "Unknown State",
    "FlowMethodDefinition": "Method",
    "FlowCodeActionDefinition": "Code Action",
    "FlowScriptActionDefinition": "Script Action",
    "FlowToolActionDefinition": "Tool Action",
    "FlowCrewActionDefinition": "Crew Action",
    "FlowAgentActionDefinition": "Agent Action",
    "FlowExpressionActionDefinition": "Expression Action",
    "FlowEachActionDefinition": "Each Action",
    "FlowEachStepDefinition": "Each Step",
    "CrewDefinition": "Crew Definition",
    "CrewAgentDefinition": "Crew Agent Definition",
    "CrewTaskDefinition": "Crew Task Definition",
    "AgentDefinition": "Agent Definition",
    "LLMDefinition": "LLM Definition",
    "FlowConfigDefinition": "Config",
    "FlowPersistenceDefinition": "Persistence",
    "FlowHumanFeedbackDefinition": "Human Feedback",
}


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "FlowDefinition",
        "Flow Definition",
        descriptions={
            "schema": "Declarative Flow schema identifier and version. Include it explicitly in authored declarations.",
            "conversational": "Top-level conversational flow configuration, only when the flow supports chat.",
        },
    ),
    ModelSpec("FlowDictStateDefinition", "State", "state[type=dict]", hidden=True),
    ModelSpec(
        "FlowPydanticStateDefinition", "State", "state[type=pydantic]", hidden=True
    ),
    ModelSpec(
        "FlowJsonSchemaStateDefinition",
        "State",
        "state[type=json_schema]",
        descriptions={
            "json_schema": "JSON Schema used to validate and document flow state. Declare required fields with JSON Schema's `required` array.",
            "default": "Default values used to initialize Flow state. Defaults are not the same as schema-required fields.",
        },
    ),
    ModelSpec(
        "FlowUnknownStateDefinition", "State", "state[type=unknown]", hidden=True
    ),
    ModelSpec(
        "FlowMethodDefinition",
        "Method",
        "methods.<name>",
        descriptions={
            "do": "Single action object executed when this method runs.",
            "start": "Marks a start method. Use `true` for the normal entrypoint. String or map conditions are advanced trigger conditions; use them only when the user asks for event/condition-based starts.",
            "listen": 'Trigger condition that runs this method after upstream events. A string target can be a method name or a router-emitted event name, and both live in the same trigger namespace. Methods must not listen to their own method name. Map conditions are for `and`/`or` trigger composition, for example `{"and": ["validated", "processed"]}`.',
            "router": "Whether the method output should be treated as the next event name. Router actions must return one event name string, with no surrounding explanation.",
            "emit": "Declared router events this method may emit. Each emitted event name should be unique and should not collide with method names.",
        },
    ),
    ModelSpec(
        "FlowCodeActionDefinition",
        "Action",
        "methods.<name>.do[call=code]",
        hidden=True,
    ),
    ModelSpec("FlowScriptActionDefinition", "Action", "methods.<name>.do[call=script]"),
    ModelSpec(
        "FlowToolActionDefinition",
        "Action",
        "methods.<name>.do[call=tool]",
        descriptions={
            "with": "Tool input arguments. Insert Flow values with `${...}`.",
        },
    ),
    ModelSpec(
        "FlowCrewActionDefinition",
        "Action",
        "methods.<name>.do[call=crew]",
        examples=True,
        descriptions={
            "call": "Action discriminator. Use crew to run an inline Crew definition.",
            "inputs": "Runtime inputs passed to the Crew. Insert Flow values with `${...}` and reference each input as `{name}` in agent or task text.",
        },
    ),
    ModelSpec(
        "FlowAgentActionDefinition",
        "Action",
        "methods.<name>.do[call=agent]",
        examples=True,
        descriptions={
            "with": "Individual Agent definition to load and execute outside of a crew for this action. Put the agent input in `with.input`; agent actions do not support action-level `inputs`.",
        },
    ),
    ModelSpec(
        "FlowExpressionActionDefinition",
        "Action",
        "methods.<name>.do[call=expression]",
    ),
    ModelSpec("FlowEachActionDefinition", "Action", "methods.<name>.do[call=each]"),
    ModelSpec(
        "FlowEachStepDefinition",
        "Each Step",
        "methods.<name>.do[call=each].do[]",
    ),
    ModelSpec(
        "CrewDefinition",
        "Crew Definition",
        "methods.<name>.do[call=crew].with",
        hidden=True,
        examples=True,
        descriptions={
            "inputs": "Static default crew inputs. Values are available to crew agent and task interpolation as `{name}` placeholders, for example `{topic}`. Prefer action-level crew `inputs` for runtime values from `state` or `outputs`, and include placeholders for any inputs the crew must reason over.",
            "manager_agent": "Optional manager agent name.",
        },
    ),
    ModelSpec(
        "CrewAgentDefinition",
        "Crew Agent Definition",
        "methods.<name>.do[call=crew].with.agents.<name>",
        hidden=True,
        examples=True,
        descriptions={
            "llm": "Language model that runs this crew agent. Use an object when setting LLM options such as `max_tokens`.",
            "planning_config": "Agent planning configuration. Set `max_attempts` to limit planning refinement attempts before task execution.",
        },
    ),
    ModelSpec(
        "LLMDefinition",
        "LLM Definition",
        hidden=True,
        examples=True,
    ),
    ModelSpec(
        "CrewTaskDefinition",
        "Crew Task Definition",
        "methods.<name>.do[call=crew].with.tasks[]",
        hidden=True,
        examples=True,
        descriptions={
            "name": "Optional task name.",
        },
    ),
    ModelSpec(
        "AgentDefinition",
        "Agent Definition",
        "methods.<name>.do[call=agent].with",
        hidden=True,
        examples=True,
        descriptions={
            "input": "Agent prompt template. Insert Flow values with `${...}`, for example `Ticket: ${state.ticket_id}`.",
            "llm": "Language model that runs this agent. Use an object when setting LLM options such as `max_tokens`.",
            "planning_config": "Agent planning configuration. Set `max_attempts` to limit planning refinement attempts before task execution.",
        },
    ),
    ModelSpec("FlowConfigDefinition", "Config", "config"),
    ModelSpec("FlowPersistenceDefinition", "Persistence", "persist"),
    ModelSpec(
        "FlowHumanFeedbackDefinition",
        "Human Feedback",
        "methods.<name>.human_feedback",
    ),
)

_SPECS_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}


@dataclass(frozen=True)
class FlowSkillReferenceExtractor:
    skips: frozenset[str]
    schema: dict[str, Any] = field(
        default_factory=lambda: FlowDefinition.model_json_schema(by_alias=True)
    )

    def extract(self) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []

        for spec in MODEL_SPECS:
            if spec.hidden or self.model_is_skipped(spec.name):
                continue

            if not sections or sections[-1]["label"] != spec.section:
                sections.append(
                    {
                        "label": spec.section,
                        "models": [],
                        "kind": "union" if spec.section == "Action" else "object",
                    }
                )
            sections[-1]["models"].append(self.extract_model(spec))

        for section in sections:
            if section["kind"] != "union" and section["models"]:
                section["label"] = section["models"][0]["label"]

        return sections

    def extract_model(self, spec: ModelSpec) -> dict[str, Any]:
        model_name = spec.name
        model_schema = (
            self.schema
            if model_name == "FlowDefinition"
            else self.schema["$defs"][model_name]
        )
        required_from_schema = set(model_schema.get("required", ()))
        fields = []

        for field_name, field_schema in model_schema.get("properties", {}).items():
            if self.field_is_hidden(model_name, field_name):
                continue

            required = (
                field_name in required_from_schema
                or (
                    model_name == "FlowDefinition"
                    and field_name in ("state", "methods")
                )
                or (model_name == "FlowCrewActionDefinition" and field_name == "with")
            )
            fields.append(
                {
                    "name": field_name,
                    "type": self.render_field_type(
                        model_name, field_name, field_schema
                    ),
                    "required": required,
                    "default": render_field_default(
                        model_name, field_name, field_schema, required
                    ),
                    "description": self.render_field_description(
                        spec, model_name, field_name, field_schema
                    ),
                    "examples": render_field_examples(spec, field_name, field_schema),
                }
            )

        return {
            "label": spec.display_label,
            "anchor": f"#{markdown_heading_anchor(spec.display_label)}",
            "link": (
                f"[{spec.display_label}](#{markdown_heading_anchor(spec.display_label)})"
            ),
            "discriminator": extract_discriminator(model_schema),
            "fields": fields,
            "inline_models": self.inline_models_for(model_name),
        }

    def inline_models_for(self, model_name: str) -> list[dict[str, Any]]:
        names_by_model = {
            "FlowCrewActionDefinition": (
                "CrewDefinition",
                "CrewAgentDefinition",
                "CrewTaskDefinition",
            ),
            "CrewAgentDefinition": ("LLMDefinition",),
            "FlowAgentActionDefinition": ("AgentDefinition",),
            "AgentDefinition": ("LLMDefinition",),
        }
        return [
            self.extract_model(_SPECS_BY_NAME[name])
            for name in names_by_model.get(model_name, ())
        ]

    def model_is_skipped(self, model_name: str) -> bool:
        skip = SKIP_BY_MODEL.get(model_name)
        return skip in self.skips if skip is not None else False

    def field_is_hidden(
        self,
        model_name: str,
        field_name: str,
    ) -> bool:
        return (
            ("hitl" in self.skips and field_name == "human_feedback")
            or ("persistence" in self.skips and field_name == "persist")
            or ("config" in self.skips and field_name == "config")
            or ("conversational" in self.skips and field_name == "conversational")
            or (model_name == "AgentDefinition" and field_name == "response_format")
            or (model_name == "CrewDefinition" and field_name == "manager_agent")
            or (model_name == "CrewTaskDefinition" and field_name == "context")
            or (
                field_name == "type"
                and model_name
                in {"AgentDefinition", "CrewAgentDefinition", "CrewTaskDefinition"}
            )
            or (field_name == "ref" and model_name != "FlowToolActionDefinition")
            or (
                model_name == "FlowCrewActionDefinition"
                and field_name == "from_declaration"
            )
        )

    def render_field_type(
        self,
        model_name: str,
        field_name: str,
        field_schema: dict[str, Any],
    ) -> str:
        if override := FIELD_TYPE_OVERRIDES.get((model_name, field_name)):
            return override
        return self.render_schema_type(field_schema) or "any"

    def render_schema_type(self, field_schema: dict[str, Any]) -> str | None:
        if "$ref" in field_schema:
            return self.render_schema_ref(field_schema["$ref"])
        if "const" in field_schema:
            return f"must be {format_inline_value(field_schema['const'])}"
        if "enum" in field_schema:
            values = ", ".join(
                format_inline_value(value) for value in field_schema["enum"]
            )
            return f"one of {values}"

        for union_key in ("anyOf", "oneOf", "allOf"):
            if union_key in field_schema:
                return join_unique(
                    self.render_schema_type(option)
                    for option in field_schema[union_key]
                )

        json_type = field_schema.get("type")
        if isinstance(json_type, list):
            return join_unique(
                self.render_schema_type({"type": item}) for item in json_type
            )
        if json_type == "array":
            item_type = self.render_schema_type(field_schema.get("items", {})) or "any"
            return (
                f"list of {item_type}"
                if item_type.startswith("[")
                else f"list[{item_type}]"
            )
        if json_type == "object":
            additional_properties = field_schema.get("additionalProperties")
            if isinstance(additional_properties, dict):
                value_type = self.render_schema_type(additional_properties) or "any"
                return f"map of string to {value_type}"
            return "map of string to any" if additional_properties is True else "object"
        if isinstance(json_type, str):
            return json_type
        return "object" if "properties" in field_schema else "any"

    def render_schema_ref(self, ref: str) -> str | None:
        schema_name = ref.rsplit("/", 1)[-1]
        if schema_name == "ExpressionData":
            return (
                "expression data"
                if "expression_action" not in self.skips
                else "dynamic value"
            )
        if schema_name == "PythonReferenceDefinition":
            return None
        spec = _SPECS_BY_NAME.get(schema_name)
        if (spec and spec.hidden) or self.model_is_skipped(schema_name):
            return None
        if spec is None:
            return "object"
        return f"[{spec.display_label}](#{markdown_heading_anchor(spec.display_label)})"

    def render_field_description(
        self,
        spec: ModelSpec,
        model_name: str,
        field_name: str,
        field_schema: dict[str, Any],
    ) -> str | None:
        if "non_linear_flows" in self.skips and model_name == "FlowMethodDefinition":
            if field_name == "start":
                return "Marks the single normal entrypoint. Use `true`."
            if field_name == "listen":
                return "Runs this method after one upstream method or router-emitted event."
        return render_field_description(spec, field_name, field_schema)


def render_field_default(
    model_name: str,
    field_name: str,
    field_schema: dict[str, Any],
    required: bool,
) -> str | None:
    if required:
        return None
    if model_name == "FlowDefinition" and field_name == "config":
        return "generated default"
    if "default" in field_schema:
        return format_inline_value(field_schema["default"])
    return None


def extract_discriminator(model_schema: dict[str, Any]) -> dict[str, str] | None:
    properties = model_schema.get("properties", {})
    for field_name in ("call", "type"):
        if field_name not in properties:
            continue
        value = properties[field_name].get(
            "const", properties[field_name].get("default")
        )
        if value is not None:
            return {"name": field_name, "value": str(value)}
    return None


def join_unique(values: Any) -> str | None:
    rendered_values = list(
        dict.fromkeys(value for value in values if value is not None)
    )
    return " | ".join(rendered_values) or None


def markdown_heading_anchor(text: str) -> str:
    heading = re.sub(r"<[^>]+>", "", text)
    heading = re.sub(r"`([^`]*)`", r"\1", heading)
    heading = heading.lower()
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s+", "-", heading.strip())


def format_inline_value(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def render_field_description(
    spec: ModelSpec, field_name: str, field_schema: dict[str, Any]
) -> str | None:
    if field_name in spec.descriptions:
        return spec.descriptions[field_name]
    return field_schema.get("description")


def render_field_examples(
    spec: ModelSpec, field_name: str, field_schema: dict[str, Any]
) -> list[str]:
    if not spec.examples:
        return []

    examples = (
        example
        for example in field_schema.get("examples", ())
        if not contains_python_reference(example)
    )
    return [format_inline_example(example) for example in examples]


def contains_python_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return "python" in value or any(
            contains_python_reference(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_python_reference(item) for item in value)
    return False


def format_inline_example(value: Any) -> str:
    if isinstance(value, str):
        return format_inline_value(value.replace("\n", "\\n"))
    if value is None or isinstance(value, (bool, int, float)):
        return format_inline_value(value)
    return f"`{json.dumps(value, ensure_ascii=True)}`"
