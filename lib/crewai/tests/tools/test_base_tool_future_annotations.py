from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Optional

from annotated_types import Gt
from pydantic import BaseModel, Field
import pytest

from crewai.tools import BaseTool, tool
from crewai.tools.base_tool import Tool

# BaseTool._validate_kwargs hands the validated arguments to the callable as a
# dumped dict, so the tool bodies below index into ``args``.


class WalletArgs(BaseModel):
    amount: int


@tool("wallet")
def wallet(args: WalletArgs, path: Path | None = None) -> str:
    """Spend from the wallet."""
    return f"spent {args['amount']} at {path}"


@tool("selector")
def selector(
    mode: Literal["fast", "slow"] = "fast",
    label: Optional[str] = None,
) -> str:
    """Pick a mode."""
    return f"{mode}:{label}"


class WalletTool(BaseTool):
    name: str = "wallet_tool"
    description: str = "Spend from the wallet."

    def _run(self, args: WalletArgs) -> str:
        return f"spent {args['amount']}"


def legacy_wallet(args: WalletArgs) -> str:
    """Spend from the wallet."""
    return f"spent {args['amount']}"


def orphan(value: MissingAtRuntime) -> str:  # noqa: F821
    """Annotated with a name that only exists under TYPE_CHECKING."""
    return str(value)


@tool("bounded")
def bounded(x: Annotated[int, Field(gt=10)]) -> str:
    """Reject values that are not greater than ten."""
    return f"got {x}"


class BoundedTool(BaseTool):
    name: str = "bounded_tool"
    description: str = "Reject values that are not greater than ten."

    def _run(self, x: Annotated[int, Field(gt=10)]) -> str:
        return f"got {x}"


class TestToolDecoratorWithFutureAnnotations:
    def test_user_declared_model_argument_validates_at_call_time(self):
        assert wallet.run(args={"amount": 5}) == "spent 5 at None"

    def test_optional_and_literal_annotations_are_resolved(self):
        assert selector.run(mode="slow", label="x") == "slow:x"
        fields = selector.args_schema.model_fields
        assert fields["mode"].annotation == Literal["fast", "slow"]
        assert fields["label"].annotation == Optional[str]

    def test_user_declared_model_annotation_is_resolved(self):
        assert wallet.args_schema.model_fields["args"].annotation is WalletArgs
        assert wallet.args_schema.model_fields["path"].annotation == Optional[Path]

    def test_argument_schema_renders_without_forward_refs(self):
        properties = wallet.args_schema.model_json_schema()["properties"]
        assert properties["args"] == {"$ref": "#/$defs/WalletArgs"}

    def test_formatted_description_renders_without_forward_refs(self):
        assert "WalletArgs" in wallet.formatted_description


class TestBaseToolSubclassWithFutureAnnotations:
    def test_user_declared_model_argument_validates_at_call_time(self):
        assert WalletTool().run(args={"amount": 3}) == "spent 3"

    def test_lazily_derived_schema_resolves_user_declared_model(self):
        instance = WalletTool()
        instance.args_schema = None  # type: ignore[assignment]
        instance._set_args_schema()
        assert instance.args_schema.model_fields["args"].annotation is WalletArgs

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            WalletTool().run(args={"amount": "not-a-number"})


class TestFromLangchainWithFutureAnnotations:
    def test_tool_from_langchain_resolves_user_declared_model(self):
        wrapper = SimpleNamespace(
            name="legacy_wallet",
            description="Spend from the wallet.",
            func=legacy_wallet,
        )
        converted = Tool.from_langchain(wrapper)
        assert converted.run(args={"amount": 7}) == "spent 7"


class TestAnnotationMissingAtRuntime:
    def test_construction_still_defers_resolution_instead_of_raising(self):
        deferred = tool("deferred")(orphan)

        assert "value" in deferred.args_schema.model_fields


class TestAnnotatedMetadataWithFutureAnnotations:
    def test_distinguishing_metadata_survives_annotation_resolution(self):
        assert bounded.args_schema.model_fields["x"].metadata == [Gt(gt=10)]

    def test_constraint_reaches_the_argument_schema(self):
        assert bounded.args_schema.model_json_schema()["properties"]["x"] == {
            "exclusiveMinimum": 10,
            "title": "X",
            "type": "integer",
        }

    def test_decorated_tool_rejects_a_violating_value(self):
        with pytest.raises(ValueError, match="Input should be greater than 10"):
            bounded.run(x=5)

    def test_base_tool_subclass_rejects_a_violating_value(self):
        with pytest.raises(ValueError, match="Input should be greater than 10"):
            BoundedTool().run(x=5)
