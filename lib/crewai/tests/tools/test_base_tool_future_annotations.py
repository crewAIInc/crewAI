from __future__ import annotations

from functools import partial
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    no_type_check,
)

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


def tracked_wallet(args: WalletArgs, note: str = "wire") -> str:
    """Spend from the wallet with an audit note."""
    return f"spent {args['amount']} with {note}"


class LedgerEntry(BaseModel):
    amount: int


@no_type_check
def ledger(entry: LedgerEntry, note: str = "wire") -> str:
    """Record a ledger entry with an audit note."""
    return f"recorded {entry['amount']} with {note}"


if TYPE_CHECKING:
    class SpendReport(BaseModel):
        total: int


def audited(entry: LedgerEntry) -> SpendReport:  # noqa: F821
    """Record a ledger entry, annotated with a TYPE_CHECKING-only report."""
    return f"recorded {entry['amount']}"


@no_type_check
def audited_ledger(entry: LedgerEntry) -> SpendReport:  # noqa: F821
    """Record a ledger entry, annotated with a TYPE_CHECKING-only report."""
    return f"recorded {entry['amount']}"


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


class AuditedTool(BaseTool):
    name: str = "audited_tool"
    description: str = "Record a ledger entry, annotated with a TYPE_CHECKING-only report."

    def _run(self, entry: LedgerEntry) -> SpendReport:  # noqa: F821
        return f"recorded {entry['amount']}"


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


class _PartialConversionTool(BaseTool):
    """Minimal concrete tool so the ``BaseTool.from_langchain`` path can run."""

    name: str = "partial_conversion_tool"
    description: str = "Convert a partial into a BaseTool."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return tracked_wallet(**kwargs)


class TestFromLangchainPartialWithFutureAnnotations:
    """A ``functools.partial`` over a PEP 563 function keeps raw string
    annotations in its signature, and ``get_type_hints`` cannot evaluate the
    adapter itself; both ``from_langchain`` paths must resolve the remaining
    parameters against the underlying function's module instead of handing
    pydantic forward references it cannot resolve.
    """

    def _make_wrapper(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="partial_wallet",
            description="Spend from the wallet with a note.",
            func=partial(tracked_wallet, note="ach"),
        )

    def test_base_tool_from_langchain_resolves_partial_annotations(self):
        converted = _PartialConversionTool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["args"].annotation is WalletArgs
        assert converted.args_schema.model_json_schema()["properties"]["args"] == {
            "$ref": "#/$defs/WalletArgs"
        }

    def test_tool_from_langchain_resolves_partial_annotations(self):
        converted = Tool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["args"].annotation is WalletArgs
        assert converted.run(args={"amount": 7}) == "spent 7 with ach"

    def test_tool_from_langchain_partial_validates_remaining_arguments(self):
        converted = Tool.from_langchain(self._make_wrapper())
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            converted.run(args={"amount": "not-a-number"})


class TestAnnotationMissingAtRuntime:
    def test_construction_still_defers_resolution_instead_of_raising(self):
        deferred = tool("deferred")(orphan)

        assert "value" in deferred.args_schema.model_fields


class _LedgerConversionTool(BaseTool):
    """Minimal concrete tool so the ``BaseTool.from_langchain`` path can run."""

    name: str = "ledger_conversion_tool"
    description: str = "Convert a callable into a BaseTool."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return ledger(**kwargs)


class TestNoTypeCheckWithFutureAnnotations:
    """``@typing.no_type_check`` makes ``get_type_hints`` return an empty dict
    while the callable keeps its raw PEP 563 string annotations; the schema
    paths must evaluate them in this module instead of handing pydantic
    forward references it cannot resolve.
    """

    def _make_wrapper(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="ledger",
            description="Record a ledger entry with an audit note.",
            func=ledger,
        )

    def test_tool_decorator_resolves_no_type_check_annotations(self):
        converted = tool("ledger")(ledger)
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.args_schema.model_fields["note"].annotation is str
        assert converted.run(entry={"amount": 4}) == "recorded 4 with wire"
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            converted.run(entry={"amount": "not-a-number"})

    def test_base_tool_from_langchain_resolves_no_type_check_annotations(self):
        converted = _LedgerConversionTool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.args_schema.model_fields["note"].annotation is str
        assert converted.args_schema.model_json_schema()["properties"]["entry"] == {
            "$ref": "#/$defs/LedgerEntry"
        }
        with pytest.raises(ValueError, match="arguments validation failed"):
            converted.run(entry={"amount": "not-a-number"})

    def test_tool_from_langchain_resolves_no_type_check_annotations(self):
        converted = Tool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.args_schema.model_fields["note"].annotation is str
        assert converted.run(entry={"amount": 7}) == "recorded 7 with wire"


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


class _AuditedLedgerConversionTool(BaseTool):
    """Minimal concrete tool so the ``BaseTool.from_langchain`` path can run."""

    name: str = "audited_ledger_conversion_tool"
    description: str = "Convert a callable into a BaseTool."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return audited_ledger(**kwargs)


class TestUnresolvableReturnWithFutureAnnotations:
    """A return annotation visible only under TYPE_CHECKING makes
    ``get_type_hints`` fail for the whole callable; the resolvable
    parameters must still be evaluated in their defining module instead of
    degrading to raw PEP 563 strings that pydantic cannot resolve.
    """

    def test_tool_decorator_resolves_parameter_annotations(self):
        converted = tool("audited")(audited)
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry

    def test_decorated_tool_validates_arguments(self):
        converted = tool("audited")(audited)
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            converted.run(entry={"amount": "not-a-number"})

    def test_decorated_tool_renders_json_schema(self):
        converted = tool("audited")(audited)
        assert converted.args_schema.model_json_schema()["properties"]["entry"] == {
            "$ref": "#/$defs/LedgerEntry"
        }

    def test_base_tool_subclass_resolves_parameter_annotations(self):
        instance = AuditedTool()
        assert instance.args_schema.model_fields["entry"].annotation is LedgerEntry

    def test_base_tool_subclass_rejects_invalid_arguments(self):
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            AuditedTool().run(entry={"amount": "not-a-number"})

    def test_tool_from_langchain_resolves_and_runs(self):
        wrapper = SimpleNamespace(
            name="audited",
            description="Record a ledger entry, annotated with a TYPE_CHECKING-only report.",
            func=audited,
        )
        converted = Tool.from_langchain(wrapper)
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.run(entry={"amount": 9}) == "recorded 9"

    def test_partial_resolves_remaining_parameters(self):
        wrapper = SimpleNamespace(
            name="partial_audited",
            description="Record a ledger entry, annotated with a TYPE_CHECKING-only report.",
            func=partial(audited),
        )
        converted = Tool.from_langchain(wrapper)
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry


class TestNoTypeCheckUnresolvableReturnWithFutureAnnotations:
    """The same TYPE_CHECKING-only return type on a ``@no_type_check``
    callable must not defeat evaluation of the resolvable parameters either;
    each annotation is resolved on its own.
    """

    def _make_wrapper(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="audited_ledger",
            description="Record a ledger entry, annotated with a TYPE_CHECKING-only report.",
            func=audited_ledger,
        )

    def test_tool_decorator_resolves_parameter_annotations(self):
        converted = tool("audited_ledger")(audited_ledger)
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry

    def test_decorated_tool_validates_arguments(self):
        converted = tool("audited_ledger")(audited_ledger)
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            converted.run(entry={"amount": "not-a-number"})

    def test_base_tool_from_langchain_resolves_parameter_annotations(self):
        converted = _AuditedLedgerConversionTool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.args_schema.model_json_schema()["properties"]["entry"] == {
            "$ref": "#/$defs/LedgerEntry"
        }

    def test_tool_from_langchain_resolves_and_runs(self):
        converted = Tool.from_langchain(self._make_wrapper())
        assert converted.args_schema.model_fields["entry"].annotation is LedgerEntry
        assert converted.run(entry={"amount": 8}) == "recorded 8"


_ORIGIN_MODULE_SOURCE = '''\
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel


class Parcel(BaseModel):
    tracking: str


if TYPE_CHECKING:
    class DeliveryNote(BaseModel):
        signed: bool


def deliver(parcel: Parcel) -> str:
    """Deliver a parcel."""
    return f"delivered {parcel['tracking']}"


def deliver_reported(parcel: Parcel) -> DeliveryNote:  # noqa: F821
    """Deliver a parcel and return a TYPE_CHECKING-only note."""
    return f"delivered {parcel['tracking']}"
'''

_DECORATOR_MODULE_SOURCE = '''\
from functools import wraps


def audited(fn):
    """Wrap ``fn`` so the wrapper's globals live in this module alone."""

    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner
'''


class TestCrossModuleWrapsWithFutureAnnotations:
    """A ``functools.wraps`` decorator defined in another module copies the
    original annotations onto the wrapper but leaves ``__globals__`` pointed
    at the decorator's own module. ``get_type_hints`` follows ``__wrapped__``
    itself, so only its failures — one annotation unresolvable at runtime,
    typically a TYPE_CHECKING-only return type — reach the per-annotation
    fallback, which must resolve the surviving parameters in the original
    function's module instead of degrading them to strings pydantic cannot
    resolve.
    """

    @pytest.fixture
    def cross_module(self, tmp_path, monkeypatch):
        (tmp_path / "wrap_origin_module.py").write_text(_ORIGIN_MODULE_SOURCE)
        (tmp_path / "wrap_decorator_module.py").write_text(_DECORATOR_MODULE_SOURCE)
        monkeypatch.syspath_prepend(tmp_path)
        for name in ("wrap_origin_module", "wrap_decorator_module"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        origin = importlib.import_module("wrap_origin_module")
        decorators = importlib.import_module("wrap_decorator_module")
        return SimpleNamespace(
            deliver=decorators.audited(origin.deliver),
            deliver_reported=decorators.audited(origin.deliver_reported),
            Parcel=origin.Parcel,
        )

    def test_tool_from_langchain_resolves_wrapped_annotations(self, cross_module):
        converted = Tool.from_langchain(
            SimpleNamespace(
                name="deliver",
                description="Deliver a parcel.",
                func=cross_module.deliver,
            )
        )
        parcel = converted.args_schema.model_fields["parcel"]
        assert parcel.annotation is cross_module.Parcel
        assert converted.run(parcel={"tracking": "TRK-1"}) == "delivered TRK-1"

    def test_wrapped_tool_with_type_checking_only_return_stays_callable(
        self, cross_module
    ):
        converted = Tool.from_langchain(
            SimpleNamespace(
                name="deliver_reported",
                description="Deliver a parcel and return a TYPE_CHECKING-only note.",
                func=cross_module.deliver_reported,
            )
        )
        parcel = converted.args_schema.model_fields["parcel"]
        assert parcel.annotation is cross_module.Parcel
        assert converted.args_schema.model_json_schema()["properties"]["parcel"] == {
            "$ref": "#/$defs/Parcel"
        }
        assert converted.run(parcel={"tracking": "TRK-2"}) == "delivered TRK-2"

    def test_tool_decorator_resolves_wrapped_annotations(self, cross_module):
        converted = tool("deliver_reported")(cross_module.deliver_reported)
        parcel = converted.args_schema.model_fields["parcel"]
        assert parcel.annotation is cross_module.Parcel
        with pytest.raises(ValueError, match="Input should be a valid string"):
            converted.run(parcel={"tracking": 3})
