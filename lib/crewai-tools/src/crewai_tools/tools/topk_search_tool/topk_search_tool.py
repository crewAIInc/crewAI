from __future__ import annotations

import importlib
import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


SearchMode = Literal["hybrid", "keyword", "semantic"]

_FILTER_OPERATORS = (
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$contains",
)


class TopKToolSchema(BaseModel):
    """Input for TopKSearchTool."""

    query: str = Field(..., description="What to search for.")
    search_mode: SearchMode = Field(
        default="hybrid",
        description=(
            "'keyword' (BM25) for exact strings that must appear in the document, such as "
            "error codes, IDs, SKUs or product names; it returns nothing if the terms don't "
            "appear. 'semantic' for paraphrased or conceptual questions where the documents "
            "may use different words. 'hybrid' ranks both together and is the best default."
        ),
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            'Optional field filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}. '
            "Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains."
        ),
    )


class TopKSearchTool(BaseTool):
    """Keyword (BM25), semantic and hybrid search over a TopK collection.

    The collection's text field needs a ``semantic_index()``, which makes TopK
    embed text on the server and also builds the BM25 index, so no embedding
    model or API key is needed on the client.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "TopKSearchTool"
    description: str = (
        "Search a TopK collection. Use search_mode='keyword' for exact terms such as "
        "error codes or IDs, 'semantic' for paraphrased questions, and 'hybrid' "
        "(the default) for everything else."
    )
    args_schema: type[BaseModel] = TopKToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["topk-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TOPK_API_KEY",
                description="TopK API key, from https://console.topk.io",
                required=True,
            ),
            EnvVar(
                name="TOPK_REGION",
                description="TopK region, e.g. aws-us-east-1-elastica",
                required=True,
            ),
        ]
    )

    collection_name: str = Field(description="TopK collection to search.")
    text_field: str = Field(
        default="text",
        description="Field with a semantic_index() that holds the document text.",
    )
    return_fields: list[str] = Field(
        default_factory=lambda: ["text"],
        description="Fields to return for each result.",
    )
    partition: str | None = Field(
        default=None,
        description="Search only this partition, e.g. one tenant's or user's data.",
    )
    limit: int = Field(default=5, ge=1, le=100, description="Results per search.")
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight of semantic similarity in hybrid search; BM25 gets 1 - alpha.",
    )
    api_key: str | None = Field(default=None, description="Defaults to TOPK_API_KEY.")
    region: str | None = Field(default=None, description="Defaults to TOPK_REGION.")
    client: Any | None = Field(
        default=None, description="An existing topk_sdk.Client to reuse."
    )

    _topk: Any = PrivateAttr(default=None)

    def _sdk(self) -> Any:
        if self._topk is None:
            try:
                self._topk = importlib.import_module("topk_sdk")
                importlib.import_module("topk_sdk.query")
            except ImportError as e:
                raise ImportError(
                    "TopKSearchTool needs the topk-sdk package: pip install topk-sdk"
                ) from e
        return self._topk

    def _collection(self) -> Any:
        sdk = self._sdk()
        if self.client is None:
            api_key = self.api_key or os.getenv("TOPK_API_KEY")
            region = self.region or os.getenv("TOPK_REGION")
            if not api_key or not region:
                raise ValueError(
                    "Set TOPK_API_KEY and TOPK_REGION, or pass api_key and region."
                )
            self.client = sdk.Client(api_key=api_key, region=region)
        return self.client.collection(self.collection_name, partition=self.partition)

    def _run(
        self,
        query: str,
        search_mode: SearchMode = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> str:
        q = self._sdk().query
        dense = q.fn.semantic_similarity(self.text_field, query)
        if search_mode == "keyword":
            scores = {"topk_score": q.fn.bm25_score()}
            text_filter = q.match(query, field=self.text_field)
        elif search_mode == "semantic":
            scores = {"topk_score": dense}
            text_filter = None
        elif search_mode == "hybrid":
            scores = {"topk_dense": dense, "topk_keyword": q.fn.bm25_score()}
            # should() boosts keyword matches without dropping documents that have none.
            text_filter = q.should(query, field=self.text_field)
        else:
            raise ValueError(
                f"Unknown search_mode {search_mode!r}; use hybrid, keyword or semantic."
            )

        query_builder = q.select(*self.return_fields, **scores)
        if text_filter is not None:
            query_builder = query_builder.filter(text_filter)
        field_filter = self._translate_filters(filters)
        if field_filter is not None:
            query_builder = query_builder.filter(field_filter)
        if search_mode == "hybrid":
            rank = q.field("topk_dense") * self.hybrid_alpha + q.field(
                "topk_keyword"
            ) * (1.0 - self.hybrid_alpha)
        else:
            rank = q.field("topk_score")
        try:
            rows = self._collection().query(
                query_builder.sort(rank, asc=False).limit(self.limit)
            )
        except Exception as e:
            # A partition that was never written has no documents yet.
            if type(e).__name__ == "PartitionNotFoundError":
                return "[]"
            raise

        results = []
        for row in rows:
            if search_mode == "hybrid":
                score = (row.get("topk_dense") or 0.0) * self.hybrid_alpha + (
                    row.get("topk_keyword") or 0.0
                ) * (1.0 - self.hybrid_alpha)
            else:
                score = row.get("topk_score") or 0.0
            result = {"id": row["_id"], "score": round(float(score), 4)}
            result.update({f: row[f] for f in self.return_fields if f in row})
            results.append(result)
        return json.dumps(results, indent=2, default=str)

    def _translate_filters(self, filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        q = self._sdk().query
        clauses = []
        for key, value in filters.items():
            f = q.field(key)
            ops = value if isinstance(value, dict) else {"$eq": value}
            for op, operand in ops.items():
                if op not in _FILTER_OPERATORS:
                    raise ValueError(
                        f"Unknown filter operator {op!r}; use one of {', '.join(_FILTER_OPERATORS)}."
                    )
                if op in ("$in", "$nin"):
                    clause = f.in_(list(operand))
                    clauses.append(q.not_(clause) if op == "$nin" else clause)
                else:
                    method = {
                        "$eq": "eq",
                        "$ne": "ne",
                        "$gt": "gt",
                        "$gte": "gte",
                        "$lt": "lt",
                        "$lte": "lte",
                        "$contains": "contains",
                    }[op]
                    clauses.append(getattr(f, method)(operand))
        return clauses[0] if len(clauses) == 1 else q.all(clauses)
