from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool

if TYPE_CHECKING:
    from supabase import Client


class SupabaseTool(BaseTool):
    name: str = "SupabaseTool"
    description: str = (
        "A tool for performing Supabase database operations such as "
        "select, insert, update, and delete."
    )

    def __init__(self) -> None:
        super().__init__()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
            )
        if not url.startswith("https://"):
            raise ValueError("SUPABASE_URL must start with 'https://'")

        from supabase import create_client

        self.client: Client = create_client(url, key)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool and return its operation result.

        Args:
            *args: Positional arguments passed to the tool.
            **kwargs: Keyword arguments passed to the tool.

        Returns:
            A normalized JSON-like operation response.

        Raises:
            ValueError: If the tool configuration or operation input is invalid.
            TypeError: If filters is not a dictionary.
        """
        return super().run(*args, **kwargs)

    def _run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an operation described by ``params``.

        Args:
            params: Operation name, table name, optional filters, and data.

        Returns:
            A normalized response containing ``data`` and ``error`` keys, or
            an ``error`` key for invalid operation input.

        Raises:
            ValueError: If ``params`` is not a dictionary or filters use an
                unsupported operator format.
            TypeError: If filters is not a dictionary.
        """
        if not isinstance(params, dict):
            raise ValueError("SupabaseTool parameters must be a dictionary")

        action = params.get("action")
        table = params.get("table")
        if not action or not table:
            return {"data": None, "error": "Missing required fields: action, table"}

        if action == "select":
            return self.select(table, params.get("filters"))
        if action == "insert":
            return self.insert(table, params.get("data"))
        if action == "update":
            return self.update(table, params.get("data"), params.get("filters"))
        if action == "delete":
            return self.delete(table, params.get("filters"))
        return {"data": None, "error": f"Unknown action: {action}"}

    def _apply_filters(self, operation: Any, filters: Any) -> Any:
        """Apply validated filters to a Supabase operation."""
        if filters is None or filters == {}:
            return operation
        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary")

        operators = {
            "eq": "eq",
            "neq": "neq",
            "gt": "gt",
            "gte": "gte",
            "lt": "lt",
            "lte": "lte",
        }
        for column, value in filters.items():
            operator = "eq"
            filter_value = value
            if isinstance(value, dict):
                operator = value.get("operator")
                if operator not in operators or "value" not in value:
                    raise ValueError(
                        "Filter descriptors must contain a supported operator "
                        "and a value"
                    )
                filter_value = value["value"]
            elif isinstance(value, str) and value.startswith(
                ("eq.", "neq.", "gt.", "gte.", "lt.", "lte.")
            ):
                raise ValueError(
                    "Filter values must be direct values or descriptors like "
                    "{'operator': 'neq', 'value': 1}; do not use 'eq.1'"
                )
            operation = getattr(operation, operators[operator])(column, filter_value)
        return operation

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        """Convert a Supabase response into a JSON-like dictionary."""
        if isinstance(response, dict):
            return response
        return {
            "data": getattr(response, "data", None),
            "error": getattr(response, "error", None),
        }

    def select(self, table: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Select rows from a table.

        Args:
            table: Supabase table name.
            filters: Optional column-to-value equality filters.

        Returns:
            A normalized response containing selected rows and any error.

        Raises:
            TypeError: If filters is not a dictionary.
        """
        operation = self.client.table(table).select("*")
        operation = self._apply_filters(operation, filters)
        return self._normalize_response(operation.execute())

    def insert(self, table: str, data: Any) -> dict[str, Any]:
        """Insert data into a table.

        Args:
            table: Supabase table name.
            data: Row dictionary or list of row dictionaries to insert.

        Returns:
            A normalized response containing inserted rows and any error.

        Raises:
            ValueError: If data is missing.
        """
        if data is None:
            return {"data": None, "error": "Missing data for insert"}
        return self._normalize_response(self.client.table(table).insert(data).execute())

    def update(
        self,
        table: str,
        data: Any,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update rows in a table.

        Args:
            table: Supabase table name.
            data: Column values to update.
            filters: Optional column-to-value equality filters.

        Returns:
            A normalized response containing updated rows and any error.

        Raises:
            TypeError: If filters is not a dictionary.
            ValueError: If data is missing.
        """
        if data is None:
            return {"data": None, "error": "Missing data for update"}
        operation = self.client.table(table).update(data)
        operation = self._apply_filters(operation, filters)
        return self._normalize_response(operation.execute())

    def delete(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delete rows from a table.

        Args:
            table: Supabase table name.
            filters: Optional column-to-value equality filters.

        Returns:
            A normalized response containing deleted rows and any error.

        Raises:
            TypeError: If filters is not a dictionary.
        """
        operation = self.client.table(table).delete()
        operation = self._apply_filters(operation, filters)
        return self._normalize_response(operation.execute())
