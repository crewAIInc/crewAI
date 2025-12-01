"""MySQL database loader."""

from typing import Any
from urllib.parse import urlparse

from pymysql import Error, connect
from pymysql.cursors import DictCursor

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class MySQLLoader(BaseLoader):
    """Loader for MySQL database content."""

    def load(self, source: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load content from a MySQL database table.

        Args:
            source: SQL query (e.g., "SELECT * FROM table_name")
            **kwargs: Additional arguments including db_uri

        Returns:
            LoaderResult with database content
        """
        metadata = kwargs.get("metadata", {})
        db_uri = metadata.get("db_uri")

        if not db_uri:
            raise ValueError("Database URI is required for MySQL loader")

        query = source.source

        parsed = urlparse(db_uri)
        if parsed.scheme not in ["mysql", "mysql+pymysql"]:
            raise ValueError(f"Invalid MySQL URI scheme: {parsed.scheme}")

        connection_params = {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/") if parsed.path else None,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
        }

        if not connection_params["database"]:
            raise ValueError("Database name is required in the URI")

        try:
            connection = connect(**connection_params)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if not rows:
                        content = "No data found in the table"
                        return LoaderResult(
                            content=content,
                            metadata={"source": query, "row_count": 0},
                            doc_id=self.generate_doc_id(
                                source_ref=query, content=content
                            ),
                        )

                    text_parts = []

                    columns = list(rows[0].keys())
                    text_parts.append(f"Columns: {', '.join(columns)}")
                    text_parts.append(f"Total rows: {len(rows)}")
                    text_parts.append("")

                    for i, row in enumerate(rows, 1):
                        text_parts.append(f"Row {i}:")
                        for col, val in row.items():
                            if val is not None:
                                text_parts.append(f"  {col}: {val}")
                        text_parts.append("")

                    content = "\n".join(text_parts)

                    if len(content) > 100000:
                        content = content[:100000] + "\n\n[Content truncated...]"

                    return LoaderResult(
                        content=content,
                        metadata={
                            "source": query,
                            "database": connection_params["database"],
                            "row_count": len(rows),
                            "columns": columns,
                        },
                        doc_id=self.generate_doc_id(source_ref=query, content=content),
                    )
            finally:
                connection.close()
        except Error as e:
            raise ValueError(f"MySQL database error: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load data from MySQL: {e}") from e
