from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class NL2SQLToolInput(BaseModel):
    sql_query: str = Field(
        title="SQL Query",
        description="The SQL query to execute.",
    )


class NL2SQLTool(BaseTool):
    name: str = "NL2SQLTool"
    description: str = "Converts natural language to SQL queries and executes them."
    db_uri: str = Field(
        title="Database URI",
        description="The URI of the database to connect to.",
    )
    tables: list = Field(default_factory=list)
    columns: dict = Field(default_factory=dict)
    args_schema: type[BaseModel] = NL2SQLToolInput

    def model_post_init(self, __context: Any) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`"
            )

        data = {}
        tables = self._fetch_available_tables()

        for table in tables:
            table_columns = self._fetch_all_available_columns(table["table_name"])
            data[f"{table['table_name']}_columns"] = table_columns

        self.tables = tables
        self.columns = data

    def _fetch_available_tables(self):
        return self.execute_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )

    def _fetch_all_available_columns(self, table_name: str):
        return self.execute_sql(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"  # noqa: S608
        )

    def _run(self, sql_query: str):
        try:
            data = self.execute_sql(sql_query)
        except Exception as exc:
            data = (
                f"Based on these tables {self.tables} and columns {self.columns}, "
                "you can create SQL queries to retrieve data from the database."
                f"Get the original request {sql_query} and the error {exc} and create the correct SQL query."
            )

        return data

    def execute_sql(self, sql_query: str) -> list | str:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`"
            )

        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)  # noqa: N806
        session = Session()
        try:
            result = session.execute(text(sql_query))
            session.commit()

            if result.returns_rows:  # type: ignore[attr-defined]
                columns = result.keys()
                return [
                    dict(zip(columns, row, strict=False)) for row in result.fetchall()
                ]
            return f"Query {sql_query} executed successfully"

        except Exception as e:
            session.rollback()
            raise e

        finally:
            session.close()
