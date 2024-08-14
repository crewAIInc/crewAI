from typing import Any, Union

from ..base_tool import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from typing import Type, Any

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
    tables: list = []
    columns: dict = {}
    args_schema: Type[BaseModel] = NL2SQLToolInput

    def model_post_init(self, __context: Any) -> None:
        data = {}
        tables = self._fetch_available_tables()

        for table in tables:
            table_columns = self._fetch_all_available_columns(table["table_name"])
            data[f'{table["table_name"]}_columns'] = table_columns

        self.tables = tables
        self.columns = data

    def _fetch_available_tables(self):
        return self.execute_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )

    def _fetch_all_available_columns(self, table_name: str):
        return self.execute_sql(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
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

    def execute_sql(self, sql_query: str) -> Union[list, str]:
        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            result = session.execute(text(sql_query))
            session.commit()

            if result.returns_rows:
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                return data
            else:
                return f"Query {sql_query} executed successfully"

        except Exception as e:
            session.rollback()
            raise e

        finally:
            session.close()
