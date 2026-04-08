from typing import Any, Type, Union, List, Dict
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import json

try:
    from sqlalchemy import create_engine, text, inspect
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
    description: str = (
        "Useful for querying a database. Input should be a raw SQL query. "
        "The database dialect is {dialect}. IMPORTANT: Do not query system tables "
        "like 'information_schema'. Use ONLY the tables and columns provided in the tool context."
    )
    

    db_uri: str = Field(
        title="Database URI",
        description="The URI of the database to connect to.",
    )
    # Use PrivateAttr for internal state so Pydantic doesn't validate them
    _tables: List[Dict] = PrivateAttr(default=[])
    _columns: Dict = PrivateAttr(default={})
    _dialect: str = PrivateAttr(default="SQL")

    args_schema: Type[BaseModel] = NL2SQLToolInput

    def model_post_init(self, __context: Any) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`")

        # Start a single engine for the initialization process
        engine = create_engine(self.db_uri)
        try:
            self._dialect = engine.dialect.name.upper()
            
            # Dynamically update the description so the Agent knows the flavor
            self.description = (
            "Useful for querying a database. Input should be a raw SQL query. "
            f"The database dialect is {self._dialect}. "
            "IMPORTANT: Do not query system tables like 'information_schema'. "
            "Use ONLY the tables and columns provided in the tool context.")
            
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            self._tables = [{"table_name": name} for name in table_names]
            
            col_data = {}
            for t_name in table_names:
                columns = inspector.get_columns(t_name)
                col_data[f'{t_name}_columns'] = [
                    {"column_name": col["name"], "data_type": str(col["type"])}
                    for col in columns
                ]
            
            self._columns = col_data
        except Exception as e:
            # Log the error but don't necessarily crash the whole app
            print(f"Error mapping schema: {e}")
        finally:
            engine.dispose()
    
    def _run(self, sql_query: str) -> str:
        try:
            result = self.execute_sql(sql_query)
            return json.dumps(result, default=str) if isinstance(result, list) else str(result)
        except Exception as exc:
            table_names = [t['table_name'] for t in self._tables]
            return (
            f"DIALECT ERROR: The query failed. You are querying a {self._dialect} database.\n"
            f"ERROR DETAIL: {exc}\n"
            f"AVAILABLE SCHEMA: Use ONLY these tables: {table_names}.\n"
            f"COLUMN DETAILS: {json.dumps(self._columns, indent=2)}\n"
            "DO NOT query 'information_schema'. Use the provided schema above to rewrite your query."
            )

    def execute_sql(self, sql_query: str) -> Union[list, str]:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is not installed. Please install it with `pip install crewai-tools[sqlalchemy]`")

        engine = create_engine(self.db_uri)
        Session = sessionmaker(bind=engine)
        
        try:
            with Session() as session:
                try:
                    result = session.execute(text(sql_query))
                    
                    if result.returns_rows:
                        columns = list(result.keys())
                        data = [dict(zip(columns, row)) for row in result.fetchall()]
                        return data
                    else:
                        session.commit()
                        return f"Query {sql_query} executed successfully."
                except Exception as e:
                    session.rollback()
                    raise e
        finally:
            # Dispose the engine AFTER the session context is completely finished
            engine.dispose()
