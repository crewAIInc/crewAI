from typing import Dict, Optional, Type, TYPE_CHECKING

from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel

if TYPE_CHECKING:
    from minds_sdk import Client


class AIMindInputSchema(BaseModel):
    """Input for AIMind Tool."""

    query: str = "Question in natural language to ask the AI-Mind"


class AIMindTool(BaseTool):
    name: str = "AIMind Tool"
    description: str = (
        "A wrapper around [AI-Minds](https://mindsdb.com/minds). "
        "Useful for when you need answers to questions from your data, stored in "
        "data sources including PostgreSQL, MySQL, MariaDB, ClickHouse, Snowflake "
        "and Google BigQuery. "
        "Input should be a question in natural language."
    )
    args_schema: Type[BaseModel] = AIMindInputSchema
    api_key: Optional[str] = None
    datasources: Optional[Dict] = None
    minds_client: Optional["Client"] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from minds_sdk import Client  # type: ignore
        except ImportError:
            raise ImportError(
                "`minds_sdk` package not found, please run `pip install minds-sdk`"
            )

        self.minds_client = Client(api_key=api_key)