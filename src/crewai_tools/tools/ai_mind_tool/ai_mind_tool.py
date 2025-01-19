import os
import secrets
from typing import Any, Dict, List, Optional, Text, Type

from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel


class AIMindToolConstants:
    MINDS_API_BASE_URL = "https://mdb.ai/"
    MIND_NAME_PREFIX = "crwai_mind_"
    DATASOURCE_NAME_PREFIX = "crwai_ds_"


class AIMindToolInputSchema(BaseModel):
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
    args_schema: Type[BaseModel] = AIMindToolInputSchema
    api_key: Optional[str] = None
    datasources: Optional[List[Dict[str, Any]]] = None
    mind_name: Optional[Text] = None

    def __init__(self, api_key: Optional[Text] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("MINDS_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either through constructor or MINDS_API_KEY environment variable")

        try:
            from minds.client import Client  # type: ignore
            from minds.datasources import DatabaseConfig  # type: ignore
        except ImportError:
            raise ImportError(
                "`minds_sdk` package not found, please run `pip install minds-sdk`"
            )

        minds_client = Client(api_key=self.api_key)

        # Convert the datasources to DatabaseConfig objects.
        datasources = []
        for datasource in self.datasources:
            config = DatabaseConfig(
                name=f"{AIMindToolConstants.DATASOURCE_NAME_PREFIX}_{secrets.token_hex(5)}",
                engine=datasource["engine"],
                description=datasource["description"],
                connection_data=datasource["connection_data"],
                tables=datasource["tables"],
            )
            datasources.append(config)

        # Generate a random name for the Mind.
        name = f"{AIMindToolConstants.MIND_NAME_PREFIX}_{secrets.token_hex(5)}"

        mind = minds_client.minds.create(
            name=name, datasources=datasources, replace=True
        )

        self.mind_name = mind.name

    def _run(
        self,
        query: Text
    ):
        # Run the query on the AI-Mind.
        # The Minds API is OpenAI compatible and therefore, the OpenAI client can be used.
        openai_client = OpenAI(base_url=AIMindToolConstants.MINDS_API_BASE_URL, api_key=self.api_key)

        completion = openai_client.chat.completions.create(
            model=self.mind_name,
            messages=[{"role": "user", "content": query}],
            stream=False,
        )

        return completion.choices[0].message.content