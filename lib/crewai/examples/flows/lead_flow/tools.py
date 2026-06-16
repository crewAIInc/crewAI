import logging
from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger("lead_flow")


class LogLeadInput(BaseModel):
    message: str = Field(description="The message to log.")
    level: Literal["debug", "info", "warning", "error"] = "info"


class LogLeadTool(BaseTool):
    name: str = "log_lead"
    description: str = "Log a message about a lead that was not pursued."
    args_schema: type[BaseModel] = LogLeadInput

    def _run(self, message: str, level: str = "info") -> str:
        logger.log(logging.getLevelName(level.upper()), message)
        return message
