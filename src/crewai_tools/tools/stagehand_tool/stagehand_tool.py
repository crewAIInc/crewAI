"""
A tool for using Stagehand's AI-powered web automation capabilities in CrewAI.

This tool provides access to Stagehand's three core APIs:
- act: Perform web interactions
- extract: Extract information from web pages
- observe: Monitor web page changes

Each function takes atomic instructions to increase reliability.
"""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type, Union

from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

# Define STAGEHAND_AVAILABLE at module level
STAGEHAND_AVAILABLE = False
try:
    import stagehand

    STAGEHAND_AVAILABLE = True
except ImportError:
    pass  # Keep STAGEHAND_AVAILABLE as False


class StagehandResult(BaseModel):
    """Result from a Stagehand operation.

    Attributes:
        success: Whether the operation completed successfully
        data: The result data from the operation
        error: Optional error message if the operation failed
    """

    success: bool = Field(
        ..., description="Whether the operation completed successfully"
    )
    data: Union[str, Dict, List] = Field(
        ..., description="The result data from the operation"
    )
    error: Optional[str] = Field(
        None, description="Optional error message if the operation failed"
    )


class StagehandToolConfig(BaseModel):
    """Configuration for the StagehandTool.

    Attributes:
        api_key: OpenAI API key for Stagehand authentication
        timeout: Maximum time in seconds to wait for operations (default: 30)
        retry_attempts: Number of times to retry failed operations (default: 3)
    """

    api_key: str = Field(..., description="OpenAI API key for Stagehand authentication")
    timeout: int = Field(
        30, description="Maximum time in seconds to wait for operations"
    )
    retry_attempts: int = Field(
        3, description="Number of times to retry failed operations"
    )


class StagehandToolSchema(BaseModel):
    """Schema for the StagehandTool input parameters.

    Examples:
        ```python
        # Using the 'act' API to click a button
        tool.run(
            api_method="act",
            instruction="Click the 'Sign In' button"
        )

        # Using the 'extract' API to get text
        tool.run(
            api_method="extract",
            instruction="Get the text content of the main article"
        )

        # Using the 'observe' API to monitor changes
        tool.run(
            api_method="observe",
            instruction="Watch for changes in the shopping cart count"
        )
        ```
    """

    api_method: str = Field(
        ...,
        description="The Stagehand API to use: 'act' for interactions, 'extract' for getting content, or 'observe' for monitoring changes",
        pattern="^(act|extract|observe)$",
    )
    instruction: str = Field(
        ...,
        description="An atomic instruction for Stagehand to execute. Instructions should be simple and specific to increase reliability.",
        min_length=1,
        max_length=500,
    )


class StagehandTool(BaseTool):
    """A tool for using Stagehand's AI-powered web automation capabilities.

    This tool provides access to Stagehand's three core APIs:
    - act: Perform web interactions (e.g., clicking buttons, filling forms)
    - extract: Extract information from web pages (e.g., getting text content)
    - observe: Monitor web page changes (e.g., watching for updates)

    Each function takes atomic instructions to increase reliability.

    Required Environment Variables:
        OPENAI_API_KEY: API key for OpenAI (required by Stagehand)

    Examples:
        ```python
        tool = StagehandTool()

        # Perform a web interaction
        result = tool.run(
            api_method="act",
            instruction="Click the 'Sign In' button"
        )

        # Extract content from a page
        content = tool.run(
            api_method="extract",
            instruction="Get the text content of the main article"
        )

        # Monitor for changes
        changes = tool.run(
            api_method="observe",
            instruction="Watch for changes in the shopping cart count"
        )
        ```
    """

    name: str = "StagehandTool"
    description: str = (
        "A tool that uses Stagehand's AI-powered web automation to interact with websites. "
        "It can perform actions (click, type, etc.), extract content, and observe changes. "
        "Each instruction should be atomic (simple and specific) to increase reliability."
    )
    args_schema: Type[BaseModel] = StagehandToolSchema

    def __init__(
        self, config: StagehandToolConfig | None = None, **kwargs: Any
    ) -> None:
        """Initialize the StagehandTool.

        Args:
            config: Optional configuration for the tool. If not provided,
                   will attempt to use OPENAI_API_KEY from environment.
            **kwargs: Additional keyword arguments passed to the base class.

        Raises:
            ImportError: If the stagehand package is not installed
            ValueError: If no API key is provided via config or environment
        """
        super().__init__(**kwargs)

        if not STAGEHAND_AVAILABLE:
            raise ImportError(
                "The 'stagehand' package is required to use this tool. "
                "Please install it with: pip install stagehand"
            )

        # Use config if provided, otherwise try environment variable
        if config is not None:
            self.config = config
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Either provide config with api_key or set OPENAI_API_KEY environment variable"
                )
            self.config = StagehandToolConfig(
                api_key=api_key, timeout=30, retry_attempts=3
            )

    @lru_cache(maxsize=100)
    def _cached_run(self, api_method: str, instruction: str) -> Any:
        """Execute a cached Stagehand command.

        This method is cached to improve performance for repeated operations.

        Args:
            api_method: The Stagehand API to use ('act', 'extract', or 'observe')
            instruction: An atomic instruction for Stagehand to execute

        Returns:
            The raw result from the Stagehand API call

        Raises:
            ValueError: If an invalid api_method is provided
            Exception: If the Stagehand API call fails
        """
        logger.debug(
            "Cache operation - Method: %s, Instruction length: %d",
            api_method,
            len(instruction),
        )

        # Initialize Stagehand with configuration
        logger.info(
            "Initializing Stagehand (timeout=%ds, retries=%d)",
            self.config.timeout,
            self.config.retry_attempts,
        )
        st = stagehand.Stagehand(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            retry_attempts=self.config.retry_attempts,
        )

        # Call the appropriate Stagehand API based on the method
        logger.info(
            "Executing %s operation with instruction: %s", api_method, instruction[:100]
        )
        try:
            if api_method == "act":
                result = st.act(instruction)
            elif api_method == "extract":
                result = st.extract(instruction)
            elif api_method == "observe":
                result = st.observe(instruction)
            else:
                raise ValueError(f"Unknown api_method: {api_method}")

            logger.info("Successfully executed %s operation", api_method)
            return result

        except Exception as e:
            logger.warning(
                "Operation failed (method=%s, error=%s), will be retried on next attempt",
                api_method,
                str(e),
            )
            raise

    def _run(self, api_method: str, instruction: str, **kwargs: Any) -> StagehandResult:
        """Execute a Stagehand command using the specified API method.

        Args:
            api_method: The Stagehand API to use ('act', 'extract', or 'observe')
            instruction: An atomic instruction for Stagehand to execute
            **kwargs: Additional keyword arguments passed to the Stagehand API

        Returns:
            StagehandResult containing the operation result and status
        """
        try:
            # Log operation context
            logger.debug(
                "Starting operation - Method: %s, Instruction length: %d, Args: %s",
                api_method,
                len(instruction),
                kwargs,
            )

            # Use cached execution
            result = self._cached_run(api_method, instruction)
            logger.info("Operation completed successfully")
            return StagehandResult(success=True, data=result)

        except stagehand.AuthenticationError as e:
            logger.error(
                "Authentication failed - Method: %s, Error: %s", api_method, str(e)
            )
            return StagehandResult(
                success=False, data={}, error=f"Authentication failed: {str(e)}"
            )
        except stagehand.APIError as e:
            logger.error("API error - Method: %s, Error: %s", api_method, str(e))
            return StagehandResult(success=False, data={}, error=f"API error: {str(e)}")
        except stagehand.BrowserError as e:
            logger.error("Browser error - Method: %s, Error: %s", api_method, str(e))
            return StagehandResult(
                success=False, data={}, error=f"Browser error: {str(e)}"
            )
        except Exception as e:
            logger.error(
                "Unexpected error - Method: %s, Error type: %s, Message: %s",
                api_method,
                type(e).__name__,
                str(e),
            )
            return StagehandResult(
                success=False, data={}, error=f"Unexpected error: {str(e)}"
            )
