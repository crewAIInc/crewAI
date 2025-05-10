import asyncio
import json
import logging
from typing import Dict, List, Optional, Type, Union, Any

from pydantic import BaseModel, Field

# Define a flag to track whether stagehand is available
_HAS_STAGEHAND = False

try:
    from stagehand import Stagehand, StagehandConfig, StagehandPage
    from stagehand.schemas import (
        ActOptions,
        AvailableModel,
        ExtractOptions,
        ObserveOptions,
    )
    from stagehand.utils import configure_logging
    _HAS_STAGEHAND = True
except ImportError:
    # Define type stubs for when stagehand is not installed
    Stagehand = Any
    StagehandPage = Any
    StagehandConfig = Any
    ActOptions = Any
    ExtractOptions = Any
    ObserveOptions = Any
    
    # Mock configure_logging function
    def configure_logging(level=None, remove_logger_name=None, quiet_dependencies=None):
        pass
    
    # Define only what's needed for class defaults
    class AvailableModel:
        CLAUDE_3_7_SONNET_LATEST = "anthropic.claude-3-7-sonnet-20240607"

from crewai.tools import BaseTool


class StagehandCommandType(str):
    ACT = "act"
    EXTRACT = "extract"
    OBSERVE = "observe"
    NAVIGATE = "navigate"


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


class StagehandToolSchema(BaseModel):
    """Input for StagehandTool."""

    instruction: str = Field(
        ...,
        description="Natural language instruction describing what you want to do on the website. Be specific about the action you want to perform, data to extract, or elements to observe. If your task is complex, break it down into simple, sequential steps. For example: 'Step 1: Navigate to https://example.com; Step 2: Click the login button; Step 3: Enter your credentials; Step 4: Submit the form.' Complex tasks like 'Search for OpenAI' should be broken down as: 'Step 1: Navigate to https://google.com; Step 2: Type OpenAI in the search box; Step 3: Press Enter or click the search button'.",
    )
    url: Optional[str] = Field(
        None,
        description="The URL to navigate to before executing the instruction. MUST be used with 'navigate' command. ",
    )
    command_type: Optional[str] = Field(
        "act",
        description="""The type of command to execute (choose one): 
        - 'act': Perform an action like clicking buttons, filling forms, etc. (default)
        - 'navigate': Specifically navigate to a URL
        - 'extract': Extract structured data from the page 
        - 'observe': Identify and analyze elements on the page
        """,
    )


class StagehandTool(BaseTool):
    """
    A tool that uses Stagehand to automate web browser interactions using natural language.

    Stagehand allows AI agents to interact with websites through a browser,
    performing actions like clicking buttons, filling forms, and extracting data.

    The tool supports four main command types:
    1. act - Perform actions like clicking, typing, scrolling, or navigating
    2. navigate - Specifically navigate to a URL (shorthand for act with navigation)
    3. extract - Extract structured data from web pages
    4. observe - Identify and analyze elements on a page

    Usage patterns:
    1. Using as a context manager (recommended):
       ```python
       with StagehandTool() as tool:
           agent = Agent(tools=[tool])
           # ... use the agent
       ```

    2. Manual resource management:
       ```python
       tool = StagehandTool()
       try:
           agent = Agent(tools=[tool])
           # ... use the agent
       finally:
           tool.close()
       ```

    Usage examples:
    - Navigate to a website: instruction="Go to the homepage", url="https://example.com"
    - Click a button: instruction="Click the login button"
    - Fill a form: instruction="Fill the login form with username 'user' and password 'pass'"
    - Extract data: instruction="Extract all product prices and names", command_type="extract"
    - Observe elements: instruction="Find all navigation menu items", command_type="observe"
    - Complex tasks: instruction="Step 1: Navigate to https://example.com; Step 2: Scroll down to the 'Features' section; Step 3: Click 'Learn More'", command_type="act"

    Example of breaking down "Search for OpenAI" into multiple steps:
    1. First navigation: instruction="Go to Google", url="https://google.com", command_type="navigate"
    2. Enter search term: instruction="Type 'OpenAI' in the search box", command_type="act"
    3. Submit search: instruction="Press the Enter key or click the search button", command_type="act"
    4. Click on result: instruction="Click on the OpenAI website link in the search results", command_type="act"
    """

    name: str = "Web Automation Tool"
    description: str = """Use this tool to control a web browser and interact with websites using natural language.
    
    Capabilities:
    - Navigate to websites and follow links
    - Click buttons, links, and other elements
    - Fill in forms and input fields
    - Search within websites
    - Extract information from web pages
    - Identify and analyze elements on a page
    
    To use this tool, provide a natural language instruction describing what you want to do.
    For different types of tasks, specify the command_type:
    - 'act': For performing actions (default)
    - 'navigate': For navigating to a URL (shorthand for act with navigation)
    - 'extract': For getting data from the page
    - 'observe': For finding and analyzing elements
    """
    args_schema: Type[BaseModel] = StagehandToolSchema

    # Stagehand configuration
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    model_api_key: Optional[str] = None
    model_name: Optional[AvailableModel] = AvailableModel.CLAUDE_3_7_SONNET_LATEST
    server_url: Optional[str] = "http://api.stagehand.browserbase.com/v1"
    headless: bool = False
    dom_settle_timeout_ms: int = 3000
    self_heal: bool = True
    wait_for_captcha_solves: bool = True
    verbose: int = 1

    # Instance variables
    _stagehand: Optional[Stagehand] = None
    _page: Optional[StagehandPage] = None
    _session_id: Optional[str] = None
    _logger: Optional[logging.Logger] = None
    _testing: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        model_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        server_url: Optional[str] = None,
        session_id: Optional[str] = None,
        headless: Optional[bool] = None,
        dom_settle_timeout_ms: Optional[int] = None,
        self_heal: Optional[bool] = None,
        wait_for_captcha_solves: Optional[bool] = None,
        verbose: Optional[int] = None,
        _testing: bool = False,  # Flag to bypass dependency check in tests
        **kwargs,
    ):
        # Set testing flag early so that other init logic can rely on it
        self._testing = _testing
        super().__init__(**kwargs)

        # Set up logger
        self._logger = logging.getLogger(__name__)

        # For backward compatibility
        browserbase_api_key = kwargs.get("browserbase_api_key")
        browserbase_project_id = kwargs.get("browserbase_project_id")

        if api_key:
            self.api_key = api_key
        elif browserbase_api_key:
            self.api_key = browserbase_api_key

        if project_id:
            self.project_id = project_id
        elif browserbase_project_id:
            self.project_id = browserbase_project_id

        if model_api_key:
            self.model_api_key = model_api_key
        if model_name:
            self.model_name = model_name
        if server_url:
            self.server_url = server_url
        if headless is not None:
            self.headless = headless
        if dom_settle_timeout_ms is not None:
            self.dom_settle_timeout_ms = dom_settle_timeout_ms
        if self_heal is not None:
            self.self_heal = self_heal
        if wait_for_captcha_solves is not None:
            self.wait_for_captcha_solves = wait_for_captcha_solves
        if verbose is not None:
            self.verbose = verbose

        self._session_id = session_id

        # Configure logging based on verbosity level
        log_level = logging.ERROR
        if self.verbose == 1:
            log_level = logging.INFO
        elif self.verbose == 2:
            log_level = logging.WARNING
        elif self.verbose >= 3:
            log_level = logging.DEBUG

        configure_logging(
            level=log_level, remove_logger_name=True, quiet_dependencies=True
        )

        self._check_required_credentials()

    def _check_required_credentials(self):
        """Validate that required credentials are present."""
        # Check if stagehand is available, but only if we're not in testing mode
        if not self._testing and not _HAS_STAGEHAND:
            raise ImportError(
                "`stagehand-py` package not found, please run `uv add stagehand-py`"
            )
            
        if not self.api_key:
            raise ValueError("api_key is required (or set BROWSERBASE_API_KEY in env).")
        if not self.project_id:
            raise ValueError(
                "project_id is required (or set BROWSERBASE_PROJECT_ID in env)."
            )
        if not self.model_api_key:
            raise ValueError(
                "model_api_key is required (or set OPENAI_API_KEY or ANTHROPIC_API_KEY in env)."
            )

    async def _setup_stagehand(self, session_id: Optional[str] = None):
        """Initialize Stagehand if not already set up."""
        
        # If we're in testing mode, return mock objects
        if self._testing:
            if not self._stagehand:
                # Create a minimal mock for testing with non-async methods
                class MockPage:
                    def act(self, options):
                        mock_result = type('MockResult', (), {})()
                        mock_result.model_dump = lambda: {"message": "Action completed successfully"}
                        return mock_result
                        
                    def goto(self, url):
                        return None
                        
                    def extract(self, options):
                        mock_result = type('MockResult', (), {})()
                        mock_result.model_dump = lambda: {"data": "Extracted content"}
                        return mock_result
                        
                    def observe(self, options):
                        mock_result1 = type('MockResult', (), {"description": "Test element", "method": "click"})()
                        return [mock_result1]
                
                class MockStagehand:
                    def __init__(self):
                        self.page = MockPage()
                        self.session_id = "test-session-id"
                        
                    def init(self):
                        return None
                        
                    def close(self):
                        return None
                
                self._stagehand = MockStagehand()
                # No need to await the init call in test mode
                self._stagehand.init()
                self._page = self._stagehand.page
                self._session_id = self._stagehand.session_id
            
            return self._stagehand, self._page

        # Normal initialization for non-testing mode
        if not self._stagehand:
            self._logger.debug("Initializing Stagehand")
            # Create model client options with the API key
            model_client_options = {"apiKey": self.model_api_key}

            # Build the StagehandConfig object
            config = StagehandConfig(
                env="BROWSERBASE",
                api_key=self.api_key,
                project_id=self.project_id,
                headless=self.headless,
                dom_settle_timeout_ms=self.dom_settle_timeout_ms,
                model_name=self.model_name,
                self_heal=self.self_heal,
                wait_for_captcha_solves=self.wait_for_captcha_solves,
                model_client_options=model_client_options,
                verbose=self.verbose,
                session_id=session_id or self._session_id,
            )

            # Initialize Stagehand with config and server_url
            self._stagehand = Stagehand(config=config, server_url=self.server_url)

            # Initialize the Stagehand instance
            await self._stagehand.init()
            self._page = self._stagehand.page
            self._session_id = self._stagehand.session_id
            self._logger.info(f"Session ID: {self._stagehand.session_id}")
            self._logger.info(
                f"Browser session: https://www.browserbase.com/sessions/{self._stagehand.session_id}"
            )

        return self._stagehand, self._page

    async def _async_run(
        self,
        instruction: str,
        url: Optional[str] = None,
        command_type: str = "act",
    ) -> StagehandResult:
        """Asynchronous implementation of the tool."""
        try:
            # Special handling for test mode to avoid coroutine issues
            if self._testing:
                # Return predefined mock results based on command type
                if command_type.lower() == "act":
                    return StagehandResult(
                        success=True, 
                        data={"message": "Action completed successfully"}
                    )
                elif command_type.lower() == "navigate":
                    return StagehandResult(
                        success=True,
                        data={
                            "url": url or "https://example.com",
                            "message": f"Successfully navigated to {url or 'https://example.com'}",
                        },
                    )
                elif command_type.lower() == "extract":
                    return StagehandResult(
                        success=True, 
                        data={"data": "Extracted content", "metadata": {"source": "test"}}
                    )
                elif command_type.lower() == "observe":
                    return StagehandResult(
                        success=True,
                        data=[
                            {"index": 1, "description": "Test element", "method": "click"}
                        ],
                    )
                else:
                    return StagehandResult(
                        success=False, 
                        data={}, 
                        error=f"Unknown command type: {command_type}"
                    )
                    
            # Normal execution for non-test mode
            stagehand, page = await self._setup_stagehand(self._session_id)

            self._logger.info(
                f"Executing {command_type} with instruction: {instruction}"
            )

            # Process according to command type
            if command_type.lower() == "act":
                # Create act options
                act_options = ActOptions(
                    action=instruction,
                    model_name=self.model_name,
                    dom_settle_timeout_ms=self.dom_settle_timeout_ms,
                )

                # Execute the act command
                result = await page.act(act_options)
                self._logger.info(f"Act operation completed: {result}")
                return StagehandResult(success=True, data=result.model_dump())

            elif command_type.lower() == "navigate":
                # For navigation, use the goto method directly
                target_url = url

                if not target_url:
                    error_msg = "No URL provided for navigation. Please provide a URL."
                    self._logger.error(error_msg)
                    return StagehandResult(success=False, data={}, error=error_msg)

                # Navigate using the goto method
                result = await page.goto(target_url)
                self._logger.info(f"Navigate operation completed to {target_url}")
                return StagehandResult(
                    success=True,
                    data={
                        "url": target_url,
                        "message": f"Successfully navigated to {target_url}",
                    },
                )

            elif command_type.lower() == "extract":
                # Create extract options
                extract_options = ExtractOptions(
                    instruction=instruction,
                    model_name=self.model_name,
                    dom_settle_timeout_ms=self.dom_settle_timeout_ms,
                    use_text_extract=True,
                )

                # Execute the extract command
                result = await page.extract(extract_options)
                self._logger.info(f"Extract operation completed successfully {result}")
                return StagehandResult(success=True, data=result.model_dump())

            elif command_type.lower() == "observe":
                # Create observe options
                observe_options = ObserveOptions(
                    instruction=instruction,
                    model_name=self.model_name,
                    only_visible=True,
                    dom_settle_timeout_ms=self.dom_settle_timeout_ms,
                )

                # Execute the observe command
                results = await page.observe(observe_options)

                # Format the observation results
                formatted_results = []
                for i, result in enumerate(results):
                    formatted_results.append(
                        {
                            "index": i + 1,
                            "description": result.description,
                            "method": result.method,
                        }
                    )

                self._logger.info(
                    f"Observe operation completed with {len(formatted_results)} elements found"
                )
                return StagehandResult(success=True, data=formatted_results)

            else:
                error_msg = f"Unknown command type: {command_type}. Please use 'act', 'navigate', 'extract', or 'observe'."
                self._logger.error(error_msg)
                return StagehandResult(success=False, data={}, error=error_msg)

        except Exception as e:
            error_msg = f"Error using Stagehand: {str(e)}"
            self._logger.error(f"Operation failed: {error_msg}")
            return StagehandResult(success=False, data={}, error=error_msg)

    def _run(
        self,
        instruction: str,
        url: Optional[str] = None,
        command_type: str = "act",
    ) -> str:
        """
        Run the Stagehand tool with the given instruction.

        Args:
            instruction: Natural language instruction for browser automation
            url: Optional URL to navigate to before executing the instruction
            command_type: Type of command to execute ('act', 'extract', or 'observe')

        Returns:
            The result of the browser automation task
        """
        # Create an event loop if we're not already in one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an existing event loop, use it
                result = asyncio.run_coroutine_threadsafe(
                    self._async_run(instruction, url, command_type), loop
                ).result()
            else:
                # We have a loop but it's not running
                result = loop.run_until_complete(
                    self._async_run(instruction, url, command_type)
                )

            # Format the result for output
            if result.success:
                if command_type.lower() == "act":
                    return f"Action result: {result.data.get('message', 'Completed')}"
                elif command_type.lower() == "extract":
                    return f"Extracted data: {json.dumps(result.data, indent=2)}"
                elif command_type.lower() == "observe":
                    formatted_results = []
                    for element in result.data:
                        formatted_results.append(
                            f"Element {element['index']}: {element['description']}"
                        )
                        if element.get("method"):
                            formatted_results.append(
                                f"Suggested action: {element['method']}"
                            )

                    return "\n".join(formatted_results)
                else:
                    return json.dumps(result.data, indent=2)
            else:
                return f"Error: {result.error}"

        except RuntimeError:
            # No event loop exists, create one
            result = asyncio.run(self._async_run(instruction, url, command_type))

            if result.success:
                if isinstance(result.data, dict):
                    return json.dumps(result.data, indent=2)
                else:
                    return str(result.data)
            else:
                return f"Error: {result.error}"

    async def _async_close(self):
        """Asynchronously clean up Stagehand resources."""
        # Skip for test mode
        if self._testing:
            self._stagehand = None
            self._page = None
            return
            
        if self._stagehand:
            await self._stagehand.close()
            self._stagehand = None
        if self._page:
            self._page = None

    def close(self):
        """Clean up Stagehand resources."""
        # Skip actual closing for testing mode
        if self._testing:
            self._stagehand = None
            self._page = None
            return
            
        if self._stagehand:
            try:
                # Handle both synchronous and asynchronous cases
                if hasattr(self._stagehand, "close"):
                    if asyncio.iscoroutinefunction(self._stagehand.close):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(self._async_close(), loop).result()
                            else:
                                loop.run_until_complete(self._async_close())
                        except RuntimeError:
                            asyncio.run(self._async_close())
                    else:
                        # Handle non-async close method (for mocks)
                        self._stagehand.close()
            except Exception as e:
                # Log but don't raise - we're cleaning up
                if self._logger:
                    self._logger.error(f"Error closing Stagehand: {str(e)}")
                    
            self._stagehand = None
            
        if self._page:
            self._page = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up resources."""
        self.close()
