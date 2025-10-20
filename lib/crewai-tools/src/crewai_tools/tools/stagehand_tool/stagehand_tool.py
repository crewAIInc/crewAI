import asyncio
import json
import os
import re
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# Define a flag to track whether stagehand is available
_HAS_STAGEHAND = False

try:
    from stagehand import (  # type: ignore[import-untyped]
        Stagehand,
        StagehandConfig,
        StagehandPage,
        configure_logging,
    )
    from stagehand.schemas import (  # type: ignore[import-untyped]
        ActOptions,
        AvailableModel,
        ExtractOptions,
        ObserveOptions,
    )

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
    class AvailableModel:  # type: ignore[no-redef]
        CLAUDE_3_7_SONNET_LATEST = "anthropic.claude-3-7-sonnet-20240607"


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
    data: str | dict | list = Field(
        ..., description="The result data from the operation"
    )
    error: str | None = Field(
        None, description="Optional error message if the operation failed"
    )


class StagehandToolSchema(BaseModel):
    """Input for StagehandTool."""

    instruction: str | None = Field(
        None,
        description="Single atomic action with location context. For reliability on complex pages, use ONE specific action with location hints. Good examples: 'Click the search input field in the header', 'Type Italy in the focused field', 'Press Enter', 'Click the first link in the results area'. Avoid combining multiple actions. For 'navigate' command type, this can be omitted if only URL is provided.",
    )
    url: str | None = Field(
        None,
        description="The URL to navigate to before executing the instruction. MUST be used with 'navigate' command. ",
    )
    command_type: str | None = Field(
        "act",
        description="""The type of command to execute (choose one):
        - 'act': Perform an action like clicking buttons, filling forms, etc. (default)
        - 'navigate': Specifically navigate to a URL
        - 'extract': Extract structured data from the page
        - 'observe': Identify and analyze elements on the page
        """,
    )


class StagehandTool(BaseTool):
    """A tool that uses Stagehand to automate web browser interactions using natural language with atomic action handling.

    Stagehand allows AI agents to interact with websites through a browser,
    performing actions like clicking buttons, filling forms, and extracting data.

    The tool supports four main command types:
    1. act - Perform actions like clicking, typing, scrolling, or navigating
    2. navigate - Specifically navigate to a URL (shorthand for act with navigation)
    3. extract - Extract structured data from web pages
    4. observe - Identify and analyze elements on a page

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
    For reliability on complex pages, use specific, atomic instructions with location hints:
    - Good: "Click the search box in the header"
    - Good: "Type 'Italy' in the focused field"
    - Bad: "Search for Italy and click the first result"

    For different types of tasks, specify the command_type:
    - 'act': For performing one atomic action (default)
    - 'navigate': For navigating to a URL
    - 'extract': For getting data from a specific page section
    - 'observe': For finding elements in a specific area
    """
    args_schema: type[BaseModel] = StagehandToolSchema

    # Stagehand configuration
    api_key: str | None = None
    project_id: str | None = None
    model_api_key: str | None = None
    model_name: AvailableModel | None = AvailableModel.CLAUDE_3_7_SONNET_LATEST
    server_url: str | None = "https://api.stagehand.browserbase.com/v1"
    headless: bool = False
    dom_settle_timeout_ms: int = 3000
    self_heal: bool = True
    wait_for_captcha_solves: bool = True
    verbose: int = 1

    # Token management settings
    max_retries_on_token_limit: int = 3
    use_simplified_dom: bool = True

    # Instance variables
    _stagehand: Stagehand | None = None
    _page: StagehandPage | None = None
    _session_id: str | None = None
    _testing: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        model_api_key: str | None = None,
        model_name: str | None = None,
        server_url: str | None = None,
        session_id: str | None = None,
        headless: bool | None = None,
        dom_settle_timeout_ms: int | None = None,
        self_heal: bool | None = None,
        wait_for_captcha_solves: bool | None = None,
        verbose: int | None = None,
        _testing: bool = False,
        **kwargs,
    ):
        # Set testing flag early so that other init logic can rely on it
        self._testing = _testing
        super().__init__(**kwargs)

        # Set up logger
        import logging

        self._logger = logging.getLogger(__name__)

        # Set configuration from parameters or environment
        self.api_key = api_key or os.getenv("BROWSERBASE_API_KEY")
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")

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
        if not self._testing:
            log_level = {1: "INFO", 2: "WARNING", 3: "DEBUG"}.get(self.verbose, "ERROR")
            configure_logging(
                level=log_level, remove_logger_name=True, quiet_dependencies=True
            )

        self._check_required_credentials()

    def _check_required_credentials(self):
        """Validate that required credentials are present."""
        if not self._testing and not _HAS_STAGEHAND:
            raise ImportError(
                "`stagehand` package not found, please run `uv add stagehand`"
            )

        if not self.api_key:
            raise ValueError("api_key is required (or set BROWSERBASE_API_KEY in env).")
        if not self.project_id:
            raise ValueError(
                "project_id is required (or set BROWSERBASE_PROJECT_ID in env)."
            )

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except Exception:  # noqa: S110
            pass

    def _get_model_api_key(self):
        """Get the appropriate API key based on the model being used."""
        # Check model type and get appropriate key
        model_str = str(self.model_name)
        if "gpt" in model_str.lower():
            return self.model_api_key or os.getenv("OPENAI_API_KEY")
        if "claude" in model_str.lower() or "anthropic" in model_str.lower():
            return self.model_api_key or os.getenv("ANTHROPIC_API_KEY")
        if "gemini" in model_str.lower():
            return self.model_api_key or os.getenv("GOOGLE_API_KEY")
        # Default to trying OpenAI, then Anthropic
        return (
            self.model_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )

    async def _setup_stagehand(self, session_id: str | None = None):
        """Initialize Stagehand if not already set up."""
        # If we're in testing mode, return mock objects
        if self._testing:
            if not self._stagehand:
                # Create mock objects for testing
                class MockPage:
                    async def act(self, options):
                        mock_result = type("MockResult", (), {})()
                        mock_result.model_dump = lambda: {
                            "message": "Action completed successfully"
                        }
                        return mock_result

                    async def goto(self, url):
                        return None

                    async def extract(self, options):
                        mock_result = type("MockResult", (), {})()
                        mock_result.model_dump = lambda: {"data": "Extracted content"}
                        return mock_result

                    async def observe(self, options):
                        mock_result1 = type(
                            "MockResult",
                            (),
                            {"description": "Test element", "method": "click"},
                        )()
                        return [mock_result1]

                    async def wait_for_load_state(self, state):
                        return None

                class MockStagehand:
                    def __init__(self):
                        self.page = MockPage()
                        self.session_id = "test-session-id"

                    async def init(self):
                        return None

                    async def close(self):
                        return None

                self._stagehand = MockStagehand()
                await self._stagehand.init()
                self._page = self._stagehand.page
                self._session_id = self._stagehand.session_id

            return self._stagehand, self._page

        # Normal initialization for non-testing mode
        if not self._stagehand:
            # Get the appropriate API key based on model type
            model_api_key = self._get_model_api_key()

            if not model_api_key:
                raise ValueError(
                    "No appropriate API key found for model. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
                )

            # Build the StagehandConfig with proper parameter names
            config = StagehandConfig(
                env="BROWSERBASE",
                apiKey=self.api_key,  # Browserbase API key (camelCase)
                projectId=self.project_id,  # Browserbase project ID (camelCase)
                modelApiKey=model_api_key,  # LLM API key - auto-detected based on model
                modelName=self.model_name,
                apiUrl=self.server_url
                if self.server_url
                else "https://api.stagehand.browserbase.com/v1",
                domSettleTimeoutMs=self.dom_settle_timeout_ms,
                selfHeal=self.self_heal,
                waitForCaptchaSolves=self.wait_for_captcha_solves,
                verbose=self.verbose,
                browserbaseSessionID=session_id or self._session_id,
            )

            # Initialize Stagehand with config
            self._stagehand = Stagehand(config=config)

            # Initialize the Stagehand instance
            await self._stagehand.init()
            self._page = self._stagehand.page
            self._session_id = self._stagehand.session_id

        return self._stagehand, self._page

    def _extract_steps(self, instruction: str) -> list[str]:
        """Extract individual steps from multi-step instructions."""
        # Check for numbered steps (Step 1:, Step 2:, etc.)
        if re.search(r"Step \d+:", instruction, re.IGNORECASE):
            steps = re.findall(
                r"Step \d+:\s*([^;]+?)(?=Step \d+:|$)",
                instruction,
                re.IGNORECASE | re.DOTALL,
            )
            return [step.strip() for step in steps if step.strip()]
        # Check for semicolon-separated instructions
        if ";" in instruction:
            return [step.strip() for step in instruction.split(";") if step.strip()]
        return [instruction]

    def _simplify_instruction(self, instruction: str) -> str:
        """Simplify complex instructions to basic actions."""
        # Extract the core action from complex instructions
        instruction_lower = instruction.lower()

        if "search" in instruction_lower and "click" in instruction_lower:
            # For search tasks, focus on the search action first
            if "type" in instruction_lower or "enter" in instruction_lower:
                return "click on the search input field"
            return "search for content on the page"
        if "click" in instruction_lower:
            # Extract what to click
            if "button" in instruction_lower:
                return "click the button"
            if "link" in instruction_lower:
                return "click the link"
            if "search" in instruction_lower:
                return "click the search field"
            return "click on the element"
        if "type" in instruction_lower or "enter" in instruction_lower:
            return "type in the input field"
        return instruction  # Return as-is if can't simplify

    async def _async_run(
        self,
        instruction: str | None = None,
        url: str | None = None,
        command_type: str = "act",
    ):
        """Override _async_run with improved atomic action handling."""
        # Handle missing instruction based on command type
        if not instruction:
            if command_type == "navigate" and url:
                instruction = f"Navigate to {url}"
            elif command_type == "observe":
                instruction = "Observe elements on the page"
            elif command_type == "extract":
                instruction = "Extract information from the page"
            else:
                instruction = "Perform the requested action"

        # For testing mode, return mock result directly without calling parent
        if self._testing:
            mock_data = {
                "message": f"Mock {command_type} completed successfully",
                "instruction": instruction,
            }
            if url:
                mock_data["url"] = url
            return self._format_result(True, mock_data)

        try:
            _, page = await self._setup_stagehand(self._session_id)

            self._logger.info(
                f"Executing {command_type} with instruction: {instruction}"
            )

            # Get the API key to pass to model operations
            model_api_key = self._get_model_api_key()
            model_client_options = {"apiKey": model_api_key}

            # Always navigate first if URL is provided and we're doing actions
            if url and command_type.lower() == "act":
                self._logger.info(f"Navigating to {url} before performing actions")
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                # Small delay to ensure page is fully loaded
                await asyncio.sleep(1)

            # Process according to command type
            if command_type.lower() == "act":
                # Extract steps from complex instructions
                steps = self._extract_steps(instruction)
                self._logger.info(f"Extracted {len(steps)} steps: {steps}")

                results = []
                for i, step in enumerate(steps):
                    self._logger.info(f"Executing step {i + 1}/{len(steps)}: {step}")

                    try:
                        # Create act options with API key for each step
                        from stagehand.schemas import ActOptions

                        act_options = ActOptions(
                            action=step,
                            modelName=self.model_name,
                            domSettleTimeoutMs=self.dom_settle_timeout_ms,
                            modelClientOptions=model_client_options,
                        )

                        result = await page.act(act_options)
                        results.append(result.model_dump())

                        # Small delay between steps to let DOM settle
                        if i < len(steps) - 1:  # Don't delay after last step
                            await asyncio.sleep(0.5)

                    except Exception as step_error:
                        error_msg = f"Step failed: {step_error}"
                        self._logger.warning(f"Step {i + 1} failed: {error_msg}")

                        # Try with simplified instruction
                        try:
                            simplified = self._simplify_instruction(step)
                            if simplified != step:
                                self._logger.info(
                                    f"Retrying with simplified instruction: {simplified}"
                                )

                                act_options = ActOptions(
                                    action=simplified,
                                    modelName=self.model_name,
                                    domSettleTimeoutMs=self.dom_settle_timeout_ms,
                                    modelClientOptions=model_client_options,
                                )

                                result = await page.act(act_options)
                                results.append(result.model_dump())
                            else:
                                # If we can't simplify or retry fails, record the error
                                results.append({"error": error_msg, "step": step})
                        except Exception as retry_error:
                            self._logger.error(f"Retry also failed: {retry_error}")
                            results.append({"error": str(retry_error), "step": step})

                # Return combined results
                if len(results) == 1:
                    # Single step, return as-is
                    if "error" in results[0]:
                        return self._format_result(
                            False, results[0], results[0]["error"]
                        )
                    return self._format_result(True, results[0])
                # Multiple steps, return all results
                has_errors = any("error" in result for result in results)
                return self._format_result(not has_errors, {"steps": results})

            if command_type.lower() == "navigate":
                # For navigation, use the goto method directly
                if not url:
                    error_msg = "No URL provided for navigation. Please provide a URL."
                    self._logger.error(error_msg)
                    return self._format_result(False, {}, error_msg)

                result = await page.goto(url)
                self._logger.info(f"Navigate operation completed to {url}")
                return self._format_result(
                    True,
                    {
                        "url": url,
                        "message": f"Successfully navigated to {url}",
                    },
                )

            if command_type.lower() == "extract":
                # Create extract options with API key
                from stagehand.schemas import ExtractOptions

                extract_options = ExtractOptions(
                    instruction=instruction,
                    modelName=self.model_name,
                    domSettleTimeoutMs=self.dom_settle_timeout_ms,
                    useTextExtract=True,
                    modelClientOptions=model_client_options,  # Add API key here
                )

                result = await page.extract(extract_options)
                self._logger.info(f"Extract operation completed successfully {result}")
                return self._format_result(True, result.model_dump())

            if command_type.lower() == "observe":
                # Create observe options with API key
                from stagehand.schemas import ObserveOptions

                observe_options = ObserveOptions(
                    instruction=instruction,
                    modelName=self.model_name,
                    onlyVisible=True,
                    domSettleTimeoutMs=self.dom_settle_timeout_ms,
                    modelClientOptions=model_client_options,  # Add API key here
                )

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
                return self._format_result(True, formatted_results)

            error_msg = f"Unknown command type: {command_type}"
            self._logger.error(error_msg)
            return self._format_result(False, {}, error_msg)

        except Exception as e:
            error_msg = f"Error using Stagehand: {e!s}"
            self._logger.error(f"Operation failed: {error_msg}")
            return self._format_result(False, {}, error_msg)

    def _format_result(self, success, data, error=None):
        """Helper to format results consistently."""
        return StagehandResult(success=success, data=data, error=error)

    def _run(
        self,
        instruction: str | None = None,
        url: str | None = None,
        command_type: str = "act",
    ) -> str:
        """Run the Stagehand tool with the given instruction.

        Args:
            instruction: Natural language instruction for browser automation
            url: Optional URL to navigate to before executing the instruction
            command_type: Type of command to execute ('act', 'extract', or 'observe')

        Returns:
            The result of the browser automation task
        """
        # Handle missing instruction based on command type
        if not instruction:
            if command_type == "navigate" and url:
                instruction = f"Navigate to {url}"
            elif command_type == "observe":
                instruction = "Observe elements on the page"
            elif command_type == "extract":
                instruction = "Extract information from the page"
            else:
                instruction = "Perform the requested action"
        # Create an event loop if we're not already in one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an existing event loop, use it
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._async_run(instruction, url, command_type)
                    )
                    result = future.result()
            else:
                # We have a loop but it's not running
                result = loop.run_until_complete(
                    self._async_run(instruction, url, command_type)
                )

            # Format the result for output
            if result.success:
                if command_type.lower() == "act":
                    if isinstance(result.data, dict) and "steps" in result.data:
                        # Multiple steps
                        step_messages = []
                        for i, step in enumerate(result.data["steps"]):
                            if "error" in step:
                                step_messages.append(
                                    f"Step {i + 1}: Failed - {step['error']}"
                                )
                            else:
                                step_messages.append(
                                    f"Step {i + 1}: {step.get('message', 'Completed')}"
                                )
                        return "\n".join(step_messages)
                    return f"Action result: {result.data.get('message', 'Completed')}"
                if command_type.lower() == "extract":
                    return f"Extracted data: {json.dumps(result.data, indent=2)}"
                if command_type.lower() == "observe":
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
                return json.dumps(result.data, indent=2)
            return f"Error: {result.error}"

        except RuntimeError:
            # No event loop exists, create one
            result = asyncio.run(self._async_run(instruction, url, command_type))

            if result.success:
                if isinstance(result.data, dict):
                    return json.dumps(result.data, indent=2)
                return str(result.data)
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
                                import concurrent.futures

                                with (
                                    concurrent.futures.ThreadPoolExecutor() as executor
                                ):
                                    future = executor.submit(
                                        asyncio.run, self._async_close()
                                    )
                                    future.result()
                            else:
                                loop.run_until_complete(self._async_close())
                        except RuntimeError:
                            asyncio.run(self._async_close())
                    else:
                        # Handle non-async close method (for mocks)
                        self._stagehand.close()
            except Exception:  # noqa: S110
                # Log but don't raise - we're cleaning up
                pass

            self._stagehand = None

        if self._page:
            self._page = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up resources."""
        self.close()
