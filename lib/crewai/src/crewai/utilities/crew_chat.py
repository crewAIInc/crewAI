"""Interactive chat interface for CrewAI crews."""

import contextvars
import json
from pathlib import Path
import platform
import re
import sys
import threading
import time
from typing import Any, Final, Literal

import click
from crewai_core.printer import PRINTER
from packaging import version
import tomli

from crewai.crew import Crew
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.types.crew_chat import ChatInputField, ChatInputs
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.project_utils import read_toml
from crewai.utilities.types import LLMMessage
from crewai.version import get_crewai_version


MIN_REQUIRED_VERSION: Final[Literal["0.98.0"]] = "0.98.0"

DEFAULT_INPUT_DESCRIPTION: Final[str] = "Input value for the crew's tasks and agents."
DEFAULT_CREW_DESCRIPTION: Final[str] = "A CrewAI crew."


def check_conversational_crews_version(
    crewai_version: str, pyproject_data: dict[str, Any]
) -> bool:
    """Check if the installed crewAI version supports conversational crews.

    Args:
        crewai_version: The current version of crewAI.
        pyproject_data: Dictionary containing pyproject.toml data.

    Returns:
        True if version check passes, False otherwise.
    """
    try:
        if version.parse(crewai_version) < version.parse(MIN_REQUIRED_VERSION):
            click.secho(
                "You are using an older version of crewAI that doesn't support conversational crews. "
                "Run 'uv upgrade crewai' to get the latest version.",
                fg="red",
            )
            return False
    except version.InvalidVersion:
        click.secho("Invalid crewAI version format detected.", fg="red")
        return False
    return True


def run_chat() -> None:
    """Run an interactive chat loop using the Crew's chat LLM with function calling.

    Incorporates crew_name, crew_description, and input fields to build a tool schema.
    Exits if crew_name or crew_description are missing.
    """
    crewai_version = get_crewai_version()
    pyproject_data = read_toml()

    if not check_conversational_crews_version(crewai_version, pyproject_data):
        return

    crew, crew_name = load_crew_and_name()
    chat_llm = initialize_chat_llm(crew)
    if not chat_llm:
        return

    click.secho(
        "\nAnalyzing crew and required inputs - this may take 3 to 30 seconds "
        "depending on the complexity of your crew.",
        fg="white",
    )

    loading_complete = threading.Event()
    ctx = contextvars.copy_context()
    loading_thread = threading.Thread(
        target=ctx.run, args=(show_loading, loading_complete)
    )
    loading_thread.start()

    try:
        crew_chat_inputs = generate_crew_chat_inputs(crew, crew_name, chat_llm)
        crew_tool_schema = generate_crew_tool_schema(crew_chat_inputs)
        system_message = build_system_message(crew_chat_inputs)

        introductory_message = chat_llm.call(
            messages=[{"role": "system", "content": system_message}]
        )
    finally:
        loading_complete.set()
        loading_thread.join()

    click.secho("\nFinished analyzing crew.\n", fg="white")

    click.secho(f"Assistant: {introductory_message}\n", fg="green")

    messages: list[LLMMessage] = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": introductory_message},
    ]

    available_functions = {
        crew_chat_inputs.crew_name: create_tool_function(crew, messages),
    }

    chat_loop(chat_llm, messages, crew_tool_schema, available_functions)


def show_loading(event: threading.Event) -> None:
    """Display animated loading dots while processing."""
    while not event.is_set():
        PRINTER.print(".", end="")
        time.sleep(1)
    PRINTER.print("")


def initialize_chat_llm(crew: Crew) -> LLM | BaseLLM | None:
    """Initialize the chat LLM and handle exceptions."""
    try:
        return create_llm(crew.chat_llm)
    except Exception as e:
        click.secho(
            f"Unable to find a Chat LLM. Please make sure you set chat_llm on the crew: {e}",
            fg="red",
        )
        return None


def build_system_message(crew_chat_inputs: ChatInputs) -> str:
    """Build the initial system message for the chat."""
    required_fields_str = (
        ", ".join(
            f"{field.name} (desc: {field.description or 'n/a'})"
            for field in crew_chat_inputs.inputs
        )
        or "(No required fields detected)"
    )

    return (
        "You are a helpful AI assistant for the CrewAI platform. "
        "Your primary purpose is to assist users with the crew's specific tasks. "
        "You can answer general questions, but should guide users back to the crew's purpose afterward. "
        "For example, after answering a general question, remind the user of your main purpose, such as generating a research report, and prompt them to specify a topic or task related to the crew's purpose. "
        "You have a function (tool) you can call by name if you have all required inputs. "
        f"Those required inputs are: {required_fields_str}. "
        "Once you have them, call the function. "
        "Please keep your responses concise and friendly. "
        "If a user asks a question outside the crew's scope, provide a brief answer and remind them of the crew's purpose. "
        "After calling the tool, be prepared to take user feedback and make adjustments as needed. "
        "If you are ever unsure about a user's request or need clarification, ask the user for more information. "
        "Before doing anything else, introduce yourself with a friendly message like: 'Hey! I'm here to help you with [crew's purpose]. Could you please provide me with [inputs] so we can get started?' "
        "For example: 'Hey! I'm here to help you with uncovering and reporting cutting-edge developments through thorough research and detailed analysis. Could you please provide me with a topic you're interested in? This will help us generate a comprehensive research report and detailed analysis.'"
        f"\nCrew Name: {crew_chat_inputs.crew_name}"
        f"\nCrew Description: {crew_chat_inputs.crew_description}"
    )


def create_tool_function(crew: Crew, messages: list[LLMMessage]) -> Any:
    """Create a wrapper function for running the crew tool with messages."""

    def run_crew_tool_with_messages(**kwargs: Any) -> str:
        return run_crew_tool(crew, messages, **kwargs)

    return run_crew_tool_with_messages


def flush_input() -> None:
    """Flush any pending input from the user."""
    if platform.system() == "Windows":
        import msvcrt

        while msvcrt.kbhit():  # type: ignore[attr-defined]
            msvcrt.getch()  # type: ignore[attr-defined]
    else:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def chat_loop(
    chat_llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    crew_tool_schema: dict[str, Any],
    available_functions: dict[str, Any],
) -> None:
    """Main chat loop for interacting with the user."""
    while True:
        try:
            flush_input()

            user_input = get_user_input()
            handle_user_input(
                user_input, chat_llm, messages, crew_tool_schema, available_functions
            )

        except KeyboardInterrupt:  # noqa: PERF203
            click.echo("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            click.secho(f"An error occurred: {e}", fg="red")
            break


def get_user_input() -> str:
    """Collect multi-line user input with exit handling."""
    click.secho(
        "\nYou (type your message below. Press 'Enter' twice when you're done):",
        fg="blue",
    )
    user_input_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            return "exit"
        if line == "":
            break
        user_input_lines.append(line)
    return "\n".join(user_input_lines)


def handle_user_input(
    user_input: str,
    chat_llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    crew_tool_schema: dict[str, Any],
    available_functions: dict[str, Any],
) -> None:
    if user_input.strip().lower() == "exit":
        click.echo("Exiting chat. Goodbye!")
        return

    if not user_input.strip():
        click.echo("Empty message. Please provide input or type 'exit' to quit.")
        return

    messages.append({"role": "user", "content": user_input})

    click.echo()
    click.secho("Assistant is processing your input. Please wait...", fg="green")

    final_response = chat_llm.call(
        messages=messages,
        tools=[crew_tool_schema],
        available_functions=available_functions,
    )

    messages.append({"role": "assistant", "content": final_response})
    click.secho(f"\nAssistant: {final_response}\n", fg="green")


def generate_crew_tool_schema(crew_inputs: ChatInputs) -> dict[str, Any]:
    """Dynamically build a Littellm 'function' schema for the given crew.

    Args:
        crew_inputs: A ChatInputs object containing crew_description
                     and a list of input fields (each with a name & description).
    """
    properties = {}
    for field in crew_inputs.inputs:
        properties[field.name] = {
            "type": "string",
            "description": field.description or "No description provided",
        }

    required_fields = [field.name for field in crew_inputs.inputs]

    return {
        "type": "function",
        "function": {
            "name": crew_inputs.crew_name,
            "description": crew_inputs.crew_description or "No crew description",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            },
        },
    }


def run_crew_tool(crew: Crew, messages: list[LLMMessage], **kwargs: Any) -> str:
    """Run the crew using crew.kickoff(inputs=kwargs) and return the output.

    Args:
        crew: The crew instance to run.
        messages: The chat messages up to this point.
        **kwargs: The inputs collected from the user.

    Returns:
        The output from the crew's execution.
    """
    try:
        kwargs["crew_chat_messages"] = json.dumps(messages)
        crew_output = crew.kickoff(inputs=kwargs)
        return str(crew_output)

    except Exception as e:
        click.secho("An error occurred while running the crew:", fg="red")
        click.secho(str(e), fg="red")
        sys.exit(1)


def load_crew_and_name() -> tuple[Crew, str]:
    """Load the crew by importing the crew class from the user's project.

    Returns:
        A tuple containing the Crew instance and the name of the crew.
    """
    cwd = Path.cwd()

    pyproject_path = cwd / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in the current directory.")

    with pyproject_path.open("rb") as f:
        pyproject_data = tomli.load(f)

    project_name = pyproject_data["project"]["name"]
    folder_name = project_name

    crew_class_name = project_name.replace("_", " ").title().replace(" ", "")

    src_path = cwd / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    crew_module_name = f"{folder_name}.crew"
    try:
        crew_module = __import__(crew_module_name, fromlist=[crew_class_name])
    except ImportError as e:
        raise ImportError(
            f"Failed to import crew module {crew_module_name}: {e}"
        ) from e

    try:
        crew_class = getattr(crew_module, crew_class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Crew class {crew_class_name} not found in module {crew_module_name}"
        ) from e

    crew_instance = crew_class().crew()
    return crew_instance, crew_class_name


def generate_crew_chat_inputs(
    crew: Crew,
    crew_name: str,
    chat_llm: LLM | BaseLLM,
    generate_descriptions: bool = True,
) -> ChatInputs:
    """Generate the ChatInputs required for the crew by analyzing the tasks and agents.

    Args:
        crew: The crew object containing tasks and agents.
        crew_name: The name of the crew.
        chat_llm: The chat language model to use for AI calls.
        generate_descriptions: When True (default), use the LLM to generate
            input and crew descriptions. When False, skip all LLM calls and
            return static defaults. Production callers that invoke this at
            startup should pass ``False`` to avoid blocking on the LLM.

    Returns:
        An object containing the crew's name, description, and input fields.
    """
    required_inputs = fetch_required_inputs(crew)

    input_fields = []
    for input_name in required_inputs:
        if generate_descriptions:
            description = generate_input_description_with_ai(input_name, crew, chat_llm)
        else:
            description = DEFAULT_INPUT_DESCRIPTION
        input_fields.append(ChatInputField(name=input_name, description=description))

    if generate_descriptions:
        crew_description = generate_crew_description_with_ai(crew, chat_llm)
    else:
        crew_description = DEFAULT_CREW_DESCRIPTION

    return ChatInputs(
        crew_name=crew_name, crew_description=crew_description, inputs=input_fields
    )


def fetch_required_inputs(crew: Crew) -> set[str]:
    """Extract placeholders from the crew's tasks and agents.

    Args:
        crew: The crew object.

    Returns:
        A set of placeholder names.
    """
    return crew.fetch_inputs()


def generate_input_description_with_ai(
    input_name: str, crew: Crew, chat_llm: LLM | BaseLLM
) -> str:
    """Generate an input description using AI based on the context of the crew.

    Args:
        input_name: The name of the input placeholder.
        crew: The crew object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        A concise description of the input.
    """
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)}")

    for task in crew.tasks:
        if (
            f"{{{input_name}}}" in task.description
            or f"{{{input_name}}}" in task.expected_output
        ):
            task_description = placeholder_pattern.sub(
                lambda m: m.group(1), task.description or ""
            )
            expected_output = placeholder_pattern.sub(
                lambda m: m.group(1), task.expected_output or ""
            )
            context_texts.append(f"Task Description: {task_description}")
            context_texts.append(f"Expected Output: {expected_output}")
    for agent in crew.agents:
        if (
            f"{{{input_name}}}" in agent.role
            or f"{{{input_name}}}" in agent.goal
            or f"{{{input_name}}}" in agent.backstory
        ):
            agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role or "")
            agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal or "")
            agent_backstory = placeholder_pattern.sub(
                lambda m: m.group(1), agent.backstory or ""
            )
            context_texts.append(f"Agent Role: {agent_role}")
            context_texts.append(f"Agent Goal: {agent_goal}")
            context_texts.append(f"Agent Backstory: {agent_backstory}")

    context = "\n".join(context_texts)
    if not context:
        raise ValueError(f"No context found for input '{input_name}'.")

    prompt = (
        f"Based on the following context, write a concise description (15 words or less) of the input '{input_name}'.\n"
        "Provide only the description, without any extra text or labels. Do not include placeholders like '{topic}' in the description.\n"
        "Context:\n"
        f"{context}"
    )
    try:
        response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        click.secho(
            f"Warning: failed to generate input description for '{input_name}' "
            f"({exc}); using default.",
            fg="yellow",
        )
        return DEFAULT_INPUT_DESCRIPTION
    return str(response).strip()


def generate_crew_description_with_ai(crew: Crew, chat_llm: LLM | BaseLLM) -> str:
    """Generate a brief description of the crew using AI.

    Args:
        crew: The crew object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        A concise description of the crew's purpose (15 words or less).
    """
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)}")

    for task in crew.tasks:
        task_description = placeholder_pattern.sub(
            lambda m: m.group(1), task.description or ""
        )
        expected_output = placeholder_pattern.sub(
            lambda m: m.group(1), task.expected_output or ""
        )
        context_texts.append(f"Task Description: {task_description}")
        context_texts.append(f"Expected Output: {expected_output}")
    for agent in crew.agents:
        agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role or "")
        agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal or "")
        agent_backstory = placeholder_pattern.sub(
            lambda m: m.group(1), agent.backstory or ""
        )
        context_texts.append(f"Agent Role: {agent_role}")
        context_texts.append(f"Agent Goal: {agent_goal}")
        context_texts.append(f"Agent Backstory: {agent_backstory}")

    context = "\n".join(context_texts)
    if not context:
        raise ValueError("No context found for generating crew description.")

    prompt = (
        "Based on the following context, write a concise, action-oriented description (15 words or less) of the crew's purpose.\n"
        "Provide only the description, without any extra text or labels. Do not include placeholders like '{topic}' in the description.\n"
        "Context:\n"
        f"{context}"
    )
    try:
        response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        click.secho(
            f"Warning: failed to generate crew description ({exc}); using default.",
            fg="yellow",
        )
        return DEFAULT_CREW_DESCRIPTION
    return str(response).strip()
