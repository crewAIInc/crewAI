import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import click
import tomli

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.cli.fetch_chat_llm import fetch_chat_llm
from crewai.cli.fetch_crew_inputs import fetch_crew_inputs
from crewai.crew import Crew
from crewai.task import Task
from crewai.types.crew_chat import ChatInputField, ChatInputs
from crewai.utilities.llm_utils import create_llm


def run_chat():
    """
    Runs an interactive chat loop using the Crew's chat LLM with function calling.
    Incorporates crew_name, crew_description, and input fields to build a tool schema.
    Exits if crew_name or crew_description are missing.
    """
    crew, crew_name = load_crew_and_name()
    click.secho("\nFetching the Chat LLM...", fg="cyan")
    try:
        chat_llm = create_llm(crew.chat_llm)
    except Exception as e:
        click.secho(f"Failed to retrieve Chat LLM: {e}", fg="red")
        return
    if not chat_llm:
        click.secho("No valid Chat LLM returned. Exiting.", fg="red")
        return

    # Generate crew chat inputs automatically
    crew_chat_inputs = generate_crew_chat_inputs(crew, crew_name, chat_llm)
    print("crew_inputs:", crew_chat_inputs)

    # Generate a tool schema from the crew inputs
    crew_tool_schema = generate_crew_tool_schema(crew_chat_inputs)
    print("crew_tool_schema:", crew_tool_schema)

    # Build initial system message
    required_fields_str = (
        ", ".join(
            f"{field.name} (desc: {field.description or 'n/a'})"
            for field in crew_chat_inputs.inputs
        )
        or "(No required fields detected)"
    )

    system_message = (
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
        "If you are ever unsure about a user's request or need clarification, ask the user for more information."
        f"\nCrew Name: {crew_chat_inputs.crew_name}"
        f"\nCrew Description: {crew_chat_inputs.crew_description}"
    )

    messages = [
        {"role": "system", "content": system_message},
    ]

    # Create a wrapper function that captures 'crew' and 'messages' from the enclosing scope
    def run_crew_tool_with_messages(**kwargs):
        return run_crew_tool(crew, messages, **kwargs)

    # Prepare available_functions with the wrapper function
    available_functions = {
        crew_chat_inputs.crew_name: run_crew_tool_with_messages,
    }

    click.secho(
        "\nEntering an interactive chat loop with function-calling.\n"
        "Type 'exit' or Ctrl+C to quit.\n",
        fg="cyan",
    )

    # Main chat loop
    while True:
        try:
            user_input = click.prompt("You", type=str)
            if user_input.strip().lower() in ["exit", "quit"]:
                click.echo("Exiting chat. Goodbye!")
                break

            # Append user message
            messages.append({"role": "user", "content": user_input})

            # Invoke the LLM, passing tools and available_functions
            final_response = chat_llm.call(
                messages=messages,
                tools=[crew_tool_schema],
                available_functions=available_functions,
            )

            # Append assistant's reply
            messages.append({"role": "assistant", "content": final_response})

            # Display assistant's reply
            click.secho(f"\nAssistant: {final_response}\n", fg="green")

        except KeyboardInterrupt:
            click.echo("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            click.secho(f"An error occurred: {e}", fg="red")
            break


def generate_crew_tool_schema(crew_inputs: ChatInputs) -> dict:
    """
    Dynamically build a Littellm 'function' schema for the given crew.

    crew_name: The name of the crew (used for the function 'name').
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


def run_crew_tool(crew: Crew, messages: List[Dict[str, str]], **kwargs):
    """
    Runs the crew using crew.kickoff(inputs=kwargs) and returns the output.

    Args:
        crew (Crew): The crew instance to run.
        messages (List[Dict[str, str]]): The chat messages up to this point.
        **kwargs: The inputs collected from the user.

    Returns:
        str: The output from the crew's execution.

    Raises:
        SystemExit: Exits the chat if an error occurs during crew execution.
    """
    try:
        # Serialize 'messages' to JSON string before adding to kwargs
        kwargs['crew_chat_messages'] = json.dumps(messages)

        # Run the crew with the provided inputs
        crew_output = crew.kickoff(inputs=kwargs)

        # Convert CrewOutput to a string to send back to the user
        result = str(crew_output)

        return result
    except Exception as e:
        # Exit the chat and show the error message
        click.secho("An error occurred while running the crew:", fg="red")
        click.secho(str(e), fg="red")
        sys.exit(1)


def load_crew_and_name() -> Tuple[Crew, str]:
    """
    Loads the crew by importing the crew class from the user's project.
    
    Returns:
        Tuple[Crew, str]: A tuple containing the Crew instance and the name of the crew.
    """
    # Get the current working directory
    cwd = Path.cwd()

    # Path to the pyproject.toml file
    pyproject_path = cwd / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in the current directory.")

    # Load the pyproject.toml file using 'tomli'
    with pyproject_path.open("rb") as f:
        pyproject_data = tomli.load(f)

    # Get the project name from the 'project' section
    project_name = pyproject_data["project"]["name"]
    folder_name = project_name

    # Derive the crew class name from the project name
    # E.g., if project_name is 'my_project', crew_class_name is 'MyProject'
    crew_class_name = project_name.replace("_", " ").title().replace(" ", "")

    # Add the 'src' directory to sys.path
    src_path = cwd / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Import the crew module
    crew_module_name = f"{folder_name}.crew"
    try:
        crew_module = __import__(crew_module_name, fromlist=[crew_class_name])
    except ImportError as e:
        raise ImportError(f"Failed to import crew module {crew_module_name}: {e}")

    # Get the crew class from the module
    try:
        crew_class = getattr(crew_module, crew_class_name)
    except AttributeError:
        raise AttributeError(
            f"Crew class {crew_class_name} not found in module {crew_module_name}"
        )

    # Instantiate the crew
    crew_instance = crew_class().crew()
    return crew_instance, crew_class_name


def generate_crew_chat_inputs(crew: Crew, crew_name: str, chat_llm) -> ChatInputs:
    """
    Generates the ChatInputs required for the crew by analyzing the tasks and agents.

    Args:
        crew (Crew): The crew object containing tasks and agents.
        crew_name (str): The name of the crew.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        ChatInputs: An object containing the crew's name, description, and input fields.
    """
    # Extract placeholders from tasks and agents
    required_inputs = fetch_required_inputs(crew)

    # Generate descriptions for each input using AI
    input_fields = []
    for input_name in required_inputs:
        description = generate_input_description_with_ai(input_name, crew, chat_llm)
        input_fields.append(ChatInputField(name=input_name, description=description))

    # Generate crew description using AI
    crew_description = generate_crew_description_with_ai(crew, chat_llm)

    return ChatInputs(
        crew_name=crew_name,
        crew_description=crew_description,
        inputs=input_fields
    )


def fetch_required_inputs(crew: Crew) -> Set[str]:
    """
    Extracts placeholders from the crew's tasks and agents.

    Args:
        crew (Crew): The crew object.

    Returns:
        Set[str]: A set of placeholder names.
    """
    placeholder_pattern = re.compile(r"\{(.+?)\}")
    required_inputs: Set[str] = set()

    # Scan tasks
    for task in crew.tasks:
        text = f"{task.description or ''} {task.expected_output or ''}"
        required_inputs.update(placeholder_pattern.findall(text))

    # Scan agents
    for agent in crew.agents:
        text = f"{agent.role or ''} {agent.goal or ''} {agent.backstory or ''}"
        required_inputs.update(placeholder_pattern.findall(text))

    return required_inputs


def generate_input_description_with_ai(input_name: str, crew: Crew, chat_llm) -> str:
    """
    Generates an input description using AI based on the context of the crew.

    Args:
        input_name (str): The name of the input placeholder.
        crew (Crew): The crew object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        str: A concise description of the input.
    """
    # Gather context from tasks and agents where the input is used
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)\}")

    for task in crew.tasks:
        if f"{{{input_name}}}" in task.description or f"{{{input_name}}}" in task.expected_output:
            # Replace placeholders with input names
            task_description = placeholder_pattern.sub(lambda m: m.group(1), task.description)
            expected_output = placeholder_pattern.sub(lambda m: m.group(1), task.expected_output)
            context_texts.append(f"Task Description: {task_description}")
            context_texts.append(f"Expected Output: {expected_output}")
    for agent in crew.agents:
        if f"{{{input_name}}}" in agent.role or f"{{{input_name}}}" in agent.goal or f"{{{input_name}}}" in agent.backstory:
            # Replace placeholders with input names
            agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role)
            agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal)
            agent_backstory = placeholder_pattern.sub(lambda m: m.group(1), agent.backstory)
            context_texts.append(f"Agent Role: {agent_role}")
            context_texts.append(f"Agent Goal: {agent_goal}")
            context_texts.append(f"Agent Backstory: {agent_backstory}")

    context = "\n".join(context_texts)
    if not context:
        # If no context is found for the input, raise an exception as per instruction
        raise ValueError(f"No context found for input '{input_name}'.")

    prompt = (
        f"Based on the following context, write a concise description (15 words or less) of the input '{input_name}'.\n"
        "Provide only the description, without any extra text or labels. Do not include placeholders like '{topic}' in the description.\n"
        "Context:\n"
        f"{context}"
    )
    response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    description = response.strip()

    return description


def generate_crew_description_with_ai(crew: Crew, chat_llm) -> str:
    """
    Generates a brief description of the crew using AI.

    Args:
        crew (Crew): The crew object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        str: A concise description of the crew's purpose (15 words or less).
    """
    # Gather context from tasks and agents
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)\}")

    for task in crew.tasks:
        # Replace placeholders with input names
        task_description = placeholder_pattern.sub(lambda m: m.group(1), task.description)
        expected_output = placeholder_pattern.sub(lambda m: m.group(1), task.expected_output)
        context_texts.append(f"Task Description: {task_description}")
        context_texts.append(f"Expected Output: {expected_output}")
    for agent in crew.agents:
        # Replace placeholders with input names
        agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role)
        agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal)
        agent_backstory = placeholder_pattern.sub(lambda m: m.group(1), agent.backstory)
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
    response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    crew_description = response.strip()

    return crew_description
