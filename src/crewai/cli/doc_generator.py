import os
import yaml
import logging


def is_project_root():
    """
    Check if the current directory is the root of a CrewAI project.

    Returns:
        bool: True if in project root, False otherwise.
    """
    # Check for key indicators of a CrewAI project root
    indicators = ["pyproject.toml", "poetry.lock", "src"]
    return all(os.path.exists(indicator) for indicator in indicators)


def generate_documentation(output_file, format):
    """
    Generate documentation for the current CrewAI project setup.

    Args:
        output_file (str): The path and filename where the generated documentation
                           will be saved.
        format (str): The desired output format for the documentation.
                      Supported values currently 'markdown'.

    Returns:
        None: The function writes the generated documentation to the specified
              output file and doesn't return any value.

    Raises:
        ValueError: If not in the project root or if an unsupported output format is specified.
    """
    if not is_project_root():
        raise ValueError(
            "Not in the root of a CrewAI project."
        )

    # Load the current project configuration
    config = load_crew_configuration()

    if config is None:
        logging.error("Failed to load crew configuration. Exiting.")
        return

    if format == "markdown":
        content = generate_markdown(config)
    else:
        raise ValueError(f"Unsupported output format: {format}")

    with open(output_file, "w") as f:
        f.write(content)

    logging.info(f"Documentation generated and saved to {output_file}")


def find_config_dir():
    """
    Find the configuration directory based on the project structure.

    This function attempts to locate the configuration directory for a CrewAI project
    by assuming a standard project structure. It starts from the current working
    directory and constructs an expected path to the config directory.

    Returns:
        str or None: The path to the configuration directory if found, None otherwise.

    The function performs the following steps:
    1. Gets the current working directory.
    2. Extracts the project name from the current directory path.
    3. Constructs the expected config path using the project structure convention.
    4. Checks if the expected config directory exists.
    5. Returns the path if found, or None if not found.

    Logging:
        - Logs debug information about the search process.
        - Logs the starting directory, the checked path, and the result of the search.

    Note:
        This function assumes a specific project structure where the config
        directory is located at 'src/<project_name>/config' relative to the
        project root.
    """
    current_dir = os.getcwd()
    logging.debug(f"Starting search from: {current_dir}")

    # Split the path to get the project name
    path_parts = current_dir.split(os.path.sep)
    project_name = path_parts[-1]

    # Construct the expected config path
    expected_config_path = os.path.join(current_dir, "src", project_name, "config")

    logging.debug(f"Checking for config directory: {expected_config_path}")

    if os.path.isdir(expected_config_path):
        logging.debug(f"Found config directory: {expected_config_path}")
        return expected_config_path

    logging.debug("Config directory not found in the expected location")
    return None


def load_crew_configuration():
    """
    Load the crew configuration from YAML files.

    This function attempts to find the configuration directory and load the agents
    and tasks configurations from their respective YAML files.

    Returns:
        dict or None: A dictionary containing 'agents' and 'tasks' configurations
                      if successful, None if there was an error.

    The function performs the following steps:
    1. Finds the configuration directory using find_config_dir().
    2. Constructs paths to agents.yaml and tasks.yaml files.
    3. Checks if both files exist.
    4. Loads and parses the YAML content of both files.
    5. Returns a dictionary with the parsed configurations.

    Logging:
        - Logs an error if the configuration directory is not found.
        - Logs an error if either agents.yaml or tasks.yaml is not found.

    Note:
        This function assumes that the configuration files are named 'agents.yaml'
        and 'tasks.yaml' and are located in the directory returned by find_config_dir().
    """
    config_dir = find_config_dir()
    if not config_dir:
        logging.error(
            "Configuration directory not found. Make sure you're in the root of your CrewAI project."
        )
        return None

    agents_file = os.path.join(config_dir, "agents.yaml")
    tasks_file = os.path.join(config_dir, "tasks.yaml")

    if not os.path.exists(agents_file) or not os.path.exists(tasks_file):
        logging.error(f"agents.yaml or tasks.yaml not found in {config_dir}")
        return None

    with open(agents_file, "r") as f:
        agents_config = yaml.safe_load(f)

    with open(tasks_file, "r") as f:
        tasks_config = yaml.safe_load(f)

    return {"agents": agents_config, "tasks": tasks_config}


def generate_markdown(config):
    """
    Generate Markdown documentation for the CrewAI project configuration.

    This function takes the parsed configuration dictionary and generates
    a formatted Markdown string containing documentation for the project's
    agents and tasks.

    Args:
        config (dict): A dictionary containing the parsed configuration
                       with 'agents' and 'tasks' keys.

    Returns:
        str: A formatted Markdown string containing the project documentation.
             If the input config is None, it returns an error message.

    The generated Markdown includes:
    1. A title for the project documentation.
    2. A section for Agents, listing each agent's name, role, goal, and backstory.
    3. A section for Tasks, listing each task's name, description, expected output,
       and assigned agent.

    Each piece of information is wrapped in code blocks for better readability
    in rendered Markdown.

    Note:
        This function assumes that the config dictionary has the correct structure
        with 'agents' and 'tasks' keys, each containing nested dictionaries of
        agent and task information respectively.
    """
    if config is None:
        return "# Error: No crew configuration available"

    md = "# CrewAI Project Documentation\n\n"

    md += "## Agents\n\n"
    for agent_name, agent_data in config["agents"].items():
        md += f"### \n```\n{agent_name}\n```\n"
        md += f"Role: \n```\n{agent_data.get('role', 'Not specified')}\n```\n"
        md += f"Goal: \n```\n{agent_data.get('goal', 'Not specified')}\n```\n"
        md += f"Backstory: \n```\n{agent_data.get('backstory', 'Not specified')}\n```\n"
        md += f""

    md += "## Tasks\n\n"
    for task_name, task_data in config["tasks"].items():
        md += f"### {task_name}\n"
        md += f"Description: \n```\n{task_data.get('description', 'Not specified')}\n```\n"
        md += f"Expected Output: \n```\n{task_data.get('expected_output', 'Not specified')}\n```\n"
        md += f"Assigned Agent: \n```\n{task_data.get('agent', 'Not assigned')}\n```\n"

    return md
