from collections import defaultdict
from pathlib import Path
import click
import json
import requests
import time
from crewai.cli.utils import copy_template

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

PROVIDERS = ['openai', 'anthropic', 'gemini', 'groq', 'ollama']

ENV_VARS = {
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    'gemini': ['GEMINI_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'ollama': ['FAKE_KEY'],
}

MODELS = {
    'openai': ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'],
    'anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
    'gemini': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-gemma-2-9b-it', 'gemini-gemma-2-27b-it'],
    'groq': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'llama-3.1-405b-reasoning', 'gemma2-9b-it', 'gemma-7b-it'],
    'ollama': ['llama3.1', 'mixtral'],
}

def load_provider_data(cache_file, cache_expiry):
    current_time = time.time()
    if cache_file.exists() and (current_time - cache_file.stat().st_mtime) < cache_expiry:
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            click.secho("Cache is corrupted. Fetching provider data from the web...", fg="yellow")
    else:
        click.secho("Cache expired or not found. Fetching provider data from the web...", fg="cyan")

    try:
        response = requests.get(JSON_URL, stream=True, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        data_chunks = []
        with click.progressbar(length=total_size, label='Downloading', show_pos=True) as progress_bar:
            for chunk in response.iter_content(block_size):
                if chunk:
                    data_chunks.append(chunk)
                    progress_bar.update(len(chunk))
        data_content = b''.join(data_chunks)
        data = json.loads(data_content.decode('utf-8'))
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except requests.RequestException as e:
        click.secho(f"Error fetching provider data: {e}", fg="red")
        return None
    except json.JSONDecodeError:
        click.secho("Error parsing provider data. Invalid JSON format.", fg="red")
        return None

def select_choice(prompt_message, choices):
    click.secho(prompt_message, fg="cyan")
    for idx, choice in enumerate(choices, start=1):
        click.secho(f"{idx}. {choice}", fg="cyan")
    try:
        selected_index = click.prompt("Enter the number of your choice", type=int) - 1
    except click.exceptions.Abort:
        click.secho("Operation aborted by the user.", fg="red")
        return None

    if not (0 <= selected_index < len(choices)):
        click.secho("Invalid selection.", fg="red")
        return None

    return choices[selected_index]

def select_provider(provider, all_providers, predefined_providers):
    provider = provider.lower() if provider else None

    # Early return if the provided provider is invalid
    if provider and provider not in all_providers and provider != 'other':
        click.secho(f"Invalid provider: {provider}", fg="red")
        return None

    # If no provider is given, prompt the user
    if not provider:
        options = predefined_providers + ['other']
        provider = select_choice("Select a provider to set up:", options)
        if not provider:
            return None

    # Handle 'other' option
    if provider == 'other':
        if not all_providers:
            click.secho("No additional providers available.", fg="yellow")
            return None
        provider = select_choice("Select a provider from the full list:", all_providers)
        if not provider:
            return None

    return provider.lower()

    return selected_provider.lower()

def select_model(provider, predefined_providers, MODELS, provider_models):
    provider = provider.lower()
    if provider in predefined_providers:
        available_models = MODELS.get(provider, [])
    else:
        available_models = provider_models.get(provider, [])

    if not available_models:
        click.secho(f"No models available for provider '{provider}'.", fg="red")
        click.secho(f"Available providers: {list(provider_models.keys())}", fg="yellow")
        return None

    selected_model = select_choice(f"Select a model to use for {provider.capitalize()}:", available_models)
    if not selected_model:
        return None
    return selected_model

def create_folder_structure(name, parent_folder=None):
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    if parent_folder:
        folder_path = Path(parent_folder) / folder_name
    else:
        folder_path = Path(folder_name)

    click.secho(
        f"Creating {'crew' if parent_folder else 'folder'} {folder_name}...",
        fg="green",
        bold=True,
    )

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        (folder_path / "tests").mkdir(exist_ok=True)
        if not parent_folder:
            (folder_path / "src" / folder_name).mkdir(parents=True)
            (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
            (folder_path / "src" / folder_name / "config").mkdir(parents=True)
    else:
        click.secho(
            f"\tFolder {folder_name} already exists.",
            fg="yellow",
        )

    return folder_path, folder_name, class_name

def create_crew(name, parent_folder=None):
    folder_path, folder_name, class_name = create_folder_structure(name, parent_folder)
    
    env_file_path = folder_path / ".env"

    env_vars = {}
    if env_file_path.exists():
        with open(env_file_path, "r") as file:
            for line in file:
                key_value = line.strip().split('=', 1)
                if len(key_value) == 2:
                    env_vars[key_value[0]] = key_value[1]

    cache_dir = Path.home() / '.crewai'
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'provider_cache.json'
    cache_expiry = 24 * 3600

    provider_models = get_provider_data(cache_file, cache_expiry)
    if not provider_models:
        return

    predefined_providers = [p.lower() for p in PROVIDERS]
    provider_models = {k.lower(): v for k, v in provider_models.items()}

    all_providers = set(predefined_providers)
    all_providers.update(provider_models.keys())
    all_providers = sorted(all_providers)

    selected_provider = select_provider(None, all_providers, predefined_providers)
    if not selected_provider:
        return
    provider = selected_provider.lower()

    selected_model = select_model(provider, predefined_providers, MODELS, provider_models)
    if not selected_model:
        return
    model = selected_model

    if provider in predefined_providers:
        api_key_var = ENV_VARS[provider][0]
    else:
        api_key_var = click.prompt(
            f"Enter the environment variable name for your {provider.capitalize()} API key",
            type=str
        )

    if api_key_var not in env_vars:
        try:
            env_vars[api_key_var] = click.prompt(
                f"Enter your {provider.capitalize()} API key", type=str, hide_input=True
            )
        except click.exceptions.Abort:
            click.secho("Operation aborted by the user.", fg="red")
            return
    else:
        click.secho(f"API key already exists for {provider.capitalize()}.", fg="yellow")

    env_vars['MODEL'] = model
    click.secho(f"Selected model: {model}", fg="green")

    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    root_template_files = (
        [".gitignore", "pyproject.toml", "README.md"] if not parent_folder else []
    )
    tools_template_files = ["tools/custom_tool.py", "tools/__init__.py"]
    config_template_files = ["config/agents.yaml", "config/tasks.yaml"]
    src_template_files = (
        ["__init__.py", "main.py", "crew.py"] if not parent_folder else ["crew.py"]
    )

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = folder_path / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    src_folder = folder_path / "src" / folder_name if not parent_folder else folder_path

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = src_folder / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    if not parent_folder:
        for file_name in tools_template_files + config_template_files:
            src_file = templates_dir / file_name
            dst_file = src_folder / file_name
            copy_template(src_file, dst_file, name, class_name, folder_name)

    click.secho(f"Crew {name} created successfully!", fg="green", bold=True)

def get_provider_data(cache_file, cache_expiry):
    data = load_provider_data(cache_file, cache_expiry)
    if not data:
        return None

    provider_models = defaultdict(list)
    for model_name, properties in data.items():
        provider_full = properties.get("litellm_provider")
        if provider_full:
            provider_key = provider_full.strip().lower()
            if 'http' in provider_key:
                continue
            if provider_key and provider_key != 'other':
                provider_models[provider_key].append(model_name)
    return provider_models