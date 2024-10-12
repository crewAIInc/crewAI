from pathlib import Path
import click
import requests
from collections import defaultdict
import json
import time
from urllib.parse import urlparse  # Added import

from crewai.cli.utils import copy_template

PROVIDERS = ['openai', 'anthropic', 'gemini', 'groq', 'ollama'] 

ENV_VARS = {
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    'gemini': ['GEMINI_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'ollama': ['FAKE_KEY'],
}

MODELS = {
    'openai': ['gpt-4', 'gpt-4o', 'gpt-4o-mini','o1-mini', 'o1-preview'],
    'anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
    'gemini': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-gemma-2-9b-it', 'gemini-gemma-2-27b-it'],
    'groq': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'llama-3.1-405b-reasoning', 'gemma2-9b-it', 'gemma-7b-it'],
    'ollama': ['llama3.1', 'mixtral'],
}

def create_crew(name, parent_folder=None):
    """Create a new crew."""
    provider = None
    model = None
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
            f"\tFolder {folder_name} already exists. Updating .env file...",
            fg="yellow",
        )
    
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

    json_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    current_time = time.time()
    data = {}          

    """
    Attempts to load provider data from the cache. If the cache is valid (i.e., not expired), it loads the data from the cache.
    If the cache is invalid or corrupted, it fetches the provider data from the web.
    """
    if cache_file.exists() and (current_time - cache_file.stat().st_mtime) < cache_expiry:
        click.secho("Loading provider data from cache...", fg="cyan")
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            click.secho("Provider data loaded from cache successfully.", fg="green")
        except json.JSONDecodeError:
            click.secho("Cache is corrupted. Fetching provider data from the web...", fg="yellow")
            data = {}
    else:
        click.secho("Cache expired or not found. Fetching provider data from the web...", fg="cyan")
        data = {}

    if not data:
        try:
            with requests.get(json_url, stream=True, timeout=10) as response:
                response.raise_for_status()
                total_size = response.headers.get('content-length')
                total_size = int(total_size) if total_size else None
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
            click.secho("Provider data fetched and cached successfully.", fg="green")
        except requests.RequestException as e:
            click.secho(f"Error fetching provider data: {e}", fg="red")
            return
        except json.JSONDecodeError:
            click.secho("Error parsing provider data. Invalid JSON format.", fg="red")
            return

    provider_models = defaultdict(list)
    for model_name, properties in data.items():
        provider_full = properties.get("litellm_provider")
        if provider_full:
            provider_key = provider_full.strip().lower() 

            if 'http' in provider_key:
                click.secho(f"Skipping invalid provider entry: '{provider_full}'", fg="yellow")
                continue

            if provider_key and provider_key != 'other':  
                provider_models[provider_key].append(model_name)

    predefined_providers = [p.lower() for p in PROVIDERS]
    all_providers = set(predefined_providers)
    all_providers.update(provider_models.keys())

    all_providers = sorted(all_providers)

    if provider:
        provider_lower = provider.lower()
        if provider_lower == 'other':
            all_providers = sorted(provider_models.keys())
            if not all_providers:
                click.secho("No additional providers available.", fg="yellow")
                return
            click.secho("Select a provider from the full list:", fg="cyan")
            for index, provider_name in enumerate(all_providers, start=1):
                click.secho(f"{index}. {provider_name}", fg="cyan")
            
            while True:
                try:
                    selected_index = click.prompt(
                        "Enter the number of your choice", type=int
                    ) - 1
                    if 0 <= selected_index < len(all_providers):
                        provider = all_providers[selected_index]
                        break
                    else:
                        click.secho("Invalid selection. Please try again.", fg="red")
                except click.exceptions.Abort:
                    click.secho("Operation aborted by the user.", fg="red")
                    return
        else:

            if provider_lower not in provider_models and provider_lower not in [p.lower() for p in PROVIDERS]:
                click.secho(f"Invalid provider: {provider}", fg="red")
                return
    else:
        click.secho("Select a provider to set up:", fg="cyan")
        for index, provider_name in enumerate(PROVIDERS + ['other'], start=1):
            click.secho(f"{index}. {provider_name}", fg="cyan")
        
        while True:
            try:
                selected_index = click.prompt(
                    "Enter the number of your choice", type=int
                ) - 1
                if 0 <= selected_index < len(PROVIDERS) + 1:
                    selected_provider = (PROVIDERS + ['other'])[selected_index]
                    if selected_provider.lower() == 'other':
                        if not all_providers:
                            click.secho("No additional providers available.", fg="yellow")
                            return
                        click.secho("Select a provider from the full list:", fg="cyan")
                        for idx, provider_name in enumerate(all_providers, start=1):
                            display_name = provider_name.capitalize() 
                            click.secho(f"{idx}. {display_name}", fg="cyan")
                        
                        while True:
                            try:
                                selected_sub_index = click.prompt(
                                    "Enter the number of your choice", type=int
                                ) - 1
                                if 0 <= selected_sub_index < len(all_providers):
                                    provider = all_providers[selected_sub_index]
                                    break
                                else:
                                    click.secho("Invalid selection. Please try again.", fg="red")
                            except click.exceptions.Abort:
                                click.secho("Operation aborted by the user.", fg="red")
                                return
                        break 
                    else:
                        provider = selected_provider.lower()
                        break
                else:
                    click.secho("Invalid selection. Please try again.", fg="red")
            except click.exceptions.Abort:
                click.secho("Operation aborted by the user.", fg="red")
                return

    provider = provider.strip().lower()

    if provider in predefined_providers:
        available_models = MODELS.get(provider, [])
    else:
        available_models = provider_models.get(provider, [])
    
    if not available_models:
        click.secho(f"No models available for provider '{provider}'.", fg="red")
        click.secho(f"Available providers: {list(provider_models.keys())}", fg="yellow")
        return

    if model:
        if model not in available_models:
            click.secho(f"Invalid model '{model}' for provider '{provider}'.", fg="red")
            return
    else:
        click.secho(f"Select a model to use for {provider.capitalize()}:", fg="cyan")
        for idx, model_name in enumerate(available_models, start=1):
            click.secho(f"{idx}. {model_name}", fg="cyan")
        
        while True:
            try:
                selected_model_index = click.prompt(
                    "Enter the number of your choice", type=int
                ) - 1
                if 0 <= selected_model_index < len(available_models):
                    model = available_models[selected_model_index]
                    break
                else:
                    click.secho("Invalid selection. Please try again.", fg="red")
            except click.exceptions.Abort:
                click.secho("Operation aborted by the user.", fg="red")
                return

    if provider.lower() in ENV_VARS:
        api_key_var = ENV_VARS[provider.lower()][0]
    else:
        api_key_var = f"{provider.upper()}_API_KEY"

    if api_key_var not in env_vars:
        env_vars[api_key_var] = click.prompt(
            f"Enter your {provider} API key", type=str, hide_input=True
        )
    else:
        click.secho(f"API key already exists for {provider}.", fg="yellow")

    if model:
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
