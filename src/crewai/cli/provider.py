import json
import time
from collections import defaultdict
from pathlib import Path

import click
import requests

from crewai.cli.constants import JSON_URL, MODELS, PROVIDERS


def select_choice(prompt_message, choices):
    """
    Presents a list of choices to the user and prompts them to select one.

    Args:
    - prompt_message (str): The message to display to the user before presenting the choices.
    - choices (list): A list of options to present to the user.

    Returns:
    - str: The selected choice from the list, or None if the user chooses to quit.
    """

    provider_models = get_provider_data()
    if not provider_models:
        return
    click.secho(prompt_message, fg="cyan")
    for idx, choice in enumerate(choices, start=1):
        click.secho(f"{idx}. {choice}", fg="cyan")
    click.secho("q. Quit", fg="cyan")

    while True:
        choice = click.prompt(
            "Enter the number of your choice or 'q' to quit", type=str
        )

        if choice.lower() == "q":
            return None

        try:
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(choices):
                return choices[selected_index]
        except ValueError:
            pass

        click.secho(
            "Invalid selection. Please select a number between 1 and 6 or 'q' to quit.",
            fg="red",
        )


def select_provider(provider_models):
    """
    Presents a list of providers to the user and prompts them to select one.

    Args:
    - provider_models (dict): A dictionary of provider models.

    Returns:
    - str: The selected provider
    - None: If user explicitly quits
    """
    predefined_providers = [p.lower() for p in PROVIDERS]
    all_providers = sorted(set(predefined_providers + list(provider_models.keys())))

    provider = select_choice(
        "Select a provider to set up:", predefined_providers + ["other"]
    )
    if provider is None:  # User typed 'q'
        return None

    if provider == "other":
        provider = select_choice("Select a provider from the full list:", all_providers)
        if provider is None:  # User typed 'q'
            return None

    return provider.lower() if provider else False


def select_model(provider, provider_models):
    """
    Presents a list of models for a given provider to the user and prompts them to select one.

    Args:
    - provider (str): The provider for which to select a model.
    - provider_models (dict): A dictionary of provider models.

    Returns:
    - str: The selected model, or None if the operation is aborted or an invalid selection is made.
    """
    predefined_providers = [p.lower() for p in PROVIDERS]

    if provider in predefined_providers:
        available_models = MODELS.get(provider, [])
    else:
        available_models = provider_models.get(provider, [])

    if not available_models:
        click.secho(f"No models available for provider '{provider}'.", fg="red")
        return None

    selected_model = select_choice(
        f"Select a model to use for {provider.capitalize()}:", available_models
    )
    return selected_model


def load_provider_data(cache_file, cache_expiry):
    """
    Loads provider data from a cache file if it exists and is not expired. If the cache is expired or corrupted, it fetches the data from the web.

    Args:
    - cache_file (Path): The path to the cache file.
    - cache_expiry (int): The cache expiry time in seconds.

    Returns:
    - dict or None: The loaded provider data or None if the operation fails.
    """
    current_time = time.time()
    if (
        cache_file.exists()
        and (current_time - cache_file.stat().st_mtime) < cache_expiry
    ):
        data = read_cache_file(cache_file)
        if data:
            return data
        click.secho(
            "Cache is corrupted. Fetching provider data from the web...", fg="yellow"
        )
    else:
        click.secho(
            "Cache expired or not found. Fetching provider data from the web...",
            fg="cyan",
        )
    return fetch_provider_data(cache_file)


def read_cache_file(cache_file):
    """
    Reads and returns the JSON content from a cache file. Returns None if the file contains invalid JSON.

    Args:
    - cache_file (Path): The path to the cache file.

    Returns:
    - dict or None: The JSON content of the cache file or None if the JSON is invalid.
    """
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def validate_response(response):
    """
    Validates the response content type.

    Args:
    - response: The HTTP response object.

    Returns:
    - bool: True if the content type is valid, False otherwise.
    """
    content_type = response.headers.get('content-type', '').lower()
    valid_types = ['application/json', 'application/json; charset=utf-8']
    if not any(content_type.startswith(t) for t in valid_types):
        click.secho(f"Error: Expected JSON response but got {content_type}", fg="red")
        return False
    return True

def handle_provider_error(error, error_type="fetch"):
    """
    Handles provider data errors with consistent messaging.

    Args:
    - error: The error object.
    - error_type: Type of error for message selection.

    Returns:
    - None: Always returns None to indicate error.
    """
    error_messages = {
        "fetch": "Error fetching provider data",
        "parse": "Error parsing provider data",
        "unexpected": "Unexpected error"
    }
    base_message = error_messages.get(error_type, "Error")
    click.secho(f"{base_message}: {str(error)}", fg="red")
    return None

def invalidate_cache(cache_file):
    """
    Invalidates the cache file in error scenarios.

    Args:
    - cache_file: Path to the cache file.
    """
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except OSError as e:
        click.secho(f"Warning: Could not clear cache file: {e}", fg="yellow")

def fetch_provider_data(cache_file):
    """
    Fetches provider data from a specified URL and caches it to a file.

    Args:
    - cache_file (Path): The path to the cache file.

    Returns:
    - dict or None: The fetched provider data or None if the operation fails.
    """
    try:
        response = requests.get(JSON_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        if not validate_response(response):
            invalidate_cache(cache_file)
            return None
            
        data = download_data(response)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except requests.RequestException as e:
        invalidate_cache(cache_file)
        return handle_provider_error(e, "fetch")
    except json.JSONDecodeError as e:
        invalidate_cache(cache_file)
        return handle_provider_error(e, "parse")
    except Exception as e:
        invalidate_cache(cache_file)
        return handle_provider_error(e, "unexpected")


def download_data(response):
    """
    Downloads data from a given HTTP response and returns the JSON content.

    Args:
    - response (requests.Response): The HTTP response object.

    Returns:
    - dict: The JSON content of the response.
    """
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    data_chunks = []
    with click.progressbar(
        length=total_size, label="Downloading", show_pos=True
    ) as progress_bar:
        for chunk in response.iter_content(block_size):
            if chunk:
                data_chunks.append(chunk)
                progress_bar.update(len(chunk))
    data_content = b"".join(data_chunks)
    return json.loads(data_content.decode("utf-8"))


def get_provider_data():
    """
    Retrieves provider data from a cache file, filters out models based on provider criteria, and returns a dictionary of providers mapped to their models.

    Returns:
    - dict: A dictionary of providers mapped to their models, using default providers if fetch fails.
    """
    cache_dir = Path.home() / ".crewai"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "provider_cache.json"
    cache_expiry = 24 * 3600

    data = load_provider_data(cache_file, cache_expiry)
    if not data:
        # Return default providers if fetch fails
        return {provider.lower(): MODELS.get(provider.lower(), []) 
                for provider in PROVIDERS}

    provider_models = defaultdict(list)
    for model_name, properties in data.items():
        provider = properties.get("litellm_provider", "").strip().lower()
        if "http" in provider or provider == "other":
            continue
        if provider:
            provider_models[provider].append(model_name)
    return provider_models
