import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Ensure the src directory is in the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from crewai.cli.cli import crewai
from crewai.cli import create_crew
from crewai.cli.constants import MODELS, ENV_VARS

# Mock provider data for testing
MOCK_PROVIDER_DATA = {
    'openai': {'models': ['gpt-4', 'gpt-3.5-turbo']}, 
    'google': {'models': ['gemini-pro']}, 
    'anthropic': {'models': ['claude-3-opus']}
}

MOCK_VALID_PROVIDERS = list(MOCK_PROVIDER_DATA.keys())

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def isolate_fs(monkeypatch):
    # Prevent tests from interacting with the actual filesystem or real env vars
    monkeypatch.setattr(Path, 'mkdir', lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, 'exists', lambda *args: False) # Assume folders don't exist initially
    monkeypatch.setattr(create_crew, 'load_env_vars', lambda *args: {}) # Start with empty env vars
    monkeypatch.setattr(create_crew, 'write_env_file', lambda *args, **kwargs: None)
    monkeypatch.setattr(create_crew, 'copy_template_files', lambda *args, **kwargs: None)

@patch('crewai.cli.create_crew.get_provider_data', return_value=MOCK_PROVIDER_DATA)
@patch('crewai.cli.create_crew.select_provider')
@patch('crewai.cli.create_crew.select_model')
@patch('click.prompt')
@patch('click.confirm', return_value=True) # Default to confirming prompts
def test_create_crew_with_valid_provider(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, runner):
    """Test `crewai create crew <name> --provider <valid_provider>`"""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew', '--provider', 'openai'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    assert "Using specified provider: Openai" in result.output
    mock_select_provider.assert_not_called() # Should not ask interactively
    # Depending on whether openai needs models/keys, check select_model/prompt calls
    assert "Crew 'testcrew' created successfully!" in result.output

@patch('crewai.cli.create_crew.get_provider_data', return_value=MOCK_PROVIDER_DATA)
@patch('crewai.cli.create_crew.select_provider', return_value='google') # Simulate user selecting google
@patch('crewai.cli.create_crew.select_model', return_value='gemini-pro')
@patch('click.prompt')
@patch('click.confirm', return_value=True)
def test_create_crew_with_invalid_provider(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, runner):
    """Test `crewai create crew <name> --provider <invalid_provider>`"""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew', '--provider', 'invalidprovider'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    assert "Warning: Specified provider 'invalidprovider' is not recognized." in result.output
    mock_select_provider.assert_called_once() # Should ask interactively
    # Check if subsequent steps for the selected provider (google) ran
    mock_select_model.assert_called_once()
    assert "Crew 'testcrew' created successfully!" in result.output

@patch('crewai.cli.create_crew.get_provider_data', return_value=MOCK_PROVIDER_DATA)
@patch('crewai.cli.create_crew.select_provider', return_value='anthropic') # Simulate user selecting anthropic
@patch('crewai.cli.create_crew.select_model', return_value='claude-3-opus')
@patch('click.prompt', return_value='sk-abc') # Simulate API key entry
@patch('click.confirm', return_value=True)
def test_create_crew_no_provider(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, runner):
    """Test `crewai create crew <name>`"""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    assert "Using specified provider:" not in result.output # Should not mention specified provider
    mock_select_provider.assert_called_once() # Should ask interactively
    mock_select_model.assert_called_once()
    # Check if prompt for API key was called (assuming anthropic needs one)
    if 'anthropic' in ENV_VARS and any('key_name' in d for d in ENV_VARS['anthropic']):
         mock_prompt.assert_called()
    assert "Crew 'testcrew' created successfully!" in result.output

@patch('crewai.cli.create_crew.get_provider_data')
@patch('crewai.cli.create_crew.select_provider')
@patch('crewai.cli.create_crew.select_model')
@patch('click.prompt')
@patch('click.confirm')
def test_create_crew_skip_provider(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, runner):
    """Test `crewai create crew <name> --skip_provider`"""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew', '--skip_provider'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    mock_get_data.assert_not_called()
    mock_select_provider.assert_not_called()
    mock_select_model.assert_not_called()
    mock_prompt.assert_not_called()
    mock_confirm.assert_not_called()
    assert "Crew 'testcrew' created successfully!" in result.output

@patch('crewai.cli.create_crew.load_env_vars', return_value={'OPENAI_API_KEY': 'existing_key'}) # Simulate existing env
@patch('crewai.cli.create_crew.get_provider_data', return_value=MOCK_PROVIDER_DATA)
@patch('crewai.cli.create_crew.select_provider', return_value='google') # Simulate selecting new provider
@patch('crewai.cli.create_crew.select_model', return_value='gemini-pro')
@patch('click.prompt')
@patch('click.confirm', return_value=True) # User confirms override
def test_create_crew_existing_override(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, mock_load_env, runner):
    """Test `crewai create crew <name>` with existing config and user overrides."""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    mock_confirm.assert_called_once_with(
        'Found existing environment variable configuration for Openai. Do you want to override it?'
    )
    mock_select_provider.assert_called_once() # Should ask for new provider after confirming override
    assert "Crew 'testcrew' created successfully!" in result.output

@patch('crewai.cli.create_crew.load_env_vars', return_value={'OPENAI_API_KEY': 'existing_key'}) # Simulate existing env
@patch('crewai.cli.create_crew.get_provider_data', return_value=MOCK_PROVIDER_DATA)
@patch('crewai.cli.create_crew.select_provider')
@patch('crewai.cli.create_crew.select_model')
@patch('click.prompt')
@patch('click.confirm', return_value=False) # User denies override
def test_create_crew_existing_keep(mock_confirm, mock_prompt, mock_select_model, mock_select_provider, mock_get_data, mock_load_env, runner):
    """Test `crewai create crew <name>` with existing config and user keeps it."""
    result = runner.invoke(crewai, ['create', 'crew', 'testcrew'])
    
    print(f"CLI Output:\n{result.output}") # Debug output
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput: {result.output}"
    mock_confirm.assert_called_once_with(
        'Found existing environment variable configuration for Openai. Do you want to override it?'
    )
    assert "Keeping existing provider configuration. Exiting provider setup." in result.output
    mock_select_provider.assert_not_called() # Should NOT ask for new provider
    assert "Crew 'testcrew' created successfully!" in result.output

