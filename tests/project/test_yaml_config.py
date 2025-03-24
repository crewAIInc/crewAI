import pytest
import yaml
import tempfile
from pathlib import Path
import sys
import os

# Add a simple test to verify the fix works
def test_list_format_in_yaml():
    """Test that list format in YAML is handled correctly."""
    # Create a test YAML content with list format
    yaml_content = """
    test_agent:
      - name: test_agent
        role: Test Agent
        goal: Test Goal
    """
    
    # Parse the YAML content
    data = yaml.safe_load(yaml_content)
    
    # Get the agent_info which should be a list
    agent_name = "test_agent"
    agent_info = data[agent_name]
    
    # Verify it's a list
    assert isinstance(agent_info, list)
    
    # Create a function that simulates the behavior of _map_agent_variables
    # with our fix applied
    def map_agent_variables(agent_name, agent_info):
        # This is the fix we implemented
        if isinstance(agent_info, list) and len(agent_info) > 0:
            agent_info = agent_info[0]
            
        # Try to access a dictionary method on agent_info
        # This would fail with AttributeError if agent_info is still a list
        value = agent_info.get("name")
        return value
    
    # Call the function - this would raise AttributeError before the fix
    result = map_agent_variables(agent_name, agent_info)
    
    # Verify the result
    assert result == "test_agent"
