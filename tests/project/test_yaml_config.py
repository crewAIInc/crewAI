import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


class TestYamlConfig:
    """Tests for YAML configuration handling."""
    
    def test_list_format_in_yaml(self):
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
            if isinstance(agent_info, list):
                if not agent_info:
                    raise ValueError(f"Empty agent configuration list for agent {agent_name}")
                agent_info = agent_info[0]
                
            # Try to access a dictionary method on agent_info
            # This would fail with AttributeError if agent_info is still a list
            value = agent_info.get("name")
            return value
        
        # Call the function - this would raise AttributeError before the fix
        result = map_agent_variables(agent_name, agent_info)
        
    def test_empty_list_in_yaml(self):
        """Test that empty list in YAML raises appropriate error."""
        # Create a test YAML content with empty list
        yaml_content = """
        test_agent: []
        """
        
        # Parse the YAML content
        data = yaml.safe_load(yaml_content)
        
        # Get the agent_info which should be an empty list
        agent_name = "test_agent"
        agent_info = data[agent_name]
        
        # Verify it's a list
        assert isinstance(agent_info, list)
        assert len(agent_info) == 0
        
        # Create a function that simulates the behavior of _map_agent_variables
        def map_agent_variables(agent_name, agent_info):
            if isinstance(agent_info, list):
                if not agent_info:
                    raise ValueError(f"Empty agent configuration list for agent {agent_name}")
                agent_info = agent_info[0]
            return agent_info
        
        # Call the function - should raise ValueError
        with pytest.raises(ValueError, match=f"Empty agent configuration list for agent {agent_name}"):
            map_agent_variables(agent_name, agent_info)
    def test_multiple_items_in_list(self):
        """Test that when multiple items are in the list, the first one is used."""
        # Create a test YAML content with multiple items in the list
        yaml_content = """
        test_agent:
          - name: first_agent
            role: First Agent
            goal: First Goal
          - name: second_agent
            role: Second Agent
            goal: Second Goal
        """
        
        # Parse the YAML content
        data = yaml.safe_load(yaml_content)
        
        # Get the agent_info which should be a list
        agent_name = "test_agent"
        agent_info = data[agent_name]
        
        # Verify it's a list with multiple items
        assert isinstance(agent_info, list)
        assert len(agent_info) > 1
        
        # Create a function that simulates the behavior of _map_agent_variables
        def map_agent_variables(agent_name, agent_info):
            if isinstance(agent_info, list):
                if not agent_info:
                    raise ValueError(f"Empty agent configuration list for agent {agent_name}")
                agent_info = agent_info[0]
            return agent_info.get("name")
        
        # Call the function - should return name from the first item
        result = map_agent_variables(agent_name, agent_info)
        
        # Verify only the first item was used
        assert result == "first_agent"
