import pytest
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from rich.text import Text
from rich.tree import Tree


def test_console_formatter_gbk_encoding():
    """Test that console formatter output can be encoded with GBK."""
    formatter = ConsoleFormatter(verbose=True)
    
    tree = Tree("Test Tree")
    
    formatter.update_tree_label(tree, "[OK] Test:", "Test Name", "green")
    label_text = str(tree.label)
    
    node = formatter.add_tree_node(tree, "[OK] Test Node", "green")
    node_text = str(node.label)
    
    try:
        label_text.encode("gbk")
        node_text.encode("gbk")
        assert True, "Text can be encoded with GBK"
    except UnicodeEncodeError as e:
        assert False, f"Failed to encode with GBK: {e}"
    
    crew_tree = formatter.create_crew_tree("Test Crew", "crew-123")
    task_branch = formatter.create_task_branch(crew_tree, "1")
    agent_branch = formatter.create_agent_branch(task_branch, "Test Agent", crew_tree)
    
    formatter.update_task_status(crew_tree, "1", "Test Agent", "completed")
    formatter.update_agent_status(agent_branch, "Test Agent", crew_tree, "completed")
    
    flow_tree = formatter.create_flow_tree("Test Flow", "flow-123")
    formatter.update_flow_status(flow_tree, "Test Flow", "flow-123", "completed")
    
    for tree_obj in [crew_tree, task_branch, agent_branch, flow_tree]:
        tree_text = str(tree_obj)
        try:
            tree_text.encode("gbk")
        except UnicodeEncodeError as e:
            assert False, f"Failed to encode tree with GBK: {e}"
