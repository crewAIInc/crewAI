import os
import pytest
from unittest.mock import patch
from rich.text import Text
from rich.tree import Tree

from crewai.utilities.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterEmojiDisable:
    """Test emoji disable functionality in ConsoleFormatter."""

    def test_emoji_enabled_by_default(self):
        """Test that emojis are enabled by default."""
        formatter = ConsoleFormatter(verbose=True)
        assert not formatter.disable_emojis
        assert formatter._get_icon("✅") == "✅"
        assert formatter._get_icon("❌") == "❌"
        assert formatter._get_icon("🚀") == "🚀"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_emoji_disabled_with_true(self):
        """Test that emojis are disabled when CREWAI_DISABLE_EMOJIS=true."""
        formatter = ConsoleFormatter(verbose=True)
        assert formatter.disable_emojis
        assert formatter._get_icon("✅") == "[DONE]"
        assert formatter._get_icon("❌") == "[FAILED]"
        assert formatter._get_icon("🚀") == "[CREW]"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "1"})
    def test_emoji_disabled_with_one(self):
        """Test that emojis are disabled when CREWAI_DISABLE_EMOJIS=1."""
        formatter = ConsoleFormatter(verbose=True)
        assert formatter.disable_emojis
        assert formatter._get_icon("✅") == "[DONE]"
        assert formatter._get_icon("❌") == "[FAILED]"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "yes"})
    def test_emoji_disabled_with_yes(self):
        """Test that emojis are disabled when CREWAI_DISABLE_EMOJIS=yes."""
        formatter = ConsoleFormatter(verbose=True)
        assert formatter.disable_emojis
        assert formatter._get_icon("✅") == "[DONE]"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "false"})
    def test_emoji_enabled_with_false(self):
        """Test that emojis are enabled when CREWAI_DISABLE_EMOJIS=false."""
        formatter = ConsoleFormatter(verbose=True)
        assert not formatter.disable_emojis
        assert formatter._get_icon("✅") == "✅"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "0"})
    def test_emoji_enabled_with_zero(self):
        """Test that emojis are enabled when CREWAI_DISABLE_EMOJIS=0."""
        formatter = ConsoleFormatter(verbose=True)
        assert not formatter.disable_emojis
        assert formatter._get_icon("✅") == "✅"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": ""})
    def test_emoji_enabled_with_empty_string(self):
        """Test that emojis are enabled when CREWAI_DISABLE_EMOJIS is empty."""
        formatter = ConsoleFormatter(verbose=True)
        assert not formatter.disable_emojis
        assert formatter._get_icon("✅") == "✅"

    def test_emoji_enabled_when_env_var_not_set(self):
        """Test that emojis are enabled when CREWAI_DISABLE_EMOJIS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            formatter = ConsoleFormatter(verbose=True)
            assert not formatter.disable_emojis
            assert formatter._get_icon("✅") == "✅"

    def test_all_emoji_mappings(self):
        """Test that all emojis in EMOJI_MAP have proper text alternatives."""
        with patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"}):
            formatter = ConsoleFormatter(verbose=True)
            
            expected_mappings = {
                "✅": "[DONE]",
                "❌": "[FAILED]", 
                "🚀": "[CREW]",
                "🔄": "[RUNNING]",
                "📋": "[TASK]",
                "🔧": "[TOOL]",
                "🧠": "[THINKING]",
                "🌊": "[FLOW]",
                "✨": "[CREATED]",
                "🧪": "[TEST]",
                "📚": "[KNOWLEDGE]",
                "🔍": "[SEARCH]",
                "🔎": "[QUERY]",
                "🤖": "[AGENT]",
            }
            
            for emoji, expected_text in expected_mappings.items():
                assert formatter._get_icon(emoji) == expected_text

    def test_unknown_emoji_fallback(self):
        """Test that unknown emojis fall back to proper representation."""
        with patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"}):
            formatter = ConsoleFormatter(verbose=True)
            result = formatter._get_icon("🦄")
            assert result == "[ICON:UNKNOWN]"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_crew_tree_creation_without_emojis(self):
        """Test that crew tree creation works without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        tree = formatter.create_crew_tree("Test Crew", "test-id")
        
        assert tree is not None
        tree_label_str = str(tree.label)
        assert "[CREW]" in tree_label_str
        assert "🚀" not in tree_label_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_task_branch_creation_without_emojis(self):
        """Test that task branch creation works without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        crew_tree = Tree("Test Crew")
        task_branch = formatter.create_task_branch(crew_tree, "test-task-id")
        
        assert task_branch is not None
        task_label_str = str(task_branch.label)
        assert "[TASK]" in task_label_str
        assert "📋" not in task_label_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_update_crew_tree_completed_without_emojis(self):
        """Test that crew tree completion updates work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        tree = Tree("Test")
        
        formatter.update_crew_tree(tree, "Test Crew", "test-id", "completed", "Test output")
        
        tree_str = str(tree.label)
        assert "[DONE]" in tree_str
        assert "✅" not in tree_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_tool_usage_without_emojis(self):
        """Test that tool usage events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        crew_tree = Tree("Test Crew")
        agent_branch = crew_tree.add("Test Agent")
        
        formatter.handle_tool_usage_started(agent_branch, "test_tool", crew_tree)
        
        found_tool_branch = False
        for child in agent_branch.children:
            child_str = str(child.label)
            if "test_tool" in child_str:
                assert "[TOOL]" in child_str
                assert "🔧" not in child_str
                found_tool_branch = True
                break
        
        assert found_tool_branch, "Tool branch should have been created"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_llm_call_without_emojis(self):
        """Test that LLM call events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        crew_tree = Tree("Test Crew")
        agent_branch = crew_tree.add("Test Agent")
        
        thinking_branch = formatter.handle_llm_call_started(agent_branch, crew_tree)
        
        if thinking_branch:
            thinking_str = str(thinking_branch.label)
            assert "[THINKING]" in thinking_str
            assert "🧠" not in thinking_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_knowledge_retrieval_without_emojis(self):
        """Test that knowledge retrieval events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        crew_tree = Tree("Test Crew")
        agent_branch = crew_tree.add("Test Agent")
        
        knowledge_branch = formatter.handle_knowledge_retrieval_started(agent_branch, crew_tree)
        
        if knowledge_branch:
            knowledge_str = str(knowledge_branch.label)
            assert "[SEARCH]" in knowledge_str
            assert "🔍" not in knowledge_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_reasoning_without_emojis(self):
        """Test that reasoning events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        crew_tree = Tree("Test Crew")
        agent_branch = crew_tree.add("Test Agent")
        
        reasoning_branch = formatter.handle_reasoning_started(agent_branch, 1, crew_tree)
        
        if reasoning_branch:
            reasoning_str = str(reasoning_branch.label)
            assert "[THINKING]" in reasoning_str
            assert "🧠" not in reasoning_str

    def test_case_insensitive_environment_variable(self):
        """Test that environment variable parsing is case insensitive."""
        test_cases = [
            ("TRUE", True),
            ("True", True), 
            ("true", True),
            ("1", True),
            ("YES", True),
            ("yes", True),
            ("Yes", True),
            ("FALSE", False),
            ("False", False),
            ("false", False),
            ("0", False),
            ("NO", False),
            ("no", False),
            ("No", False),
            ("", False),
            ("random", False),
        ]
        
        for env_value, expected_disabled in test_cases:
            with patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": env_value}):
                formatter = ConsoleFormatter(verbose=True)
                assert formatter.disable_emojis == expected_disabled, f"Failed for env_value: {env_value}"

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_flow_events_without_emojis(self):
        """Test that flow events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        
        flow_tree = formatter.create_flow_tree("Test Flow", "test-flow-id")
        
        if flow_tree:
            flow_str = str(flow_tree.label)
            assert "[FLOW]" in flow_str
            assert "🌊" not in flow_str
            
            for child in flow_tree.children:
                child_str = str(child.label)
                if "Created" in child_str:
                    assert "[CREATED]" in child_str
                    assert "✨" not in child_str

    @patch.dict(os.environ, {"CREWAI_DISABLE_EMOJIS": "true"})
    def test_lite_agent_without_emojis(self):
        """Test that lite agent events work without emojis."""
        formatter = ConsoleFormatter(verbose=True)
        
        formatter.handle_lite_agent_execution("Test Agent", "started")
        
        if formatter.current_lite_agent_branch:
            agent_str = str(formatter.current_lite_agent_branch.label)
            assert "[AGENT]" in agent_str
            assert "🤖" not in agent_str

    def test_backward_compatibility(self):
        """Test that the default behavior (emojis enabled) is preserved."""
        with patch.dict(os.environ, {}, clear=True):
            formatter = ConsoleFormatter(verbose=True)
            
            assert not formatter.disable_emojis
            assert formatter._get_icon("✅") == "✅"
            assert formatter._get_icon("❌") == "❌"
            assert formatter._get_icon("🚀") == "🚀"
            
            tree = formatter.create_crew_tree("Test Crew", "test-id")
            if tree:
                tree_str = str(tree.label)
                assert "🚀" in tree_str
                assert "[CREW]" not in tree_str
