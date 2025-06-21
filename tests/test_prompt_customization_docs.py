import pytest
from pydantic import BaseModel
from typing import List

from crewai import Agent, Task
from crewai.lite_agent import LiteAgent
from crewai.utilities.prompts import Prompts
from crewai.utilities import I18N
from crewai.tools import BaseTool


class ResearchOutput(BaseModel):
    summary: str
    key_findings: List[str]
    confidence_score: float


class CodeReviewOutput(BaseModel):
    issues: List[str]
    recommendations: List[str]


class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


class TestPromptCustomizationDocs:
    """Test suite for prompt customization documentation examples."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures for isolation."""
        self.i18n = I18N()

    def test_custom_system_and_prompt_templates(self):
        """Test basic custom template functionality.
        
        Validates:
        - Custom system template assignment
        - Custom prompt template assignment
        - Custom response template assignment
        - System prompt flag configuration
        """
        system_template = """{{ .System }}

Additional context: You are working in a production environment.
Always prioritize accuracy and provide detailed explanations."""

        prompt_template = """{{ .Prompt }}

Remember to validate your approach before proceeding."""

        response_template = """Please format your response as follows:
{{ .Response }}
End of response."""

        agent = Agent(
            role="Data Analyst",
            goal="Analyze data with precision and accuracy",
            backstory="You are an experienced data analyst with expertise in statistical analysis.",
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            use_system_prompt=True,
            llm="gpt-4o-mini"
        )

        assert agent.system_template == system_template
        assert agent.prompt_template == prompt_template
        assert agent.response_template == response_template
        assert agent.use_system_prompt is True

    def test_system_user_prompt_split(self):
        """Test system/user prompt separation."""
        agent = Agent(
            role="Research Assistant",
            goal="Conduct thorough research on given topics",
            backstory="You are a meticulous researcher with access to various information sources.",
            use_system_prompt=True,
            llm="gpt-4o-mini"
        )

        prompts = Prompts(
            i18n=I18N(),
            has_tools=False,
            use_system_prompt=True,
            agent=agent
        )

        prompt_dict = prompts.task_execution()
        
        assert "system" in prompt_dict
        assert "user" in prompt_dict
        assert "You are Research Assistant" in prompt_dict["system"]
        assert agent.goal in prompt_dict["system"]

    def test_structured_output_with_pydantic(self):
        """Test structured output using Pydantic models."""
        class ResearchOutput(BaseModel):
            summary: str
            key_findings: List[str]
            confidence_score: float

        agent = Agent(
            role="Research Assistant",
            goal="Conduct thorough research",
            backstory="You are a meticulous researcher.",
            llm="gpt-4o-mini"
        )

        task = Task(
            description="Research the latest trends in AI development",
            expected_output="A structured research report",
            output_pydantic=ResearchOutput,
            agent=agent
        )

        assert task.output_pydantic == ResearchOutput
        assert task.expected_output == "A structured research report"

    def test_custom_format_instructions(self):
        """Test custom output format instructions."""
        output_format = """{
    "total_sales": "number",
    "growth_rate": "percentage", 
    "top_products": ["list of strings"],
    "recommendations": "detailed string"
}"""

        task = Task(
            description="Analyze the quarterly sales data",
            expected_output="Analysis in JSON format with specific fields",
            output_format=output_format,
            agent=Agent(
                role="Sales Analyst",
                goal="Analyze sales data",
                backstory="You are a sales data expert.",
                llm="gpt-4o-mini"
            )
        )

        assert task.output_format == output_format

    def test_stop_words_configuration(self):
        """Test stop words configuration through response template."""
        response_template = """Provide your analysis:
{{ .Response }}
---END---"""

        agent = Agent(
            role="Analyst", 
            goal="Perform detailed analysis",
            backstory="You are an expert analyst.",
            response_template=response_template,
            llm="gpt-4o-mini"
        )

        assert agent.response_template == response_template

        assert hasattr(agent, 'create_agent_executor')
        assert callable(getattr(agent, 'create_agent_executor'))

    def test_lite_agent_prompt_customization(self):
        """Test LiteAgent prompt customization."""
        class CodeReviewOutput(BaseModel):
            issues_found: List[str]
            severity: str
            recommendations: List[str]

        lite_agent = LiteAgent(
            role="Code Reviewer",
            goal="Review code for quality and security",
            backstory="You are an experienced software engineer specializing in code review.",
            response_format=CodeReviewOutput,
            llm="gpt-4o-mini"
        )

        system_prompt = lite_agent._get_default_system_prompt()
        
        assert "Code Reviewer" in system_prompt
        assert "Review code for quality and security" in system_prompt
        assert "experienced software engineer" in system_prompt

    def test_multi_language_support(self):
        """Test custom templates for different languages."""
        spanish_system_template = """{{ .System }}

Instrucciones adicionales: Responde siempre en español y proporciona explicaciones detalladas."""

        agent = Agent(
            role="Asistente de Investigación",
            goal="Realizar investigación exhaustiva en español",
            backstory="Eres un investigador experimentado que trabaja en español.",
            system_template=spanish_system_template,
            use_system_prompt=True,
            llm="gpt-4o-mini"
        )

        assert agent.system_template == spanish_system_template
        assert "español" in agent.system_template

    def test_domain_specific_formatting(self):
        """Test domain-specific response formatting.
        
        Validates:
        - Domain-specific templates can be applied
        - Response templates support specialized formatting
        """
        medical_response_template = """MEDICAL ANALYSIS REPORT
{{ .Response }}

DISCLAIMER: This analysis is for informational purposes only."""

        medical_agent = Agent(
            role="Medical Data Analyst",
            goal="Analyze medical data with clinical precision",
            backstory="You are a certified medical data analyst with 10 years of experience.",
            response_template=medical_response_template,
            use_system_prompt=True,
            llm="gpt-4o-mini"
        )

        assert "MEDICAL ANALYSIS REPORT" in medical_agent.response_template
        assert "DISCLAIMER" in medical_agent.response_template

    def test_prompt_components_assembly(self):
        """Test how prompt components are assembled."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm="gpt-4o-mini"
        )

        prompts = Prompts(
            i18n=I18N(),
            has_tools=False,
            agent=agent
        )

        prompt_dict = prompts.task_execution()
        
        assert "prompt" in prompt_dict
        assert "Test Agent" in prompt_dict["prompt"]
        assert "Test goal" in prompt_dict["prompt"]
        assert "Test backstory" in prompt_dict["prompt"]

    def test_tools_vs_no_tools_prompts(self):
        """Test prompt generation differences between agents with and without tools.
        
        Validates:
        - Agents without tools use 'no_tools' template slice
        - Agents with tools use 'tools' template slice
        - Prompt generation differs based on tool availability
        """
        agent_no_tools = Agent(
            role="Analyst",
            goal="Analyze data",
            backstory="Expert analyst",
            llm="gpt-4o-mini"
        )
        
        mock_tool = MockTool()
        
        agent_with_tools = Agent(
            role="Analyst",
            goal="Analyze data", 
            backstory="Expert analyst",
            tools=[mock_tool],
            llm="gpt-4o-mini"
        )
        
        assert len(agent_with_tools.tools) == 1
        assert agent_with_tools.tools[0].name == "mock_tool"
        
        prompts_no_tools = Prompts(i18n=I18N(), has_tools=False, agent=agent_no_tools)
        prompts_with_tools = Prompts(i18n=I18N(), has_tools=True, agent=agent_with_tools)
        
        prompt_dict_no_tools = prompts_no_tools.task_execution()
        prompt_dict_with_tools = prompts_with_tools.task_execution()
        
        assert isinstance(prompt_dict_no_tools, dict)
        assert isinstance(prompt_dict_with_tools, dict)
        
        assert prompt_dict_no_tools != prompt_dict_with_tools

    def test_template_placeholder_replacement(self):
        """Test that template placeholders are properly replaced."""
        system_template = "SYSTEM: {{ .System }} - Custom addition"
        prompt_template = "PROMPT: {{ .Prompt }} - Custom addition"
        response_template = "RESPONSE: {{ .Response }} - Custom addition"

        agent = Agent(
            role="Template Tester",
            goal="Test template replacement",
            backstory="You test templates.",
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            llm="gpt-4o-mini"
        )

        prompts = Prompts(
            i18n=I18N(),
            has_tools=False,
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            agent=agent
        )

        prompt_dict = prompts.task_execution()
        
        assert "SYSTEM:" in prompt_dict["prompt"]
        assert "PROMPT:" in prompt_dict["prompt"]
        assert "RESPONSE:" in prompt_dict["prompt"]
        assert "Custom addition" in prompt_dict["prompt"]

    def test_verbose_mode_configuration(self):
        """Test verbose mode for debugging prompts.
        
        Validates:
        - verbose=True parameter can be set on agents
        - Verbose mode configuration is properly stored
        """
        agent = Agent(
            role="Debug Agent",
            goal="Help debug prompt issues", 
            backstory="You are a debugging specialist.",
            verbose=True,
            llm="gpt-4o-mini"
        )

        assert agent.verbose is True

    def test_i18n_slice_access(self):
        """Test internationalization slice access.
        
        Validates:
        - I18N class provides access to template slices
        - Template slices contain expected prompt components
        """
        i18n = I18N()
        
        role_playing_slice = i18n.slice("role_playing")
        observation_slice = i18n.slice("observation")
        tools_slice = i18n.slice("tools")
        no_tools_slice = i18n.slice("no_tools")

        assert "You are {role}" in role_playing_slice
        assert "Your personal goal is: {goal}" in role_playing_slice
        assert "\nObservation:" == observation_slice
        assert "Action:" in tools_slice
        assert "Final Answer:" in no_tools_slice

    @pytest.mark.parametrize("role,goal,backstory", [
        ("Analyst", "Analyze data", "Expert analyst"),
        ("Researcher", "Find facts", "Experienced researcher"),
        ("Writer", "Create content", "Professional writer"),
    ])
    def test_agent_initialization_parametrized(self, role, goal, backstory):
        """Test agent initialization with different role combinations.
        
        Validates:
        - Agents can be created with various role/goal/backstory combinations
        - All parameters are properly stored
        """
        agent = Agent(role=role, goal=goal, backstory=backstory, llm="gpt-4o-mini")
        assert agent.role == role
        assert agent.goal == goal
        assert agent.backstory == backstory

    def test_default_template_behavior(self):
        """Test behavior when no custom templates are provided.
        
        Validates:
        - Agents work correctly with default templates
        - Default templates from translations/en.json are used
        """
        agent = Agent(
            role="Default Agent",
            goal="Test default behavior",
            backstory="Testing default templates",
            llm="gpt-4o-mini"
        )
        
        assert agent.system_template is None
        assert agent.prompt_template is None
        assert agent.response_template is None
        
        prompts = Prompts(i18n=self.i18n, has_tools=False, agent=agent)
        prompt_dict = prompts.task_execution()
        assert isinstance(prompt_dict, dict)

    def test_incomplete_template_definitions(self):
        """Test behavior with incomplete template definitions.
        
        Validates:
        - Agents handle partial template customization gracefully
        - Missing templates fall back to defaults
        """
        agent_partial = Agent(
            role="Partial Agent",
            goal="Test partial templates",
            backstory="Testing incomplete templates",
            system_template="{{ .System }} - Custom system only",
            llm="gpt-4o-mini"
        )
        
        assert agent_partial.system_template is not None
        assert agent_partial.prompt_template is None
        assert agent_partial.response_template is None
        
        prompts = Prompts(i18n=self.i18n, has_tools=False, agent=agent_partial)
        prompt_dict = prompts.task_execution()
        assert isinstance(prompt_dict, dict)

    def test_lite_agent_with_and_without_tools(self):
        """Test LiteAgent behavior with and without tools.
        
        Validates:
        - LiteAgent can be created with and without tools
        - Tool configuration is properly stored
        """
        lite_agent_no_tools = LiteAgent(
            role="Reviewer",
            goal="Review content",
            backstory="Expert reviewer",
            llm="gpt-4o-mini"
        )
        
        mock_tool = MockTool()
        
        lite_agent_with_tools = LiteAgent(
            role="Reviewer",
            goal="Review content",
            backstory="Expert reviewer",
            tools=[mock_tool],
            llm="gpt-4o-mini"
        )
        
        assert lite_agent_no_tools.role == "Reviewer"
        assert lite_agent_with_tools.role == "Reviewer"
        assert len(lite_agent_with_tools.tools) == 1
        assert lite_agent_with_tools.tools[0].name == "mock_tool"
