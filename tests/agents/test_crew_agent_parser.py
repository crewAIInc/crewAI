import pytest

from crewai.agents.crew_agent_executor import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from crewai.agents.parser import CrewAgentParser


@pytest.fixture
def parser():
    agent = MockAgent()
    p = CrewAgentParser(agent)
    return p


def test_valid_action_parsing_special_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what's the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_json_tool_input(parser):
    text = """
    Thought: Let's find the information
    Action: query
    Action Input: ** {"task": "What are some common challenges or barriers that you have observed or experienced when implementing AI-powered solutions in healthcare settings?", "context": "As we've discussed recent advancements in AI applications in healthcare, it's crucial to acknowledge the potential hurdles. Some possible obstacles include...", "coworker": "Senior Researcher"}
    """
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    expected_tool_input = '{"task": "What are some common challenges or barriers that you have observed or experienced when implementing AI-powered solutions in healthcare settings?", "context": "As we\'ve discussed recent advancements in AI applications in healthcare, it\'s crucial to acknowledge the potential hurdles. Some possible obstacles include...", "coworker": "Senior Researcher"}'
    assert result.tool == "query"
    assert result.tool_input == expected_tool_input


def test_valid_action_parsing_with_quotes(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "temperature in SF"


def test_valid_action_parsing_with_curly_braces(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: {temperature in SF}"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "{temperature in SF}"


def test_valid_action_parsing_with_angle_brackets(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: <temperature in SF>"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "<temperature in SF>"


def test_valid_action_parsing_with_parentheses(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: (temperature in SF)"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "(temperature in SF)"


def test_valid_action_parsing_with_mixed_brackets(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: [temperature in {SF}]"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[temperature in {SF}]"


def test_valid_action_parsing_with_nested_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in 'SF'?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in 'SF'?"


def test_valid_action_parsing_with_incomplete_json(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_special_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? @$%^&*"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? @$%^&*"


def test_valid_action_parsing_with_combination(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "[what is the temperature in SF?]"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[what is the temperature in SF?]"


def test_valid_action_parsing_with_mixed_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in SF?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_newlines(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is\nthe temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is\nthe temperature in SF?"


def test_valid_action_parsing_with_escaped_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? \\n"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? \\n"


def test_valid_action_parsing_with_json_string(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"}'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_unbalanced_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what is the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_clean_action_no_formatting(parser):
    action = "Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_asterisks(parser):
    action = "** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_trailing_asterisks(parser):
    action = "Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_and_trailing_asterisks(parser):
    action = "** Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_leading_asterisks(parser):
    action = "**** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_trailing_asterisks(parser):
    action = "Ask question to senior researcher ****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_spaces_and_asterisks(parser):
    action = "  **  Ask question to senior researcher  **  "
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_only_asterisks(parser):
    action = "****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_clean_action_with_empty_string(parser):
    action = ""
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_valid_final_answer_parsing(parser):
    text = (
        "Thought: I found the information\nFinal Answer: The temperature is 100 degrees"
    )
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.output == "The temperature is 100 degrees"


def test_missing_action_error(parser):
    text = "Thought: Let's find the temperature\nAction Input: what is the temperature in SF?"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "Invalid Format: I missed the 'Action:' after 'Thought:'." in str(
        exc_info.value
    )


def test_missing_action_input_error(parser):
    text = "Thought: Let's find the temperature\nAction: search"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "I missed the 'Action Input:' after 'Action:'." in str(exc_info.value)


def test_safe_repair_json(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": Senior Researcher'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unrepairable(parser):
    invalid_json = "{invalid_json"
    result = parser._safe_repair_json(invalid_json)
    assert result == invalid_json  # Should return the original if unrepairable


def test_safe_repair_json_missing_quotes(parser):
    invalid_json = (
        '{task: "Research XAI", context: "Explainable AI", coworker: Senior Researcher}'
    )
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unclosed_brackets(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_extra_commas(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_trailing_commas(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_single_quotes(parser):
    invalid_json = "{'task': 'Research XAI', 'context': 'Explainable AI', 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_mixed_quotes(parser):
    invalid_json = "{'task': \"Research XAI\", 'context': \"Explainable AI\", 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unescaped_characters(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher\n"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_colon(parser):
    invalid_json = '{"task" "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_comma(parser):
    invalid_json = '{"task": "Research XAI" "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unexpected_trailing_characters(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"} random text'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_special_characters_key(parser):
    invalid_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    expected_repaired_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_parsing_with_whitespace(parser):
    text = " Thought: Let's find the temperature \n Action: search \n Action Input: what is the temperature in SF? "
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_parsing_with_special_characters(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "what is the temperature in SF?"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_integration_valid_and_invalid(parser):
    text = """
    Thought: Let's find the temperature
    Action: search
    Action Input: what is the temperature in SF?

    Thought: I found the information
    Final Answer: The temperature is 100 degrees

    Thought: Missing action
    Action Input: invalid

    Thought: Missing action input
    Action: invalid
    """
    parts = text.strip().split("\n\n")
    results = []
    for part in parts:
        try:
            result = parser.parse(part.strip())
            results.append(result)
        except OutputParserException as e:
            results.append(e)

    assert isinstance(results[0], AgentAction)
    assert isinstance(results[1], AgentFinish)
    assert isinstance(results[2], OutputParserException)
    assert isinstance(results[3], OutputParserException)


class MockAgent:
    def increment_formatting_errors(self):
        pass


# TODO: ADD TEST TO MAKE SURE ** REMOVAL DOESN'T MESS UP ANYTHING

def test_ensure_agent_action_is_selected_when_model_hallucinates_observation_and_final_answer(parser):
    text = """Let's continue our effort to gather comprehensive, well-rounded information about AI in healthcare in 2023 to compile a detailed research report effectively.

    Action: Web Search
    Action Input: {"search_query": "innovations in AI for healthcare 2023 latest updates and challenges"}

    Observation: The search is yielding repeated and abundant information on the fragmented, redundant regulatory frameworks, clinical validation importance, and varied insights about AI’s ongoing integration challenges in healthcare. To ensure a rich mix of insights, let's compile, structure, and organize these insights into a coherent report.

    Content Synthesis:
    - **Innovations and Trends**:
    - AI is significantly contributing to personalized medicine, enabling more accurate patient diagnosis and treatment plans.
    - Deep learning models, especially in image and pattern recognition, are revolutionizing radiology and pathology.
    - AI's role in drug discovery is speeding up research and reducing costs and time for new drugs entering the market.
    - AI-driven wearable devices are proving crucial for patient monitoring, predicting potential health issues, and facilitating proactive care.

    Thought: I now have ample information to construct a research report detailing innovations, challenges, and opportunities of AI in healthcare in 2023.

    Final Answer: The finalized detailed research report on AI in Healthcare, 2023:

    Title: Current Innovations, Challenges, and Potential of AI in Healthcare - 2023 Overview

    Introduction:
    The integration of Artificial Intelligence (AI) in healthcare is heralding a new era of modern medicine. In 2023, substantial technological advancements have brought about transformative changes in healthcare delivery. This report explores the latest AI innovations, identifies prevalent challenges, and discusses the potential opportunities in healthcare.

    Potential and Opportunities:
    AI's potential in healthcare is vast, presenting numerous opportunities:
    - Cost Reduction: AI has the capacity to streamline operations, cutting costs significantly.
    - Preventive Healthcare: Utilizing predictive analytics allows for early intervention and prevention, alleviating pressure on emergency and critical care resources.
    - Enhanced Surgeries: Robotic surgeries guided by AI improve surgical outcomes and patient recovery times.
    - Improved Patient Experience: AI-driven solutions personalize patient interaction, improving engagement and healthcare experiences.

    Conclusion:
    AI continues to reshape the healthcare landscape in 2023. Facing challenges head-on with robust solutions will unlock unparalleled benefits, positioning AI as a cornerstone for future medical and healthcare advancements. With ongoing improvements in regulations, data quality, and validation processes, the full potential of AI in healthcare stands to be realized.
    """
    result = parser.parse(text)
    expected_text = """Let's continue our effort to gather comprehensive, well-rounded information about AI in healthcare in 2023 to compile a detailed research report effectively.

    Action: Web Search
    Action Input: {"search_query": "innovations in AI for healthcare 2023 latest updates and challenges"}

    Thought: I now have ample information to construct a research report detailing innovations, challenges, and opportunities of AI in healthcare in 2023.
    """
    assert isinstance(result, AgentAction)
    assert result.text.strip() == expected_text.strip()

def test_ensure_agent_action_is_selected_when_model_hallucinates_observation_field(parser):
    text = """Let's continue our effort to gather comprehensive, well-rounded information about AI in healthcare in 2023 to compile a detailed research report effectively.

    Action: Web Search
    Action Input: {"search_query": "innovations in AI for healthcare 2023 latest updates and challenges"}

    Observation: The search is yielding repeated and abundant information on the fragmented, redundant regulatory frameworks, clinical validation importance, and varied insights about AI’s ongoing integration challenges in healthcare. To ensure a rich mix of insights, let's compile, structure, and organize these insights into a coherent report.

    Content Synthesis:
    - **Innovations and Trends**:
    - AI is significantly contributing to personalized medicine, enabling more accurate patient diagnosis and treatment plans.
    - Deep learning models, especially in image and pattern recognition, are revolutionizing radiology and pathology.

    Final Answer: The finalized detailed research report on AI in Healthcare, 2023:

    Title: Current Innovations, Challenges, and Potential of AI in Healthcare - 2023 Overview

    Introduction:
    The integration of Artificial Intelligence (AI) in healthcare is heralding a new era of modern medicine. In 2023, substantial technological advancements have brought about transformative changes in healthcare delivery. This report explores the latest AI innovations, identifies prevalent challenges, and discusses the potential opportunities in healthcare.

    Innovations and Trends:
    AI technologies are becoming deeply embedded in various aspects of healthcare operations. Key advancements include:
    - Personalized Medicine: AI's analytical capabilities produce precise diagnostic outcomes and tailored treatment plans, fostering personalized medicine.
    - Radiology and Pathology: AI, particularly through advanced deep learning models, is improving imaging accuracy, thereby transforming radiological and pathological analyses.
    """
    result = parser.parse(text)
    expected_text = """Let's continue our effort to gather comprehensive, well-rounded information about AI in healthcare in 2023 to compile a detailed research report effectively.

    Action: Web Search
    Action Input: {"search_query": "innovations in AI for healthcare 2023 latest updates and challenges"}
    """
    assert isinstance(result, AgentAction)
    assert result.text.strip() == expected_text.strip()


def test_ensure_agent_finish_is_selected_when_no_action_was_provided(parser):
    text = """
    ```
    Thought: The repeated results indicate that there may be a technical issue retrieving new information. I will summarize the available knowledge to complete the task.
    Final Answer:
    Research Report on AI in Healthcare (2023)

    1. Introduction:
    AI technologies have become increasingly important in healthcare for their potential to transform patient care, diagnostics, and operational efficiencies. As we progress through 2023, significant advancements are noted alongside various challenges that need addressing.

    2. Developments in AI Technologies:
    Recent years have seen AI significantly impact medical imaging, precision medicine, drug discovery, and robotic surgery. AI algorithms, such as neural networks and machine learning models, provide breakthroughs in analyzing large datasets to identify disease patterns, optimize treatment plans, and predict outcomes. In 2023, AI continues to be integrated within electronic health records, telemedicine platforms, and virtual health assistants, expanding its access and utility.

    3. Challenges:
    - **Data Quality and Availability:** AI models require accurate, comprehensive data. However, healthcare data often remains fragmented and inconsistent, limiting AI's efficacy. High-quality data collection and management are crucial.
    - **Regulatory Frameworks:** Establishing clear regulations is imperative to ensure AI is used safely in clinical environments. Policymakers need to develop standards for AI research, implementation, and continuous monitoring.
    - **Clinical Validation:** Before deploying AI models in healthcare applications, they must undergo rigorous clinical validation to confirm their safety and effectiveness.
    - **Privacy and Consent:** Patient data privacy concerns persist. AI systems need robust mechanisms for data protection and maintaining patient consent when using personal health information.

    4. Future Potentials:
    AI holds the potential to democratize access to healthcare services by making diagnostic tools more accessible and improving personalized treatment plans. Future research and investments are expected to focus on enhancing AI models to process and generate insights from electronic health records, predict patient admissions, and improve monitoring systems in real time.

    5. Conclusion:
    In 2023, AI in healthcare continues to grow, supported by technological advancements and increased investment, despite ongoing challenges. Addressing these issues could allow AI to revolutionize healthcare, improving patient outcomes, and streamlining the efficiency of healthcare systems worldwide.
    ```
    """
    result = parser.parse(text)

    assert isinstance(result, AgentFinish)
    assert result.text.strip() == text.strip()
