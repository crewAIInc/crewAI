import pytest
from crewai.utilities.xml_parser import (
    extract_xml_content,
    extract_all_xml_content,
    extract_multiple_xml_tags,
    remove_xml_tags,
    strip_xml_tags_keep_content,
)


class TestXMLParserExamples:
    """Test XML parser with realistic agent output examples."""

    def test_agent_thinking_extraction(self):
        """Test extracting thinking content from agent output."""
        agent_output = """
        I need to solve this problem step by step.
        
        <thinking>
        Let me break this down:
        1. First, I need to understand the requirements
        2. Then, I'll analyze the constraints
        3. Finally, I'll propose a solution
        
        The key insight is that we need to balance efficiency with accuracy.
        </thinking>
        
        Based on my analysis, here's my recommendation: Use approach A.
        """
        
        thinking = extract_xml_content(agent_output, "thinking")
        assert thinking is not None
        assert "break this down" in thinking
        assert "requirements" in thinking
        assert "constraints" in thinking
        assert "efficiency with accuracy" in thinking

    def test_multiple_reasoning_tags(self):
        """Test extracting multiple reasoning sections."""
        agent_output = """
        <reasoning>
        Initial analysis shows three possible approaches.
        </reasoning>
        
        Let me explore each option:
        
        <reasoning>
        Option A: Fast but less accurate
        Option B: Slow but very accurate  
        Option C: Balanced approach
        </reasoning>
        
        My final recommendation is Option C.
        """
        
        reasoning_sections = extract_all_xml_content(agent_output, "reasoning")
        assert len(reasoning_sections) == 2
        assert "three possible approaches" in reasoning_sections[0]
        assert "Option A" in reasoning_sections[1]
        assert "Option B" in reasoning_sections[1]
        assert "Option C" in reasoning_sections[1]

    def test_complex_agent_workflow(self):
        """Test complex agent output with multiple tag types."""
        complex_output = """
        <thinking>
        This is a complex problem requiring systematic analysis.
        I need to consider multiple factors.
        </thinking>
        
        <analysis>
        Factor 1: Performance requirements
        Factor 2: Cost constraints
        Factor 3: Time limitations
        </analysis>
        
        <reasoning>
        Given the analysis above, I believe we should prioritize performance
        while keeping costs reasonable. Time is less critical in this case.
        </reasoning>
        
        <conclusion>
        Recommend Solution X with performance optimizations.
        </conclusion>
        
        Final answer: Implement Solution X with the following optimizations...
        """
        
        extracted = extract_multiple_xml_tags(
            complex_output, 
            ["thinking", "analysis", "reasoning", "conclusion"]
        )
        
        assert extracted["thinking"] is not None
        assert "systematic analysis" in extracted["thinking"]
        
        assert extracted["analysis"] is not None
        assert "Factor 1" in extracted["analysis"]
        assert "Factor 2" in extracted["analysis"]
        assert "Factor 3" in extracted["analysis"]
        
        assert extracted["reasoning"] is not None
        assert "prioritize performance" in extracted["reasoning"]
        
        assert extracted["conclusion"] is not None
        assert "Solution X" in extracted["conclusion"]

    def test_clean_output_for_user(self):
        """Test cleaning agent output for user presentation."""
        raw_output = """
        <thinking>
        Internal reasoning that user shouldn't see.
        This contains implementation details.
        </thinking>
        
        <debug>
        Debug information: variable X = 42
        </debug>
        
        Here's the answer to your question: The solution is to use method Y.
        
        <internal_notes>
        Remember to update the documentation later.
        </internal_notes>
        
        This approach will give you the best results.
        """
        
        clean_output = remove_xml_tags(
            raw_output, 
            ["thinking", "debug", "internal_notes"]
        )
        
        assert "Internal reasoning" not in clean_output
        assert "Debug information" not in clean_output
        assert "update the documentation" not in clean_output
        assert "Here's the answer" in clean_output
        assert "method Y" in clean_output
        assert "best results" in clean_output

    def test_preserve_structured_content(self):
        """Test preserving structured content while removing tags."""
        structured_output = """
        <steps>
        1. Initialize the system
        2. Load the configuration
        3. Process the data
        4. Generate the report
        </steps>
        
        Follow these steps to complete the task.
        """
        
        clean_output = strip_xml_tags_keep_content(structured_output, ["steps"])
        
        assert "<steps>" not in clean_output
        assert "</steps>" not in clean_output
        assert "1. Initialize" in clean_output
        assert "2. Load" in clean_output
        assert "3. Process" in clean_output
        assert "4. Generate" in clean_output
        assert "Follow these steps" in clean_output
