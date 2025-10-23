import pytest
import crewai_tools


def test_typo_in_tool_name_lowercase_t():
    """Test that accessing a tool with lowercase 't' in 'tool' provides a helpful error message."""
    with pytest.raises(ImportError) as exc_info:
        getattr(crewai_tools, "CSVSearchtool")
    
    assert "Cannot import name 'CSVSearchtool' from 'crewai_tools'" in str(exc_info.value)
    assert "Did you mean 'CSVSearchTool'?" in str(exc_info.value)
    assert "Tool names use capital 'T' in 'Tool'" in str(exc_info.value)


def test_typo_pgsearchtool_lowercase_t():
    """Test that accessing PGSearchtool (with lowercase 't') provides a helpful error message."""
    with pytest.raises(ImportError) as exc_info:
        getattr(crewai_tools, "PGSearchtool")
    
    assert "Cannot import name 'PGSearchtool' from 'crewai_tools'" in str(exc_info.value)
    assert "Did you mean 'PGSearchTool'?" in str(exc_info.value)


def test_pgsearchtool_not_implemented():
    """Test that accessing PGSearchTool (correct spelling) shows it's not yet implemented."""
    with pytest.raises(NotImplementedError) as exc_info:
        getattr(crewai_tools, "PGSearchTool")
    
    assert "'PGSearchTool' is currently under development" in str(exc_info.value)
    assert "not yet available" in str(exc_info.value)


def test_nonexistent_tool():
    """Test that accessing a completely nonexistent tool gives a standard AttributeError."""
    with pytest.raises(AttributeError) as exc_info:
        getattr(crewai_tools, "NonExistentTool")
    
    assert "module 'crewai_tools' has no attribute 'NonExistentTool'" in str(exc_info.value)


def test_multiple_typos():
    """Test multiple common typos to ensure they all get helpful messages."""
    typos = [
        ("FileReadtool", "FileReadTool"),
        ("PDFSearchtool", "PDFSearchTool"),
        ("MySQLSearchtool", "MySQLSearchTool"),
    ]
    
    for typo, correct in typos:
        with pytest.raises(ImportError) as exc_info:
            getattr(crewai_tools, typo)
        
        assert f"Cannot import name '{typo}' from 'crewai_tools'" in str(exc_info.value)
        assert f"Did you mean '{correct}'?" in str(exc_info.value)
