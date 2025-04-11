import pytest

from crewai.utilities import sanitize_collection_name


def test_sanitize_collection_name_with_non_ascii_chars():
    """Test that sanitize_collection_name properly handles non-ASCII characters."""
    chinese_role = "一位有 20 年经验的 GraphQL 查询专家"
    sanitized_name = sanitize_collection_name(chinese_role)
    
    assert len(sanitized_name) >= 3
    assert len(sanitized_name) <= 63
    assert sanitized_name[0].isalnum()
    assert sanitized_name[-1].isalnum()
    assert all(c.isalnum() or c == '_' or c == '-' for c in sanitized_name)
    assert '__' not in sanitized_name  # No consecutive underscores
    
    special_chars_role = "Café Owner & Barista (España) 🇪🇸"
    sanitized_name = sanitize_collection_name(special_chars_role)
    
    assert len(sanitized_name) >= 3
    assert len(sanitized_name) <= 63
    assert sanitized_name[0].isalnum()
    assert sanitized_name[-1].isalnum()
    assert all(c.isalnum() or c == '_' or c == '-' for c in sanitized_name)
    assert '__' not in sanitized_name  # No consecutive underscores


def test_sanitize_collection_name_edge_cases():
    """Test edge cases for sanitize_collection_name function."""
    empty_role = ""
    sanitized_name = sanitize_collection_name(empty_role)
    assert len(sanitized_name) >= 3  # Should be padded to minimum length
    
    special_only = "!@#$%^&*()"
    sanitized_name = sanitize_collection_name(special_only)
    assert len(sanitized_name) >= 3
    assert sanitized_name[0].isalnum()
    assert sanitized_name[-1].isalnum()
    
    long_role = "a" * 100
    sanitized_name = sanitize_collection_name(long_role)
    assert len(sanitized_name) <= 63
    
    consecutive_spaces = "Hello   World"
    sanitized_name = sanitize_collection_name(consecutive_spaces)
    assert "__" not in sanitized_name


def test_sanitize_collection_name_reproduces_issue_2534():
    """Test that reproduces the specific issue from #2534."""
    problematic_role = "一位有 20 年经验的 GraphQL 查询专家"
    
    sanitized_name = sanitize_collection_name(problematic_role)
    
    assert len(sanitized_name) >= 3
    assert len(sanitized_name) <= 63
    assert sanitized_name[0].isalnum()
    assert sanitized_name[-1].isalnum()
    assert all(c.isalnum() or c == '_' or c == '-' for c in sanitized_name)
    assert '__' not in sanitized_name  # No consecutive underscores
