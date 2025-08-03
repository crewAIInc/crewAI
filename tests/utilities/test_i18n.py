import json
import os
import pytest

from crewai.utilities.i18n import I18N


def test_load_prompts():
    i18n = I18N()
    i18n.load_prompts()
    assert i18n._prompts is not None


def test_slice():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.slice("role_playing"), str)


def test_tools():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.tools("ask_question"), str)


def test_retrieve():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)


def test_retrieve_not_found():
    i18n = I18N()
    i18n.load_prompts()
    with pytest.raises(Exception):
        i18n.retrieve("nonexistent_kind", "nonexistent_key")


def test_prompt_file():
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"


def test_global_i18n_file_environment_variable(monkeypatch):
    """Test that CREWAI_I18N_FILE environment variable is respected"""
    test_translations = {
        "slices": {"role_playing": "Test role playing message"},
        "tools": {"ask_question": "Test ask question message"}
    }
    
    test_file_path = os.path.join(os.path.dirname(__file__), "test_env_prompts.json")
    with open(test_file_path, "w", encoding="utf-8") as f:
        json.dump(test_translations, f)
    
    try:
        monkeypatch.setenv("CREWAI_I18N_FILE", test_file_path)
        
        i18n = I18N()
        i18n.load_prompts()
        
        assert i18n.slice("role_playing") == "Test role playing message"
        assert i18n.tools("ask_question") == "Test ask question message"
        
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_prompt_file_priority_over_environment_variable(monkeypatch):
    """Test that explicit prompt_file takes priority over environment variable"""
    monkeypatch.setenv("CREWAI_I18N_FILE", "/nonexistent/path.json")
    
    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"


def test_environment_variable_file_not_found(monkeypatch):
    """Test proper error handling when environment variable points to non-existent file"""
    monkeypatch.setenv("CREWAI_I18N_FILE", "/nonexistent/file.json")
    
    with pytest.raises(Exception, match="Prompt file '/nonexistent/file.json' not found"):
        I18N()


def test_fallback_to_default_when_no_environment_variable(monkeypatch):
    """Test that it falls back to default en.json when no environment variable is set"""
    monkeypatch.delenv("CREWAI_I18N_FILE", raising=False)
    
    i18n = I18N()
    i18n.load_prompts()
    
    assert isinstance(i18n.slice("role_playing"), str)
    assert len(i18n.slice("role_playing")) > 0


def test_chinese_translation_file():
    """Test loading Chinese translation file"""
    import os
    
    zh_path = os.path.join(os.path.dirname(__file__), "../../src/crewai/translations/zh.json")
    zh_path = os.path.abspath(zh_path)
    
    i18n = I18N(prompt_file=zh_path)
    i18n.load_prompts()
    
    assert i18n.retrieve("hierarchical_manager_agent", "role") == "团队经理"
    assert i18n.slice("observation") == "\n观察:"
    assert i18n.errors("tool_usage_error") == "我遇到了错误: {error}"


def test_spanish_translation_file():
    """Test loading Spanish translation file"""
    import os
    
    es_path = os.path.join(os.path.dirname(__file__), "../../src/crewai/translations/es.json")
    es_path = os.path.abspath(es_path)
    
    i18n = I18N(prompt_file=es_path)
    i18n.load_prompts()
    
    assert i18n.retrieve("hierarchical_manager_agent", "role") == "Gerente del Equipo"
    assert i18n.slice("observation") == "\nObservación:"
    assert i18n.errors("tool_usage_error") == "Encontré un error: {error}"
