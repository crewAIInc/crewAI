import pytest
from crewai.utilities.i18n import I18N


def test_default_language_is_english():
    i18n = I18N()
    assert i18n.language == "en"
    assert isinstance(i18n.slice("role_playing"), str)
    assert "You are {role}" in i18n.slice("role_playing")


def test_explicit_english_language():
    i18n = I18N(language="en")
    assert i18n.language == "en"
    assert isinstance(i18n.slice("role_playing"), str)
    assert "You are {role}" in i18n.slice("role_playing")


def test_spanish_language():
    i18n = I18N(language="es")
    assert i18n.language == "es"
    assert isinstance(i18n.slice("role_playing"), str)
    assert "Eres {role}" in i18n.slice("role_playing")


def test_spanish_hierarchical_manager():
    i18n = I18N(language="es")
    role = i18n.retrieve("hierarchical_manager_agent", "role")
    goal = i18n.retrieve("hierarchical_manager_agent", "goal")
    backstory = i18n.retrieve("hierarchical_manager_agent", "backstory")
    
    assert role == "Gerente del Equipo"
    assert "Gestionar el equipo" in goal
    assert "gerente experimentado" in backstory


def test_spanish_errors():
    i18n = I18N(language="es")
    error = i18n.errors("tool_usage_error")
    assert "Encontré un error" in error


def test_spanish_tools():
    i18n = I18N(language="es")
    delegate_work = i18n.tools("delegate_work")
    assert "Delega una tarea específica" in delegate_work


def test_fallback_to_english_for_unsupported_language():
    i18n = I18N(language="fr")
    assert isinstance(i18n.slice("role_playing"), str)
    assert "You are {role}" in i18n.slice("role_playing")


def test_custom_prompt_file_overrides_language():
    import os
    import tempfile
    import json
    
    custom_prompts = {
        "slices": {
            "role_playing": "Custom role playing prompt"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_prompts, f)
        temp_file = f.name
    
    try:
        i18n = I18N(prompt_file=temp_file, language="es")
        assert i18n.slice("role_playing") == "Custom role playing prompt"
    finally:
        os.unlink(temp_file)


def test_retrieve_with_spanish():
    i18n = I18N(language="es")
    observation = i18n.retrieve("slices", "observation")
    assert "Observación" in observation


def test_spanish_reasoning_prompts():
    i18n = I18N(language="es")
    initial_plan = i18n.retrieve("reasoning", "initial_plan")
    assert "Eres {role}" in initial_plan
    assert "profesional" in initial_plan


def test_language_parameter_validation():
    i18n = I18N(language="en")
    assert i18n.language == "en"
    
    i18n_es = I18N(language="es")
    assert i18n_es.language == "es"
    
    i18n_pt = I18N(language="pt")
    assert i18n_pt.language == "pt"
