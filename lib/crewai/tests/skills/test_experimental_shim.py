"""The old experimental namespace must keep re-exporting the public names."""


def test_experimental_namespace_reexports_public_names():
    from crewai.experimental import skills as experimental_skills
    from crewai.skills.cache import SkillCacheManager
    from crewai.skills.registry import (
        SkillNotCachedError,
        is_registry_ref,
        parse_registry_ref,
        resolve_registry_ref,
    )

    assert experimental_skills.SkillCacheManager is SkillCacheManager
    assert experimental_skills.SkillNotCachedError is SkillNotCachedError
    assert experimental_skills.is_registry_ref is is_registry_ref
    assert experimental_skills.parse_registry_ref is parse_registry_ref
    assert experimental_skills.resolve_registry_ref is resolve_registry_ref


def test_experimental_submodule_imports_alias_real_modules():
    """Old submodule import style must resolve to the crewai.skills modules."""
    import importlib

    from crewai.experimental.skills import cache as shim_cache
    from crewai.experimental.skills import events as shim_events
    from crewai.experimental.skills import registry as shim_registry
    from crewai.experimental.skills.cache import SkillCacheManager
    from crewai.experimental.skills.registry import resolve_registry_ref
    from crewai.skills import cache as real_cache
    from crewai.skills import events as real_events
    from crewai.skills import registry as real_registry

    assert shim_cache is real_cache
    assert shim_events is real_events
    assert shim_registry is real_registry
    assert SkillCacheManager is real_cache.SkillCacheManager
    assert resolve_registry_ref is real_registry.resolve_registry_ref
    # Dotted-path import resolves through sys.modules to the same module.
    assert (
        importlib.import_module("crewai.experimental.skills.registry")
        is real_registry
    )
