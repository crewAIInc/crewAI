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
    from crewai.experimental.skills import cache as shim_cache
    from crewai.experimental.skills import events as shim_events
    from crewai.experimental.skills import registry as shim_registry
    import crewai.experimental.skills.registry as shim_registry_direct
    from crewai.experimental.skills.cache import SkillCacheManager
    from crewai.experimental.skills.registry import resolve_registry_ref
    import crewai.skills.cache
    import crewai.skills.events
    import crewai.skills.registry

    assert shim_cache is crewai.skills.cache
    assert shim_events is crewai.skills.events
    assert shim_registry is crewai.skills.registry
    assert shim_registry_direct is crewai.skills.registry
    assert SkillCacheManager is crewai.skills.cache.SkillCacheManager
    assert resolve_registry_ref is crewai.skills.registry.resolve_registry_ref
