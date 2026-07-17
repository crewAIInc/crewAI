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
