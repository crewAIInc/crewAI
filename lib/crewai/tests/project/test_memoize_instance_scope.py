"""Memoized decorator methods must not leak results across instances.

``memoize`` cached into a module-level ``CacheHandler`` keyed by
``id(self)``. When an instance is garbage-collected, CPython can hand its
``id()`` to a fresh instance, and the new instance then receives the dead
instance's cached @crew/@agent/@task/... result — e.g. a serving loop doing
``MyCrew().crew().kickoff()`` per request silently reuses a previous
request's Crew, agents, and tasks. The memo store must live on the instance
so it dies with it.
"""


from crewai.project.utils import memoize


class CrewLike:
    """Minimal stand-in with the shape memoize cares about (has __dict__)."""


def test_memoized_method_does_not_serve_a_recycled_id():
    calls: list[int] = []

    @memoize
    def build(self):  # noqa: ANN001, ANN202
        calls.append(id(self))
        return object()

    stale_hits = 0
    for _ in range(200):
        # create, call, drop: the freed slot is handed to the next instance
        # on CPython, so id() recycles within a few iterations
        instance = CrewLike()
        executions_before = len(calls)
        result = build(instance)
        if len(calls) == executions_before:
            stale_hits += 1  # served a cached result from a previous instance
        del instance, result
        # No gc.collect() here: CPython refcounting frees the slot at the
        # del above (CrewLike has no cycles), and 200 full collections blew
        # CI's 60s per-test timeout on slower runners.

    assert stale_hits == 0, (
        f"{stale_hits}/200 fresh instances received a recycled id's cached "
        "result from another instance"
    )


def test_memoized_method_still_idempotent_per_instance():
    calls: list[int] = []

    @memoize
    def build(self):  # noqa: ANN001, ANN202
        calls.append(id(self))
        return object()

    instance = CrewLike()
    first = build(instance)
    second = build(instance)

    assert first is second
    assert len(calls) == 1


def test_two_live_instances_do_not_share_results():
    @memoize
    def build(self):  # noqa: ANN001, ANN202
        return object()

    a = CrewLike()
    b = CrewLike()

    assert build(a) is not build(b)
    assert build(a) is build(a)
