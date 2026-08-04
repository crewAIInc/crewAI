# Conversational route labels vs method trigger namespace

**Status:** Proposal — awaiting go-ahead  
**Branch:** `vo/fix/seperate-namespace`

## Problem

In conversational Flows, `@listen("create_video")` binds a handler to a **router intent
label** emitted by `route_conversation` / `route_turn`. It does not mean “run again when
the `create_video` method completes.”

`FlowDefinition._validate_trigger_namespace` treats route labels and method-completion
triggers as one string namespace, so the natural handler name fails at instantiation:

```python
@listen("create_video")
def create_video(self) -> str:
    return "made a video"

MyFlow()  # ValidationError: methods.create_video.listen must not reference itself
```

---

## Plan A — Minimal fix (ship first)

**Goal:** Allow conversational route handlers whose method name matches their `@listen`
label; keep guards for true method-to-method self-triggers.

### Validation (`flow_definition.py`)

Exempt conversational, non-router handlers:

```python
if _is_conversational_route_handler(self, method_name, method):
    continue
```

Returns `True` when `conversational.enabled`, `not method.router`, and listen references
method name. Routers like `@router("route") def route` remain rejected.

### Runtime (`runtime/__init__.py`)

Add `suppress_self_retrigger_on_completion: bool = False` to `_execute_listeners`:

| Call site | Flag |
|-----------|------|
| After method completes in `_execute_single_listener` | `True` |
| Router result from `_execute_start_method` (e.g. `"create_video"`) | `False` |
| Default elsewhere | `False` |

In `_find_triggered_methods`, skip when flag is set and trigger equals listener with
self-referencing listen condition. Do **not** use `idx == 0` alone — router start methods
call `_execute_listeners(router_result, ...)` directly.

### Tests

`tests/test_flow_trigger_namespace_conversational.py` — golden repro, `handle_turn`
runtime, non-conversational self-listen guards.

### Scope

Non-conversational router-label == handler-name collisions stay rejected.

**Effort:** ~50–80 lines + tests.

---

## Plan B — Structural fix (follow-up)

**Goal:** Internal trigger provenance instead of label-only dispatch.

### `FlowTrigger` model

```python
@dataclass(frozen=True)
class FlowTrigger:
    label: str
    emitter: str
    kind: Literal["method_completion", "router_emit"]
```

### Matching policy

- Method chain: match on `label`
- Route handler with `listen == method_name`: match on `label` + `kind == router_emit`
- Self-loop guard: skip when `kind == method_completion` and `emitter == listener_name`

Subsumes Plan A special cases; public `@listen` API unchanged.

**Effort:** Medium runtime refactor.

---

## Recommendation

Ship **Plan A** after approval; track **Plan B** as runtime cleanup.
