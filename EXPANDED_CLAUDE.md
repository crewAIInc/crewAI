# EXPANDED_CLAUDE.md

Deep architectural reference for the CrewAI codebase. See [CLAUDE.md](./CLAUDE.md) for quick-start commands and overview.

## Table of Contents

- [1. Execution Flow: Crew.kickoff() to Agent Output](#1-execution-flow)
- [2. Agent System](#2-agent-system)
- [3. Task System](#3-task-system)
- [4. Flow System](#4-flow-system)
- [5. Memory System](#5-memory-system)
- [6. Tool System](#6-tool-system)
- [7. Event System](#7-event-system)
- [8. LLM Abstraction](#8-llm-abstraction)
- [9. crewai-tools Package](#9-crewai-tools-package)
- [10. crewai-files Package](#10-crewai-files-package)
- [11. CLI & Project Scaffolding](#11-cli--project-scaffolding)
- [12. Project Decorators (@CrewBase)](#12-project-decorators)
- [13. Knowledge & RAG](#13-knowledge--rag)
- [14. Security & Fingerprinting](#14-security--fingerprinting)
- [15. Agent-to-Agent (A2A)](#15-agent-to-agent-a2a)
- [16. Translations & i18n](#16-translations--i18n)

---

## 1. Execution Flow

The end-to-end path from `Crew.kickoff()` to final output:

```
Crew.kickoff(inputs)
├── prepare_kickoff()                    # Validate inputs, store files
├── Determine process type
│   ├── Sequential: _run_sequential_process()
│   └── Hierarchical: _run_hierarchical_process() → _create_manager_agent()
├── _execute_tasks(tasks)                # Main loop
│   └── For each task:
│       ├── If ConditionalTask: check condition(previous_output)
│       ├── If async_execution: create asyncio task
│       └── If sync: task.execute_sync(agent, context, tools)
│           └── agent.execute_task(task, context, tools)
│               ├── Memory recall (if enabled)
│               ├── Knowledge retrieval (if enabled)
│               ├── Build prompt with context
│               └── CrewAgentExecutor.invoke()
│                   └── Loop until AgentFinish:
│                       ├── Native tool calling (if LLM supports)
│                       └── OR ReAct text pattern (fallback)
├── Apply guardrails with retries
├── after_kickoff_callbacks()
└── Return CrewOutput
```

**Process types:**
- **Sequential**: Tasks execute in order; each gets context from all prior TaskOutputs
- **Hierarchical**: A manager agent delegates to other agents via delegation tools

**Agent execution loop** (`agents/crew_agent_executor.py`):
- **Native function calling**: LLM returns structured `tool_calls`; executor runs first tool, appends result, loops
- **ReAct text pattern** (fallback): LLM outputs `Thought/Action/Action Input`; executor parses text, runs tool, appends `Observation`

---

## 2. Agent System

**Key files:** `agent/core.py`, `agents/agent_builder/base_agent.py`, `agents/crew_agent_executor.py`

### Agent class (`agent/core.py`)

Extends `BaseAgent`. Core fields:
- `role`, `goal`, `backstory` — define agent identity/prompting
- `llm` — BaseLLM instance (auto-created from string)
- `function_calling_llm` — optional specialized LLM for tool calls
- `tools` — list of BaseTool instances
- `memory` — optional unified Memory instance
- `knowledge_sources` — optional knowledge base
- `max_iter` (default 25), `max_rpm`, `max_retry_limit` (default 2)
- `allow_delegation` — enables delegation tools
- `reasoning` — enables planning before execution
- `guardrail` — validation function for output
- `code_execution_mode` — "safe" (Docker) or "unsafe" (local)
- `apps` — platform integrations (Asana, GitHub, Slack, etc.)
- `mcps` — MCP server configurations

### BaseAgent (`agents/agent_builder/base_agent.py`)

Abstract base with: `id` (UUID4), `agent_executor`, `cache_handler`, `tools_handler`, `security_config`, `i18n`. Defines abstract methods: `execute_task()`, `create_agent_executor()`, `get_delegation_tools()`, `get_platform_tools()`.

### CrewAgentExecutor (`agents/crew_agent_executor.py`)

The agent execution loop. Key attributes: `llm`, `task`, `crew`, `agent`, `prompt`, `tools`, `messages`, `iterations`, `max_iter`, `respect_context_window`. Entry point: `invoke(inputs)` → `_invoke_loop()`.

### LiteAgent (`lite_agent.py`)

Lightweight alternative agent implementation with: event-driven execution, memory integration, LLM hooks, guardrail support, structured output via Converter.

---

## 3. Task System

**Key files:** `task.py`, `tasks/task_output.py`, `tasks/conditional_task.py`

### Task class (`task.py`)

Core fields:
- `description`, `expected_output` — task prompt and LLM guidance
- `agent` — assigned BaseAgent
- `tools` — optional task-specific tools (override agent tools)
- `context` — list of prior Tasks whose output provides context
- `output_file`, `output_pydantic`, `output_json` — output format
- `guardrail` + `guardrail_max_retries` (default 3) — output validation
- `async_execution` — run in background thread
- `human_input` — request human feedback
- `callback` — post-completion callback

### TaskOutput (`tasks/task_output.py`)

Result container: `raw` (text), `pydantic` (model instance), `json_dict`, `agent` (role string), `output_format`, `messages`.

### ConditionalTask (`tasks/conditional_task.py`)

Extends Task with `condition: Callable[[TaskOutput], bool]`. Evaluates against previous output; if False, appends empty TaskOutput and skips. Cannot be first/only task or async.

---

## 4. Flow System

**Key files:** `flow/flow.py`, `flow/flow_wrappers.py`, `flow/persistence/`, `flow/human_feedback.py`

### Flow class (`flow/flow.py`)

Generic `Flow[T]` where T is `dict` or Pydantic `BaseModel` (must have `id` field). Uses `FlowMeta` metaclass to register decorators at class definition.

**Decorator API:**
```python
@start(condition=None)          # Entry point (unconditional or conditional)
@listen(condition)              # Event handler (fires when condition met)
@router(condition)              # Decision point (return value becomes trigger)
@human_feedback(message, emit)  # Collect human feedback, optionally route

or_(*conditions)                # Fire when ANY condition met
and_(*conditions)               # Fire when ALL conditions met
```

**Execution model:**
1. Execute all unconditional `@start` methods in parallel
2. After each method completes: find triggered routers (sequential), then listeners (parallel)
3. Continue chain until no more triggers

**Key rules:**
- Routers are sequential; listeners are parallel
- OR listeners fire once on first trigger; AND listeners wait for all
- State access is thread-safe via `StateProxy` with `_state_lock`
- Cyclic flows: methods cleared from `_completed_methods` to allow re-execution

### Persistence (`flow/persistence/`)

- `FlowPersistence` ABC: `save_state()`, `load_state()`, `save_pending_feedback()`, `load_pending_feedback()`
- `SQLiteFlowPersistence`: stores in `~/.crewai/flows.db`
- Enables resumption via `Flow.from_pending(flow_id, persistence)`

### Human Feedback (`flow/human_feedback.py`)

`@human_feedback` decorator wraps method to collect feedback. With `emit` parameter, acts as router (LLM collapses feedback to outcome). Supports async providers that raise `HumanFeedbackPending` to pause flow. Optional `learn=True` stores lessons in memory.

### Flow Methods

- `kickoff(inputs)` / `akickoff(inputs)` — sync/async execution
- `resume(feedback)` / `resume_async(feedback)` — resume from pause
- `ask(message, timeout)` — request user input (auto-checkpoints state)
- `state` — thread-safe state proxy
- `recall(query)` / `remember(content)` — memory integration

---

## 5. Memory System

**Key files:** `memory/unified_memory.py`, `memory/types.py`, `memory/encoding_flow.py`, `memory/recall_flow.py`, `memory/memory_scope.py`, `memory/analyze.py`, `memory/storage/`

### Memory class (`memory/unified_memory.py`)

Singleton-style with lazy LLM/embedder init. Pluggable storage backend (default LanceDB). Background save queue via ThreadPoolExecutor(max_workers=1).

**Public API:**
- **Write:** `remember(content, scope, categories, importance, ...)`, `remember_many(contents, ...)` (non-blocking batch)
- **Read:** `recall(query, scope, categories, limit, depth="shallow"|"deep")`
- **Manage:** `forget(scope, categories, older_than, ...)`, `update(record_id, ...)`, `drain_writes()`
- **Scoping:** `scope(path)` → `MemoryScope`, `slice(scopes, read_only)` → `MemorySlice`
- **Introspection:** `list_scopes()`, `list_records()`, `list_categories()`, `info()`, `tree()`

**Configuration:**
- Scoring weights: `semantic_weight=0.5`, `recency_weight=0.3`, `importance_weight=0.2`
- `recency_half_life_days=30` — exponential decay
- `consolidation_threshold=0.85` — dedup trigger similarity
- `confidence_threshold_high=0.8`, `confidence_threshold_low=0.5` — recall routing
- `exploration_budget=1` — LLM exploration rounds for deep recall

### Data Types (`memory/types.py`)

- **MemoryRecord**: `id`, `content`, `scope` (hierarchical path like `/company/team`), `categories`, `metadata`, `importance` (0-1), `created_at`, `last_accessed`, `embedding`, `source`, `private`
- **MemoryMatch**: `record`, `score` (composite), `match_reasons`, `evidence_gaps`
- **ScopeInfo**: `path`, `record_count`, `categories`, date range, `child_scopes`

**Composite scoring formula:**
```
score = semantic_weight × similarity + recency_weight × (0.5 ^ (age_days / half_life)) + importance_weight × importance
```

### Encoding Flow (`memory/encoding_flow.py`)

5-step batch pipeline on save:
1. **Batch embed** all items (single API call)
2. **Intra-batch dedup** via cosine similarity matrix (threshold 0.98)
3. **Parallel find similar** records in storage (8 workers)
4. **Parallel analyze** — Groups: A (insert, 0 LLM), B (consolidation, 1 LLM), C (save analysis, 1 LLM), D (both, 2 LLM) — 10 workers
5. **Execute plans** — batch re-embed, atomic storage mutations (delete + update + insert under write lock)

### Recall Flow (`memory/recall_flow.py`)

Adaptive recall pipeline:
1. **Analyze query** — short queries skip LLM; long queries get sub-queries, scope suggestions, complexity classification, time filters
2. **Filter & chunk** candidate scopes (max 20)
3. **Parallel search** across queries × scopes (4 workers), apply filters, compute composite scores
4. **Route** — high confidence → synthesize; low confidence + budget → explore deeper
5. **Recursive exploration** (if deeper) — LLM extracts relevant info + gaps; decrements budget; re-searches
6. **Synthesize** — deduplicate by ID, rank by composite score, return top N

### Storage Backend (`memory/storage/backend.py`)

Protocol interface: `save()`, `update()`, `delete()`, `search()`, `get_record()`, `list_records()`, `get_scope_info()`, `list_scopes()`, `list_categories()`, `count()`, `reset()`, `write_lock` property.

**LanceDB implementation** (`memory/storage/lancedb_storage.py`): auto-detects vector dimensions, class-level shared RLock per DB path, auto-compaction every 100 saves, retry logic for commit conflicts (exponential backoff, 5 retries), oversamples 3x when filters present.

### Scoped Views (`memory/memory_scope.py`)

- **MemoryScope**: wraps Memory with root_path prefix; all operations relative to that root
- **MemorySlice**: multi-scope view; recall searches all scopes in parallel; optional `read_only=True`

---

## 6. Tool System

**Key files:** `tools/base_tool.py`, `tools/structured_tool.py`, `tools/tool_calling.py`, `tools/tool_usage.py`, `tools/memory_tools.py`

### BaseTool (`tools/base_tool.py`)

Abstract Pydantic BaseModel. Key fields: `name`, `description`, `args_schema` (Pydantic model), `result_as_answer`, `max_usage_count`, `cache_function`. Subclasses implement `_run(**kwargs)` and optionally `_arun(**kwargs)`.

**`@tool` decorator:** creates tool from function, auto-infers schema from type hints.

### CrewStructuredTool (`tools/structured_tool.py`)

Wraps functions as structured tools for LLM function calling. `from_function()` factory. Validates inputs before execution, enforces usage limits.

### Tool Execution Flow (`tools/tool_usage.py`)

`ToolUsage` manages selection → validation → execution:
1. Parse tool call from LLM output
2. Select tool (fuzzy matching, 85%+ ratio)
3. Validate arguments against schema
4. Execute with fingerprint metadata
5. Cache results if configured
6. Emit events throughout lifecycle

Retry: max 3 parsing attempts with fallback methods (JSON, JSON5, AST, JSON repair).

### Memory Tools (`tools/memory_tools.py`)

- **RecallMemoryTool**: searches memory with single/multiple queries, returns formatted results with deduplication
- **RememberTool**: stores facts/decisions, infers scope/categories/importance
- **CalculatorTool**: safe arithmetic via AST parser (no `eval()`), supports date differences

### MCP Integration

- **MCPToolWrapper** (`tools/mcp_tool_wrapper.py`): on-demand connections, retry with exponential backoff, timeouts (15s connect, 60s execute)
- **MCPNativeTool** (`tools/mcp_native_tool.py`): reuses persistent MCP sessions, auto-reconnect on event loop changes

---

## 7. Event System

**Key files:** `events/event_bus.py`, `events/event_listener.py`, `events/base_events.py`, `events/types/`

### Event Bus (`events/event_bus.py`)

Singleton `CrewAIEventsBus`. Thread-safe with RWLock. Supports sync handlers (ThreadPoolExecutor, 10 workers) and async handlers (dedicated daemon event loop). Handler dependency injection via `Depends()`.

**Key methods:** `emit(source, event)`, `aemit()`, `flush(timeout=30)`, `register_handler()`, `scoped_handlers()` (context manager for temporary handlers).

### Event Types (`events/types/`)

- **Tool events**: `ToolUsageStartedEvent`, `ToolUsageFinishedEvent`, `ToolUsageErrorEvent`, `ToolValidateInputErrorEvent`, `ToolSelectionErrorEvent`
- **LLM events**: `LLMCallStartedEvent`, `LLMCallCompletedEvent`, `LLMCallFailedEvent`, `LLMStreamChunkEvent`, `LLMThinkingChunkEvent`
- **Agent/Task/Crew events**: lifecycle tracking (started, completed, failed)
- **Flow events**: method execution states, paused, input requested/received
- **Memory events**: retrieval started/completed/failed
- **MCP events**: connection, tool execution
- **A2A events**: agent-to-agent delegation

All events carry: UUID, timestamp, parent/previous chain, fingerprint context.

---

## 8. LLM Abstraction

**Key files:** `llm.py`, `llms/base_llm.py`, `llms/providers/`

### BaseLLM (`llms/base_llm.py`)

Abstract interface: `call(messages, tools, ...)` and `acall(...)`. Provider-specific constants for context windows (1KB–2MB). Emits LLM events. Handles context window management, timeout/auth errors, streaming.

### LLM class (`llm.py`)

High-level wrapper integrating with litellm for multi-provider support. Handles model identification, tool function calling, JSON schema responses, streaming chunk aggregation, multimodal content formatting.

### Providers (`llms/providers/`)

Per-provider adapters: OpenAI, Azure, Gemini, Claude/Anthropic, Bedrock, Watson, etc.

---

## 9. crewai-tools Package

**Location:** `lib/crewai-tools/`

93+ pre-built tools. All inherit from `crewai.tools.BaseTool`.

**Pattern for creating tools:**
```python
class MyToolSchema(BaseModel):
    param: str = Field(..., description="...")

class MyTool(BaseTool):
    name: str = "My Tool"
    description: str = "..."
    args_schema: type[BaseModel] = MyToolSchema

    def _run(self, param: str) -> str:
        return result
```

**Tool categories:**
- **Search/Web**: BraveSearch, Tavily, EXASearch, Serper, Spider, SerpAPI
- **Scraping**: Firecrawl, Jina, Scrapfly, Selenium, Browserbase, Stagehand
- **File search**: PDF, CSV, JSON, XML, MDX, DOCX, TXT search tools
- **Database**: MySQL, Snowflake, SingleStore, MongoDB, Qdrant, Weaviate, Couchbase
- **File I/O**: FileRead, FileWriter, DirectoryRead, DirectorySearch, FileCompressor, OCR, Vision
- **Code**: CodeInterpreter, CodeDocsSearch, NL2SQL, DallE
- **AWS**: Bedrock agent/KB, S3 reader/writer
- **Integrations**: Composio, Zapier, MCP, LlamaIndex, GitHub
- **RAG**: RagTool base with 17 loaders (CSV, Directory, Docs, DOCX, GitHub, JSON, MySQL, Postgres, etc.)

**43+ optional dependency groups** for external services.

---

## 10. crewai-files Package

**Location:** `lib/crewai-files/`

Multimodal file handling for LLM providers.

**Structure:**
- `core/` — File type classes (Image, PDF, Audio, Video, Text), source types (FilePath, FileBytes, FileUrl, FileStream), resolved representations
- `processing/` — FileProcessor validates against per-provider constraints, optional transforms (resize, compress, chunk)
- `uploaders/` — Provider-specific uploaders (Anthropic, OpenAI, Gemini, Bedrock/S3)
- `formatting/` — Format files for provider APIs: `format_multimodal_content()`, `aformat_multimodal_content()`
- `resolution/` — FileResolver decides inline base64 vs upload based on size/provider
- `cache/` — UploadCache tracks uploads by content hash, cleanup utilities

**Provider constraints**: max file sizes, supported formats, image dimensions per provider (Anthropic, OpenAI, Gemini, Bedrock).

---

## 11. CLI & Project Scaffolding

**Key file:** `cli/cli.py` (Click-based)

**Core commands:**
- `crewai create <crew|flow> <name>` — scaffold project
- `crewai run` / `crewai flow kickoff` — execute crew/flow
- `crewai chat` — interactive conversation with crew
- `crewai train [-n N]` / `crewai test [-n N] [-m MODEL]` — training and evaluation
- `crewai replay [-t TASK_ID]` — replay from specific task

**Memory/config:**
- `crewai reset_memories` — reset memory, knowledge, or all
- `crewai memory` — open Memory TUI
- `crewai config list|set|reset` — CLI configuration

**Deployment:**
- `crewai deploy create|list|push|status|logs|remove`

**Tool repository:**
- `crewai tool create|install|publish`

**Flow-specific:**
- `crewai flow kickoff|plot|add-crew`

**Other:** `crewai login`, `crewai org list|switch|current`, `crewai traces enable|disable|status`, `crewai env view`

---

## 12. Project Decorators

**Key files:** `project/crew_base.py`, `project/annotations.py`

### @CrewBase decorator

Applies `CrewBaseMeta` metaclass. Auto-loads YAML configs (`config/agents.yaml`, `config/tasks.yaml`). Registers agent/task factory methods, MCP adapters, lifecycle hooks.

### Method decorators (`project/annotations.py`)

**Component factories** (all memoized):
- `@agent` — agent factory method
- `@task` — task factory method
- `@llm` — LLM provider factory
- `@tool` — tool factory
- `@callback`, `@cache_handler`

**Lifecycle:**
- `@before_kickoff` / `@after_kickoff` — pre/post execution hooks
- `@crew` — main crew entry point (instantiates agents/tasks, manages callbacks)

**Output format:** `@output_json`, `@output_pydantic`

**LLM/Tool hooks** (optional agent/tool filtering):
- `@before_llm_call_hook` / `@after_llm_call_hook`
- `@before_tool_call_hook` / `@after_tool_call_hook`

---

## 13. Knowledge & RAG

**Key files:** `knowledge/knowledge.py`, `rag/`

### Knowledge class

Vector store integration: `query(queries, results_limit, score_threshold)`, `add_sources()`, `reset()`. Async variants available. Used by agents via `knowledge_sources` parameter.

### RAG system (`rag/`)

- **Vector DBs**: ChromaDB, Qdrant (client wrappers, factories, config)
- **Embeddings**: 25+ providers (OpenAI, Cohere, HuggingFace, Jina, Voyage, Ollama, Bedrock, Azure, Vertex, etc.)
- **Core**: `BaseClient`, `BaseEmbeddingsProvider` abstractions
- **Storage**: `BaseRAGStorage` interface

---

## 14. Security & Fingerprinting

**Key files:** `security/security_config.py`, `security/fingerprint.py`

- **SecurityConfig**: manages component fingerprints, serialization
- **Fingerprint**: dual identifiers (human-readable ID + UUID), `uuid5()` with CrewAI namespace for deterministic seeding, metadata support (1-level nesting, 10KB limit), timestamp tracking
- Every event carries fingerprint context for audit trails

---

## 15. Agent-to-Agent (A2A)

**Key files:** `a2a/config.py`, `a2a/`

Protocol for inter-agent communication:
- `A2AClientConfig`, `A2AServerConfig` — configuration
- `AgentCardSigningConfig` — JWS signing (RS256, ES256, PS256)
- `GRPCServerConfig` — gRPC transport with TLS
- Supporting: `auth/`, `updates/` (polling/push/streaming), `extensions/`, `utils/`

---

## 16. Translations & i18n

**Key file:** `translations/en.json`

All agent-facing prompts are externalized. Key sections:
- `slices/` — agent prompting templates (task, memory, role_playing, tools, format, final_answer_format)
- `errors/` — tool execution, validation, format violation, guardrail failure messages
- `tools/` — tool descriptions (delegate_work, ask_question, recall_memory, calculator, save_to_memory)
- `memory/` — query analysis, extraction rules, consolidation logic, temporal reasoning
- HITL prompts — pre-review, lesson distillation
- Lite agent prompts — system prompts with/without tools
