# Flow Definition

You are writing a CrewAI Flow declaration for the user.
Use these instructions when the user asks you to create or edit a Flow.
Return one valid `crewai.flow/v1` YAML or JSON document.

Treat this document as instructions for you, not as text to show the user.
Follow the examples for shape and formatting, then use the API reference to check exact fields.

## Output Format

Return one valid `crewai.flow/v1` Flow declaration.
Do not include explanatory prose unless the user asks for it.

## Build It In This Order

1. Define `state` first. Use `type: json_schema` and put the JSON Schema inline.
2. Put required input fields in `state.json_schema.required`. Do not rely on `state.default` to make fields required.
3. Add exactly one method with `start: true`.
4. Add later methods with `listen`.
5. Give each method exactly one `do` action object. Never make `do` a list.
6. Pass data with `${...}` mappings from `state` and completed `outputs`.
7. Before final output, check every `listen`, `emit`, and `outputs.some_method` reference.

Set optional fields only when you are confident they are needed. Otherwise, trust CrewAI defaults and omit them.

Method names must match `^[A-Za-z_][A-Za-z0-9_]*$`.

## Choose One Action Per Method

Pick the simplest action that does the job.

- Use `call: expression` for simple reads, filters, computed values, and deterministic routing.
- Use `call: agent` for one AI worker that classifies, decides, summarizes, writes, or drafts. Put `role`, `goal`, `backstory`, and `input` under `with`. Do not add an action-level `inputs` map to an agent.
- Use `call: crew` for coordinated AI work with multiple agents or tasks. Define the crew under `with`. Pass runtime values with the action-level `inputs` map.

## Wire Methods Explicitly

- `state` is the initial shared data shape. Action results do not automatically merge into `state`.
- Read method results with `outputs.method_name` after that method can run.
- `listen` targets a method name or a router-emitted event name.
- Methods must not listen to their own method name.
- Method names and emitted event names share one namespace. Avoid reusing the same string for both unless the user explicitly wants that.
- Use `router: true` plus `emit` when one method chooses between named branches.
- A router action must return exactly one emitted event string. It must not return JSON, a list, or an explanation.
- Use `start: true` for the single entrypoint.

If an agent is a router, make its goal say exactly what to return, for example:
`Return exactly one bare value: approved, rejected, or needs_review. Do not include explanation.`
Prefer `call: expression` when routing can be computed without an agent.

## CEL And Dynamic Values

CEL is the expression language for reading Flow data and making small decisions.
Use agents and crews for larger work or side effects.

Use these expression forms correctly:

- Raw CEL: use in `expr`. Do not wrap raw CEL in `${...}`.
- Use `${...}` inside action mapping strings to read Flow data with CEL. Example value: `Ticket: ${state.ticket_id}`.
- Use `state` for input data. Use `outputs.step_name` for a completed method result.
- If a value is only one `${...}` expression, the result keeps its type. Use this for numbers, booleans, objects, and lists.
- If the string has other text, the final value is text. Non-text values become JSON. `null` becomes empty text.
- Use `text(root, "path", "default")` for values that may be missing or null. The default is optional and is `""`.

Expression examples:

Mix text and Flow data:

```yaml
query: "News about ${state.topic}"
```

Keep a list or number type:

```yaml
domains: "${state.domains}"
limit: "${state.limit}"
```

Use a default for missing text:

```yaml
input: "Ticket ${text(state, \"ticket.id\", \"unknown\")}"
```

- Crew text: use `{name}` placeholders from crew inputs. Example: `Research {topic}`.
- Crew inputs become prompt text only when agent or task text references matching `{name}` placeholders.
- Passing an input that is not referenced by any `{name}` placeholder does not ground the crew. If the crew needs a field, put that placeholder in an agent `goal`, task `description`, or task `expected_output`.

Available CEL variables:

- `state`: initial input data, for example `state.ticket.subject`.
- `outputs`: completed method outputs, for example `outputs.classify_ticket`.

Dynamic value rules:

- When an agent needs multiple fields, write one text value with labels and separators. Example value: `Ticket ID: ${state.ticket_id}; Message: ${state.message}`.
- Crew action-level `inputs` are the actual Crew kickoff inputs. Use `${...}` values there for runtime data from `state` or `outputs`.
- Crew action-level `inputs` alone are not grounding. Include placeholders for the facts the model must use.
- Crew outputs are objects. Use `${outputs.research_brief.raw}` for text.
- For structured crew output, use fields like `${outputs.research_brief.json_dict.field}` or `${outputs.research_brief.pydantic.field}`.
- Do not pass a whole crew output to an agent input, like `${outputs.research_brief}`.
- Agent outputs may also be objects. Use fields like `${outputs.classify_ticket.raw}` or `${outputs.classify_ticket.pydantic.category}`.
- Use `with.inputs` only for static Crew input defaults.
- Agent action `with.input` is the agent's single input value.

## Do Not

- Do not invent top-level keys outside the Flow declaration shape.
- Do not use fields outside the declaration schema.
- Do not put more than one action under a method's `do`.
- Do not make `do` a list.
- Do not reference `outputs.some_method` before `some_method` can run.
- Do not set a method's `listen` to its own method name.
- Do not use the same string for an emitted event and a method name unless the user asks for it.
- Do not use `emit` without `router: true`.
- Do not rely on crew action-level `inputs` alone to ground agent behavior. Inputs that do not match placeholders are effectively unused by the prompt.
- Do not ask agents to infer missing facts when accuracy matters. Tell them to mark missing dates, amounts, offers, logs, or constraints as unknown.
- Do not set `config.stream: true` unless the caller is expected to consume a streaming result. For normal generated flows and CLI smoke tests, omit it.

## Examples

### Crew review with routed follow-up

```yaml
schema: crewai.flow/v1
name: ResearchReviewFlow
state:
  type: json_schema
  json_schema:
    type: object
    properties:
      topic:
        type: string
      audience:
        type: string
    required:
      - topic
      - audience
  default:
    topic: AI agent orchestration
    audience: platform engineering leaders
methods:
  research_brief:
    start: true
    do:
      call: crew
      with:
        agents:
          researcher:
            role: Research analyst
            goal: Research {topic} for {audience}
            backstory: Expert at concise technical research.
          reviewer:
            role: Strategy reviewer
            goal: Decide whether the research needs an executive follow-up
            backstory: Experienced at reviewing technical briefs for leaders.
        tasks:
          - name: research_task
            description: Research {topic} for {audience}.
            expected_output: Key findings and tradeoffs.
            agent: researcher
          - name: review_task
            description: Review the research and decide if an executive follow-up is needed.
            expected_output: 'A brief review ending with `needs_followup: true` or `needs_followup: false`.'
            agent: reviewer
        inputs:
          topic: Default topic
          audience: Default audience
      inputs:
        topic: "${state.topic}"
        audience: "${state.audience}"
  route_followup:
    listen: research_brief
    router: true
    emit:
      - followup
      - done
    do:
      call: agent
      with:
        role: Follow-up router
        goal: 'Return exactly one bare value: followup or done. Do not include explanation.'
        backstory: Skilled at routing reviewed research briefs.
        input: "Reviewed research: ${outputs.research_brief.raw}"
  write_followup:
    listen: followup
    do:
      call: agent
      with:
        role: Executive communications specialist
        goal: Draft a concise executive follow-up from the reviewed research
        backstory: Writes crisp follow-ups for technical leaders.
        input: "${outputs.research_brief.raw}"
```

## API Reference

Use this appendix to check exact field names, required fields, linked object types, and allowed action/state shapes. Linked type names point to another section in this reference.

### Flow Definition

Fields:
- `schema` (optional): must be `crewai.flow/v1`; default `crewai.flow/v1`. Declarative Flow schema identifier and version. Include it explicitly in authored declarations.
- `name` (required): string. Unique flow name used in logs, events, and traces.
- `description` (optional): string | null; default `null`. Human-readable summary of the flow.
- `state` (required): [State](#json-schema-state-statetypejson_schema). State contract for the initial state and updates during execution.
- `config` (optional): [Config (`config`)](#config-config); default generated default. Serializable flow-level execution configuration.
- `methods` (required): map of string to [Method](#method-methods). Mapping of method names to method definitions.

### JSON Schema State (`state[type=json_schema]`)

Shape:
- `type: json_schema`

Fields:
- `type` (optional): must be `json_schema`; default `json_schema`. Inline JSON Schema used as the Flow state contract.
- `json_schema` (required): map of string to any. JSON Schema used to validate and document flow state. Declare required fields with JSON Schema's `required` array.
- `default` (optional): map of string to any | null; default `null`. Default values used to initialize Flow state. Defaults are not the same as schema-required fields.

### Method (`methods.<name>`)

Fields:
- `description` (optional): string | null; default `null`. Human-readable summary of what this method does.
- `do` (required): [Action](#action). Single action object executed when this method runs.
- `start` (optional): boolean | string | map of string to any | null; default `null`. Marks the single normal entrypoint. Use `true`.
- `listen` (optional): string | map of string to any | null; default `null`. Runs this method after one upstream method or router-emitted event.
- `router` (optional): boolean; default `false`. Whether the method output should be treated as the next event name. Router actions must return one event name string, with no surrounding explanation.
- `emit` (optional): list[string] | null; default `null`. Declared router events this method may emit. Each emitted event name should be unique and should not collide with method names.

### Action

Discriminated union by `call`.

Allowed shapes:
- [`call: crew`](#crew-action-methodsdocallcrew)
- [`call: agent`](#agent-action-methodsdocallagent)
- [`call: expression`](#expression-action-methodsdocallexpression)

### Crew Action (`methods.<name>.do[call=crew]`)

Shape:
- `call: crew`

Fields:
- `call` (required): must be `crew`. Action discriminator. Use crew to run an inline Crew definition. Example: `crew`
- `with` (required): inline crew definition. Inline Crew definition to load and execute for this action. Example: `{"agents": {"researcher": {"backstory": "Knows the domain.", "goal": "Research {topic}", "role": "Researcher"}}, "name": "inline_research", "tasks": [{"agent": "researcher", "description": "Research {topic}", "expected_output": "Findings about {topic}", "name": "research_task"}]}`
- `inputs` (optional): map of string to expression data | null; default `null`. Actual kickoff inputs passed to the Crew. Use `${...}` inside action mapping strings to read Flow data with CEL. Example value: `Ticket: ${state.ticket_id}`. Use `state` for input data. Use `outputs.step_name` for a completed method result. If a value is only one `${...}` expression, the result keeps its type. Use this for numbers, booleans, objects, and lists. If the string has other text, the final value is text. Non-text values become JSON. `null` becomes empty text. Use `text(root, "path", "default")` for values that may be missing or null. The default is optional and is `""`. The evaluated values are available to crew agent and task interpolation as `{name}` placeholders; reference each input the crew needs in agent or task text. Example: `{"topic": "${state.topic}"}`

#### Crew Definition (`methods.<name>.do[call=crew].with`)

Fields:
- `agents` (required): map of string to any | list[map of string to any]. Inline crew agents keyed by agent name. Example: `{"researcher": {"backstory": "Expert at concise technical research.", "goal": "Research {topic}", "role": "Research analyst"}}`
- `tasks` (required): list[any]. Ordered crew tasks. Example: `[{"agent": "researcher", "description": "Research {topic}.", "expected_output": "Key findings about {topic}.", "name": "research_task"}]`
- `inputs` (optional): map of string to any. Static default crew inputs. Values are available to crew agent and task interpolation as `{name}` placeholders, for example `{topic}`. Prefer action-level crew `inputs` for runtime values from `state` or `outputs`, and include placeholders for any inputs the crew must reason over. Example: `{"topic": "AI agents"}`

#### Crew Agent Definition (`methods.<name>.do[call=crew].with.agents.<name>`)

Fields:
- `role` (required): string. Crew agent role. Crew inputs are interpolated with `{name}` placeholders such as `{topic}`; this is not CEL. Example: `Research analyst`
- `goal` (required): string. Crew agent goal. Crew inputs are interpolated with `{name}` placeholders such as `{topic}`; this is not CEL. Example: `Research {topic}`
- `backstory` (required): string. Crew agent backstory. Crew inputs are interpolated with `{name}` placeholders such as `{topic}`; this is not CEL. Example: `Expert at concise technical research.`
- `settings` (optional): map of string to any. Additional agent settings passed to the loader. Example: `{"llm": "openai/gpt-4o-mini"}`
- `llm` (optional): string or inline LLM config; default `null`. Language model that runs this crew agent. Use an object when setting LLM options such as `max_tokens`. Example: `{"max_tokens": 4096, "model": "openai/gpt-4o-mini"}`
- `planning_config` (optional): object | null; default `null`. Agent planning configuration. Set `max_attempts` to limit planning refinement attempts before task execution. Example: `{"max_attempts": 3}`
- `allow_delegation` (optional): boolean | null; default `null`. Enable agent to delegate and ask questions among each other. Example: `false`
- `max_iter` (optional): integer | null; default `null`. Maximum iterations for an agent to execute a task Example: `25`
- `max_rpm` (optional): integer | null; default `null`. Maximum number of requests per minute for the agent execution to be respected. Example: `10`
- `max_execution_time` (optional): integer | null; default `null`. Maximum execution time in seconds for an agent to execute a task Example: `300`
- `tools` (optional): list[string | map of string to any] | null; default `null`. Tool refs or serialized tool definitions available to this agent. String refs can use CrewAI tool names, `custom:<name>`, or fully qualified `module:Class` references. Example: `["crewai_tools:SerperDevTool", "custom:file_read"]`
- `apps` (optional): list[string] | null; default `null`. Platform apps available to this agent. Can contain app names such as `gmail` or app/action refs such as `gmail/send_email`. Example: `["gmail", "slack/send_message"]`
- `mcps` (optional): list[string | map of string to any] | null; default `null`. MCP server refs or serialized MCP server configs available to this agent. String refs can use HTTPS URLs, connected MCP integration slugs, or refs with a `#tool_name` suffix for specific tools. Example: `["https://api.weather.com/mcp#get_current_weather", "snowflake", "stripe#list_invoices", {"cache_tools_list": true, "headers": {"Authorization": "Bearer your_token"}, "streamable": true, "url": "https://api.example.com/mcp"}]`

#### LLM Definition

Fields:
- `model` (required): string. Model identifier used to instantiate the LLM. Example: `openai/gpt-4o-mini`
- `max_tokens` (optional): integer | null; default `null`. Maximum number of tokens the LLM can generate. If null, CrewAI does not set an explicit output token cap and the provider's default applies. Example: `4096`

#### Crew Task Definition (`methods.<name>.do[call=crew].with.tasks[]`)

Fields:
- `description` (required): string. Task instructions. Crew inputs are interpolated with `{name}` placeholders such as `{topic}`; this is not CEL. Example: `Research {topic}.`
- `expected_output` (required): string. Expected task output. Crew inputs are interpolated with `{name}` placeholders such as `{topic}`; this is not CEL. Example: `Key findings about {topic}.`
- `name` (optional): string | null; default `null`. Optional task name. Example: `research_task`
- `agent` (optional): string | null; default `null`. Name of the crew agent assigned to this task. Example: `researcher`

### Agent Action (`methods.<name>.do[call=agent]`)

Shape:
- `call: agent`

Fields:
- `call` (required): must be `agent`. Action discriminator. Use agent to run an individual inline Agent definition outside of a crew. Example: `agent`
- `with` (required): any. Individual Agent definition to load and execute outside of a crew for this action. Put the agent input in `with.input`; agent actions do not support action-level `inputs`. Example: `{"backstory": "Precise and concise.", "goal": "Answer user questions", "input": "${state.question}", "role": "Analyst", "settings": {"llm": "openai/gpt-4o-mini"}}`

#### Agent Definition (`methods.<name>.do[call=agent].with`)

Fields:
- `role` (required): string. Individual agent role used by a Flow agent action outside of a crew. Example: `Support specialist`
- `goal` (required): string. Individual agent goal for the Flow agent action outside of a crew. Example: `Draft a concise customer reply`
- `backstory` (required): string. Individual agent backstory used to shape behavior outside of a crew. Example: `Expert at resolving SaaS support questions.`
- `settings` (optional): map of string to any. Additional agent settings passed to the loader. Example: `{"llm": "openai/gpt-4o-mini"}`
- `llm` (optional): string or inline LLM config; default `null`. Language model that runs this agent. Use an object when setting LLM options such as `max_tokens`. Example: `{"max_tokens": 4096, "model": "openai/gpt-4o-mini"}`
- `planning_config` (optional): object | null; default `null`. Agent planning configuration. Set `max_attempts` to limit planning refinement attempts before task execution. Example: `{"max_attempts": 3}`
- `allow_delegation` (optional): boolean | null; default `null`. Enable agent to delegate and ask questions among each other. Example: `false`
- `max_iter` (optional): integer | null; default `null`. Maximum iterations for an agent to execute a task Example: `25`
- `max_rpm` (optional): integer | null; default `null`. Maximum number of requests per minute for the agent execution to be respected. Example: `10`
- `max_execution_time` (optional): integer | null; default `null`. Maximum execution time in seconds for an agent to execute a task Example: `300`
- `tools` (optional): list[string | map of string to any] | null; default `null`. Tool refs or serialized tool definitions available to this agent. String refs can use CrewAI tool names, `custom:<name>`, or fully qualified `module:Class` references. Example: `["crewai_tools:SerperDevTool", "custom:file_read"]`
- `apps` (optional): list[string] | null; default `null`. Platform apps available to this agent. Can contain app names such as `gmail` or app/action refs such as `gmail/send_email`. Example: `["gmail", "slack/send_message"]`
- `mcps` (optional): list[string | map of string to any] | null; default `null`. MCP server refs or serialized MCP server configs available to this agent. String refs can use HTTPS URLs, connected MCP integration slugs, or refs with a `#tool_name` suffix for specific tools. Example: `["https://api.weather.com/mcp#get_current_weather", "snowflake", "stripe#list_invoices", {"cache_tools_list": true, "headers": {"Authorization": "Bearer your_token"}, "streamable": true, "url": "https://api.example.com/mcp"}]`
- `input` (required): string. Input passed to the individual agent kickoff outside of a crew. Use one string. Use `${...}` inside action mapping strings to read Flow data with CEL. Example value: `Ticket: ${state.ticket_id}`. Use `state` for input data. Use `outputs.step_name` for a completed method result. If a value is only one `${...}` expression, the result keeps its type. Use this for numbers, booleans, objects, and lists. If the string has other text, the final value is text. Non-text values become JSON. `null` becomes empty text. Use `text(root, "path", "default")` for values that may be missing or null. The default is optional and is `""`. When an agent needs multiple fields, write one string with labels and separators, for example `Ticket ID: ${state.ticket_id}; Message: ${state.message}`. Example: `${state.ticket.body}`

#### LLM Definition

Fields:
- `model` (required): string. Model identifier used to instantiate the LLM. Example: `openai/gpt-4o-mini`
- `max_tokens` (optional): integer | null; default `null`. Maximum number of tokens the LLM can generate. If null, CrewAI does not set an explicit output token cap and the provider's default applies. Example: `4096`

### Expression Action (`methods.<name>.do[call=expression]`)

Shape:
- `call: expression`

Fields:
- `call` (required): must be `expression`. Action discriminator. Use expression to evaluate a CEL expression.
- `expr` (required): string. CEL expression evaluated against state, outputs, and local context.

### Config (`config`)

Fields:
- `tracing` (optional): boolean | null; default `null`. Override for flow tracing; when omitted, execution defaults apply.
- `stream` (optional): boolean; default `false`. Whether the flow should emit streaming events when supported.
- `memory` (optional): map of string to any | null; default `null`. Serializable memory configuration passed to flow execution.
- `input_provider` (optional): string | null; default `null`. Provider key used to supply initial state.
- `suppress_flow_events` (optional): boolean; default `false`. Disable flow event emission for this definition.
- `max_method_calls` (optional): integer; default `100`. Maximum number of method executions allowed during one kickoff.
- `defer_trace_finalization` (optional): boolean; default `false`. Defer trace finalization so callers can complete tracing later.
- `checkpoint` (optional): boolean | map of string to any | null; default `null`. Checkpointing configuration, or true to use default checkpointing.

### Cross-Field Rules

- A method has exactly one `do` action object with one `call` discriminator.
- `listen` targets method names and router-emitted event names in one shared namespace.
- Methods cannot listen to their own method name.
- A router method result must match one declared `emit` value.
- Crew action-level `inputs` are the Crew kickoff inputs; use CEL-wrapped strings there for runtime values.
- Crew agent/task interpolation uses `{name}` placeholders from evaluated crew inputs.
- Agent `with.input` must be text. Use `${outputs.method_name.raw}` or a text field like `${outputs.method_name.json_dict.summary}`.

