# Flow System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/crewai/flow/flow.py](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py)
- [tests/test_multimodal_validation.py](https://github.com/crewAIInc/crewAI/blob/81bd81e5/tests/test_multimodal_validation.py)
- [tests/utilities/test_events.py](https://github.com/crewAIInc/crewAI/blob/81bd81e5/tests/utilities/test_events.py)

</details>



## Purpose and Scope

The Flow System provides event-driven workflow orchestration for precise control over execution sequences in CrewAI. Unlike the autonomous agent-driven approach of Crews (see [Core Components](#2)), Flows use declarative decorators to define exact execution paths, conditional branching, and state transitions. This system enables deterministic workflows where method execution is triggered by specific events and conditions.

For autonomous multi-agent coordination, see [Crew Orchestration](#2.1). For memory management within flows, see [Memory System](#4).

## Core Architecture

The Flow system is built around a metaclass-driven architecture that automatically discovers and registers decorated methods at class creation time. The `FlowMeta` metaclass processes method attributes to build internal registries during class construction.

### Flow Architecture Diagram
```mermaid
graph TB
    subgraph "Metaclass Processing"
        FlowMeta["FlowMeta"]
        MethodDiscovery["Method Discovery"]
        AttributeRegistry["Attribute Registry Building"]
    end
    
    subgraph "Flow Base Classes"
        FlowState["FlowState"]
        FlowGeneric["Flow[T]"]
        TypeVarT["TypeVar T"]
    end
    
    subgraph "Method Decorators"
        StartDecorator["@start()"]
        ListenDecorator["@listen()"]
        RouterDecorator["@router()"]
    end
    
    subgraph "Flow Instance Attributes"
        Methods["_methods: Dict[str, Callable]"]
        State["_state: T"]
        StartMethods["_start_methods: List[str]"]
        Listeners["_listeners: Dict[str, tuple]"]
        Routers["_routers: Set[str]"]
        RouterPaths["_router_paths: Dict[str, List[str]]"]
        Persistence["_persistence: FlowPersistence"]
        CompletedMethods["_completed_methods: Set[str]"]
        MethodOutputs["_method_outputs: List[Any]"]
        ExecutionCounts["_method_execution_counts: Dict[str, int]"]
        PendingAndListeners["_pending_and_listeners: Dict[str, Set[str]]"]
    end
    
    FlowMeta --> FlowGeneric
    FlowMeta --> MethodDiscovery
    MethodDiscovery --> AttributeRegistry
    AttributeRegistry --> StartMethods
    AttributeRegistry --> Listeners
    AttributeRegistry --> Routers
    AttributeRegistry --> RouterPaths
    
    FlowState --> FlowGeneric
    TypeVarT --> FlowGeneric
    
    StartDecorator --> Methods
    ListenDecorator --> Methods
    RouterDecorator --> Methods
    
    FlowGeneric --> Methods
    FlowGeneric --> State
    FlowGeneric --> StartMethods
    FlowGeneric --> Listeners
    FlowGeneric --> Routers
    FlowGeneric --> RouterPaths
    FlowGeneric --> Persistence
    FlowGeneric --> CompletedMethods
    FlowGeneric --> MethodOutputs
    FlowGeneric --> ExecutionCounts
    FlowGeneric --> PendingAndListeners
```

Sources: [src/crewai/flow/flow.py:392-433](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L392-L433), [src/crewai/flow/flow.py:436-449](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L436-L449), [src/crewai/flow/flow.py:49-56](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L49-L56), [src/crewai/flow/flow.py:458-512](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L458-L512)

### Flow Class Hierarchy

The `Flow` class uses a generic type parameter to support both structured (Pydantic models) and unstructured (dictionary) state management:

| Component | Purpose | Key Attributes |
|-----------|---------|----------------|
| `FlowMeta` | Metaclass for method discovery and registration | `_start_methods`, `_listeners`, `_routers`, `_router_paths` |
| `Flow[T]` | Base flow class with generic state type | `state: T`, `_methods: Dict`, `_persistence`, `tracing: bool` |
| `FlowState` | Base state model with UUID | `id: str` (auto-generated via `uuid4()`) |

The `FlowMeta` metaclass processes method attributes during class creation, scanning for:
- `__is_start_method__` - marks methods as flow entry points
- `__trigger_methods__` and `__condition_type__` - defines listener conditions  
- `__is_router__` - marks methods as routing decisions
- Router return constants via `get_possible_return_constants()`

The generic type parameter `T` is bound to `Union[Dict[str, Any], BaseModel]` allowing flexible state management approaches.

Sources: [src/crewai/flow/flow.py:392-433](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L392-L433), [src/crewai/flow/flow.py:436-449](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L436-L449), [src/crewai/flow/flow.py:49-56](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L49-L56), [src/crewai/flow/flow.py:58-65](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L58-L65)

## Flow Decorators and Conditions

### Core Decorators

The Flow system provides three primary decorators for method registration:

#### `@start()` Decorator
Marks methods as flow entry points. Can be unconditional or conditional based on other method completions:

```python
@start()  # Unconditional start
def begin_flow(self):
    pass

@start("method_name")  # Conditional start
def conditional_start(self):
    pass
```

Sources: [src/crewai/flow/flow.py:98-162](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L98-L162)

#### `@listen()` Decorator
Creates listeners that execute when specified conditions are met:

```python
@listen("process_data")  # Single method trigger
def handle_data(self):
    pass

@listen(or_("success", "failure"))  # Multiple method triggers
def handle_completion(self):
    pass
```

Sources: [src/crewai/flow/flow.py:165-222](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L165-L222)

#### `@router()` Decorator
Creates routing methods that dynamically determine flow paths:

```python
@router("check_status")
def route_based_on_status(self):
    if self.state.status == "success":
        return "SUCCESS_PATH"
    return "FAILURE_PATH"
```

Sources: [src/crewai/flow/flow.py:225-288](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L225-L288)

### Condition Combinators

Complex triggering conditions are created using `or_()` and `and_()` combinators:

#### Condition Logic Diagram
```mermaid
graph TB
    subgraph "OR Conditions"
        OrFunc["or_('method1', 'method2')"]
        OrResult["{type: 'OR', methods: ['method1', 'method2']}"]
        OrTrigger["Triggers when ANY method completes"]
    end
    
    subgraph "AND Conditions"
        AndFunc["and_('method1', 'method2')"]
        AndResult["{type: 'AND', methods: ['method1', 'method2']}"]
        AndTrigger["Triggers when ALL methods complete"]
    end
    
    OrFunc --> OrResult --> OrTrigger
    AndFunc --> AndResult --> AndTrigger
```

Sources: [src/crewai/flow/flow.py:291-334](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L291-L334), [src/crewai/flow/flow.py:337-380](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L337-L380)

The `and_()` combinator uses a pending listener mechanism to track which conditions remain unfulfilled before triggering execution.

## Flow Execution Model

### Execution Lifecycle

Flow execution follows a structured async lifecycle with comprehensive state management and event emission:

#### Flow Execution Sequence Diagram
```mermaid
sequenceDiagram
    participant Client
    participant Flow
    participant EventBus as "crewai_event_bus"
    participant Persistence as "FlowPersistence"
    participant Methods as "Flow Methods"
    
    Client->>Flow: kickoff(inputs)
    Flow->>Flow: kickoff_async(inputs)
    
    alt State Restoration
        Flow->>Persistence: load_state(inputs['id'])
        Persistence-->>Flow: stored_state
        Flow->>Flow: _restore_state(stored_state)
    else Fresh Execution
        Flow->>Flow: _initialize_state(inputs)
    end
    
    Flow->>EventBus: emit FlowStartedEvent
    
    loop For each start method in _start_methods
        Flow->>Flow: _execute_start_method(method_name)
        
        alt Method already completed (resumption)
            Flow->>Flow: Skip execution, continue listeners
        else Normal execution
            Flow->>Methods: _execute_method(method_name, method)
            Methods->>EventBus: emit MethodExecutionStartedEvent
            
            alt Sync method
                Methods->>Methods: Execute method
            else Async method  
                Methods->>Methods: await Execute method
            end
            
            Methods->>Flow: Store result in _method_outputs
            Methods->>EventBus: emit MethodExecutionFinishedEvent
            Flow->>Flow: _execute_listeners(method_name, result)
        end
    end
    
    loop Router Chain Processing
        Flow->>Flow: _find_triggered_methods(trigger, router_only=True)
        Flow->>Methods: Execute routers sequentially
        Methods->>Flow: Return router path constant
        Flow->>Flow: Use router result as new trigger
    end
    
    loop Normal Listener Processing  
        Flow->>Flow: _find_triggered_methods(trigger, router_only=False)
        Flow->>Methods: Execute listeners in parallel (asyncio.gather)
        Methods->>Flow: _execute_listeners(listener_name, result)
    end
    
    Flow->>EventBus: emit FlowFinishedEvent
    Flow->>Client: Return final output from _method_outputs[-1]
```

The execution model handles both synchronous and asynchronous methods via `asyncio.iscoroutinefunction()` checks, and maintains detailed execution state for resumption scenarios.

Sources: [src/crewai/flow/flow.py:795-903](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L795-L903), [src/crewai/flow/flow.py:808-904](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L808-L904), [src/crewai/flow/flow.py:905-938](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L905-L938), [src/crewai/flow/flow.py:975-1027](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L975-L1027)

### Method Triggering Logic

The flow execution engine uses a sophisticated triggering system that separates router and listener execution phases:

#### Triggering Logic Flow
```mermaid
graph TB
    subgraph "Method Completion Event"
        TriggerMethod["trigger_method: str"]
        MethodResult["result: Any"]
    end
    
    subgraph "Router Phase (Sequential)"
        FindRouters["_find_triggered_methods(trigger, router_only=True)"]
        ExecuteRouter["_execute_single_listener(router_name, result)"]
        RouterResult["Router returns path constant"]
        NewTrigger["Use router result as new trigger"]
    end
    
    subgraph "Listener Phase (Parallel)"
        FindListeners["_find_triggered_methods(trigger, router_only=False)"]
        ExecuteListeners["asyncio.gather(*listener_tasks)"]
        ListenerExecution["_execute_single_listener(listener_name, result)"]
    end
    
    subgraph "Condition Evaluation"
        OrCondition["OR: trigger_method in methods"]
        AndCondition["AND: _pending_and_listeners tracking"]
        AndState["Remove trigger from pending set"]
        AndCheck["Check if pending set empty"]
    end
    
    TriggerMethod --> FindRouters
    FindRouters --> ExecuteRouter
    ExecuteRouter --> RouterResult
    RouterResult --> NewTrigger
    NewTrigger --> FindRouters
    
    TriggerMethod --> FindListeners
    FindListeners --> ExecuteListeners
    ExecuteListeners --> ListenerExecution
    
    FindRouters --> OrCondition
    FindRouters --> AndCondition
    FindListeners --> OrCondition
    FindListeners --> AndCondition
    
    AndCondition --> AndState
    AndState --> AndCheck
```

The `_find_triggered_methods()` function evaluates both OR and AND conditions:

| Condition Type | Evaluation Logic | State Management |
|----------------|------------------|------------------|
| OR | `trigger_method in methods` | Immediate execution when any condition met |
| AND | Track completion in `_pending_and_listeners[listener_name]` | Execute when `pending_set` becomes empty |
| Router Separation | `router_only` parameter distinguishes execution phases | Routers execute sequentially, listeners in parallel |

Sources: [src/crewai/flow/flow.py:1110-1168](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1110-L1168), [src/crewai/flow/flow.py:1029-1109](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1029-L1109), [src/crewai/flow/flow.py:1170-1226](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1170-L1226)

### Router Execution Pattern

Routers execute sequentially in chains, with each router's output becoming a new trigger for subsequent routers and listeners. The router chain continues until no more routers are triggered:

#### Router Chain Execution Flow
```mermaid
graph TD
    subgraph "Router Chain Loop"
        CurrentTrigger["current_trigger = trigger_method"]
        FindRouters["routers_triggered = _find_triggered_methods(current_trigger, router_only=True)"]
        CheckRouters{"routers_triggered?"}
        ExecuteRouter["_execute_single_listener(router_name, result)"]
        GetRouterResult["router_result = _method_outputs[-1]"]
        UpdateTrigger["current_trigger = str(router_result)"]
        StoreResult["router_results.append(router_result)"]
    end
    
    subgraph "Listener Processing"
        AllTriggers["all_triggers = [trigger_method] + router_results"]
        ProcessTriggers["For each trigger in all_triggers"]
        FindListeners["listeners = _find_triggered_methods(trigger, router_only=False)"]
        ExecuteListeners["asyncio.gather(*listener_tasks)"]
    end
    
    subgraph "Cyclic Flow Support"
        CheckStartMethods["Check if router result triggers start methods"]
        CyclicExecution["Execute start method for cycles"]
        TempClearResumption["Temporarily clear _is_execution_resuming"]
    end
    
    CurrentTrigger --> FindRouters
    FindRouters --> CheckRouters
    CheckRouters -->|Yes| ExecuteRouter
    ExecuteRouter --> GetRouterResult
    GetRouterResult --> StoreResult
    StoreResult --> UpdateTrigger
    UpdateTrigger --> FindRouters
    CheckRouters -->|No| AllTriggers
    
    AllTriggers --> ProcessTriggers
    ProcessTriggers --> FindListeners
    FindListeners --> ExecuteListeners
    
    ProcessTriggers --> CheckStartMethods
    CheckStartMethods --> CyclicExecution
    CyclicExecution --> TempClearResumption
```

The router chain supports cyclic flows where router results can re-trigger start methods that were previously completed, enabling complex branching and looping patterns.

Sources: [src/crewai/flow/flow.py:1052-1109](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1052-L1109), [src/crewai/flow/flow.py:1092-1108](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1092-L1108)

## State Management

The Flow system supports both structured (Pydantic) and unstructured (dictionary) state management with automatic UUID generation.

### State Architecture Diagram
```mermaid
graph TB
    subgraph "State Types"
        DictState["Dict State<br/>{id: str, ...}"]
        StructuredState["Pydantic State<br/>BaseModel with id field"]
        FlowStateBase["FlowState<br/>(BaseModel base class)"]
    end
    
    subgraph "State Operations"
        InitState["_create_initial_state()"]
        UpdateState["_initialize_state(inputs)"]
        RestoreState["_restore_state(stored_state)"]
        CopyState["_copy_state()"]
    end
    
    subgraph "State Persistence"
        FlowPersistence["FlowPersistence"]
        LoadState["load_state(uuid)"]
        SaveState["save_state(uuid, state)"]
    end
    
    FlowStateBase --> StructuredState
    InitState --> DictState
    InitState --> StructuredState
    
    UpdateState --> DictState
    UpdateState --> StructuredState
    
    RestoreState --> FlowPersistence
    LoadState --> RestoreState
    SaveState --> FlowPersistence
```

Sources: [src/crewai/flow/flow.py:497-577](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L497-L577), [src/crewai/flow/flow.py:624-676](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L624-L676), [src/crewai/flow/flow.py:678-710](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L678-L710)

### State Initialization

State initialization follows a priority order:
1. Restore from persistence if `id` provided
2. Apply input parameters
3. Ensure UUID is present
4. Validate structured state models

The `_create_initial_state()` method handles type detection and UUID generation automatically.

Sources: [src/crewai/flow/flow.py:497-577](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L497-L577)

### State Access Patterns

| Access Method | Return Type | Use Case |
|---------------|-------------|----------|
| `flow.state` | `T` (generic) | Direct state access |
| `flow.flow_id` | `str` | Safe UUID access |
| `flow._copy_state()` | `T` | Event emission snapshots |

The `flow_id` property provides safe access to the state UUID regardless of state type (dict or BaseModel).

Sources: [src/crewai/flow/flow.py:582-623](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L582-L623)

### Event Emission and Observability

The Flow system provides comprehensive observability through the `crewai_event_bus` with detailed event emission at every execution stage:

#### Flow Event Emission Architecture
```mermaid
graph TB
    subgraph "Flow Lifecycle Events"
        FlowCreatedEvent["FlowCreatedEvent<br/>(on __init__)"]
        FlowStartedEvent["FlowStartedEvent<br/>(on kickoff_async)"]
        FlowFinishedEvent["FlowFinishedEvent<br/>(on completion)"]
        FlowPlotEvent["FlowPlotEvent<br/>(on plot())"]
    end
    
    subgraph "Method Execution Events"
        MethodExecutionStartedEvent["MethodExecutionStartedEvent<br/>(method start)"]
        MethodExecutionFinishedEvent["MethodExecutionFinishedEvent<br/>(method success)"]
        MethodExecutionFailedEvent["MethodExecutionFailedEvent<br/>(method exception)"]
    end
    
    subgraph "Event Context Data"
        FlowContext["flow_name, inputs, timestamp"]
        MethodContext["method_name, params, state, result"]
        StateSnapshot["_copy_state() for state snapshots"]
        ErrorContext["method_name, flow_name, error"]
    end
    
    subgraph "crewai_event_bus Integration"
        EventBus["crewai_event_bus"]
        EventEmit["emit(source=self, event=event_instance)"]
        OpenTelemetryBaggage["baggage.set_baggage('flow_inputs', inputs)"]
        TracingIntegration["TraceCollectionListener setup"]
    end
    
    FlowCreatedEvent --> EventBus
    FlowStartedEvent --> EventBus
    FlowFinishedEvent --> EventBus
    FlowPlotEvent --> EventBus
    
    MethodExecutionStartedEvent --> EventBus
    MethodExecutionFinishedEvent --> EventBus
    MethodExecutionFailedEvent --> EventBus
    
    FlowContext --> FlowCreatedEvent
    FlowContext --> FlowStartedEvent
    FlowContext --> FlowFinishedEvent
    
    MethodContext --> MethodExecutionStartedEvent
    MethodContext --> MethodExecutionFinishedEvent
    ErrorContext --> MethodExecutionFailedEvent
    
    StateSnapshot --> MethodExecutionStartedEvent
    StateSnapshot --> MethodExecutionFinishedEvent
    
    EventBus --> EventEmit
    EventBus --> OpenTelemetryBaggage
    EventBus --> TracingIntegration
```

The Flow system supports tracing integration via the `tracing` parameter and OpenTelemetry baggage for context propagation. The `_copy_state()` method creates deep copies for event snapshots to prevent state mutation.

Sources: [src/crewai/flow/flow.py:28-42](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L28-L42), [src/crewai/flow/flow.py:489-496](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L489-L496), [src/crewai/flow/flow.py:866-899](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L866-L899), [src/crewai/flow/flow.py:978-1026](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L978-L1026), [src/crewai/flow/flow.py:1254-1262](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1254-L1262)

### Event Data Structure

Each event provides comprehensive execution context for observability and debugging:

| Event Type | Key Fields | Emission Context |
|------------|------------|------------------|
| `FlowCreatedEvent` | `flow_name`, `timestamp`, `type="flow_created"` | Flow instance initialization |
| `FlowStartedEvent` | `flow_name`, `inputs`, `timestamp`, `type="flow_started"` | `kickoff_async()` entry |
| `MethodExecutionStartedEvent` | `method_name`, `flow_name`, `params`, `state`, `type="method_execution_started"` | Before method execution in `_execute_method()` |
| `MethodExecutionFinishedEvent` | `method_name`, `flow_name`, `state`, `result`, `type="method_execution_finished"` | After successful method execution |
| `MethodExecutionFailedEvent` | `method_name`, `flow_name`, `error`, `type="method_execution_failed"` | Exception handling in `_execute_method()` |
| `FlowFinishedEvent` | `flow_name`, `result`, `timestamp`, `type="flow_finished"` | Flow completion with final output |
| `FlowPlotEvent` | `flow_name`, `type="flow_plot"` | `plot()` method invocation |

Event emission uses `crewai_event_bus.emit(self, event_instance)` where `self` is the Flow instance as the event source. State snapshots are created via `self._copy_state()` to prevent mutation during event processing.

The comprehensive event structure enables integration with observability platforms, debugging tools, and real-time flow monitoring systems.

Sources: [src/crewai/flow/flow.py:489-496](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L489-L496), [src/crewai/flow/flow.py:866-899](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L866-L899), [src/crewai/flow/flow.py:982-1013](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L982-L1013), [src/crewai/flow/flow.py:1018-1026](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1018-L1026), [tests/utilities/test_events.py:436-510](https://github.com/crewAIInc/crewAI/blob/81bd81e5/tests/utilities/test_events.py#L436-L510)

## Visualization and Persistence

### Flow Visualization

The `plot()` method generates visual representations of flow structure using the `plot_flow()` function:

```python
flow.plot("my_flow_diagram")  # Generates flow visualization
```

This emits a `FlowPlotEvent` and creates graphical representations showing method dependencies and execution paths.

Sources: [src/crewai/flow/flow.py:1075-1083](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L1075-L1083), [src/crewai/flow/flow_visualizer.py:22](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow_visualizer.py#L22)

### Flow Persistence and Restoration

Flow state persistence enables resuming interrupted executions through the `FlowPersistence` interface and the `reload()` mechanism:

#### Flow Persistence and Reload Architecture
```mermaid
graph TB
    subgraph "Persistence Interface"
        FlowPersistence["FlowPersistence"]
        LoadState["load_state(uuid: str)"]
        SaveState["save_state(uuid: str, state: Dict)"]
    end
    
    subgraph "State Restoration Process"
        CheckInputs["inputs contains 'id'?"]
        LoadStoredState["stored_state = persistence.load_state(inputs['id'])"]
        RestoreState["_restore_state(stored_state)"]
        InitializeState["_initialize_state(filtered_inputs)"]
    end
    
    subgraph "Execution Data Reload"
        ReloadMethod["reload(execution_data: FlowExecutionData)"]
        ExtractFlowId["Update state with execution_data['id']"]
        BuildCompletedMethods["_completed_methods from completed_methods"]
        ApplyFinalState["Apply final_state from last execution_method"]
        SortExecutionMethods["Sort by started_at timestamp"]
    end
    
    subgraph "State Application"
        UpdateStateField["_update_state_field(field_name, value)"]
        ApplyStateUpdates["_apply_state_updates(updates: Dict)"]
        StateFieldUpdate["Dict: direct assignment / BaseModel: setattr"]
    end
    
    subgraph "Execution Resumption"
        IsExecutionResuming["_is_execution_resuming: bool"]
        CompletedMethodsSet["_completed_methods: Set[str]"]
        SkipCompletedMethods["Skip execution, continue listeners"]
        CyclicFlowSupport["Clear completed for re-execution"]
    end
    
    FlowPersistence --> LoadState
    FlowPersistence --> SaveState
    
    CheckInputs -->|Yes| LoadStoredState
    LoadStoredState --> RestoreState
    CheckInputs -->|No| InitializeState
    
    ReloadMethod --> ExtractFlowId
    ReloadMethod --> BuildCompletedMethods
    ReloadMethod --> ApplyFinalState
    ApplyFinalState --> SortExecutionMethods
    
    RestoreState --> UpdateStateField
    ApplyFinalState --> ApplyStateUpdates
    UpdateStateField --> StateFieldUpdate
    ApplyStateUpdates --> StateFieldUpdate
    
    BuildCompletedMethods --> CompletedMethodsSet
    CompletedMethodsSet --> IsExecutionResuming
    IsExecutionResuming --> SkipCompletedMethods
    SkipCompletedMethods --> CyclicFlowSupport
```

The persistence system supports two restoration modes:
1. **Simple State Restoration**: Using `inputs['id']` during `kickoff_async()` 
2. **Execution Data Reload**: Using `reload(execution_data)` with complete execution history

State restoration handles both dictionary and BaseModel state types through `_restore_state()`, while execution resumption uses `_completed_methods` tracking to skip already-executed methods.

Sources: [src/crewai/flow/flow.py:694-727](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L694-L727), [src/crewai/flow/flow.py:728-793](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L728-L793), [src/crewai/flow/flow.py:779-793](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L779-L793), [src/crewai/flow/flow.py:846-863](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L846-L863), [src/crewai/flow/flow.py:925-932](https://github.com/crewAIInc/crewAI/blob/81bd81e5/src/crewai/flow/flow.py#L925-L932)