# Advanced Flow Patterns

## Table of Contents
- [Resumable Flows](#resumable-flows)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [Parallel Execution](#parallel-execution)
- [Error Handling and Retries](#error-handling-and-retries)
- [State Persistence](#state-persistence)

## Resumable Flows

Flows can be resumed from specific points using conditional starts:

```python
from crewai.flow.flow import Flow, start, listen

class ResumableFlow(Flow[State]):
    @start()  # Unconditional start
    def init(self):
        self.state.initialized = True
        return "initialized"

    @start("init")  # Conditional: runs after init OR as external trigger
    def maybe_begin(self):
        return "began"

    @listen(and_(init, maybe_begin))
    def proceed(self):
        return "proceeding"
```

### Resuming from External Trigger

```python
# Start from beginning
flow = MyFlow()
result = flow.kickoff()

# Resume from specific method
flow = MyFlow()
flow.state.previous_data = loaded_state
result = flow.kickoff(start_method="maybe_begin")
```

## Human-in-the-Loop (HITL)

### Pause for Human Review

```python
class HITLFlow(Flow[ReviewState]):
    @start()
    def generate_draft(self):
        agent = Agent(role="Writer", ...)
        result = agent.kickoff("Write initial draft")
        self.state.draft = result.raw
        return result

    @router(generate_draft)
    def check_confidence(self):
        if self.state.confidence < 0.7:
            return "needs_human_review"
        return "auto_approve"

    @listen("needs_human_review")
    def pause_for_review(self):
        # Save state for later resume
        self.state.status = "awaiting_review"
        self.state.save()  # Persist state
        return "Paused for human review"

    @listen("auto_approve")
    def proceed_automatically(self):
        return self.finalize()

    @start("human_approved")  # Resume point after human approval
    def after_human_review(self):
        # Human has reviewed and approved
        return self.finalize()

    def finalize(self):
        return {"final": self.state.draft}
```

### Integration with External Systems

```python
@listen("needs_approval")
def request_approval(self):
    # Send to Slack, email, or queue
    send_approval_request(
        content=self.state.draft,
        callback_id=self.state.flow_id
    )
    self.state.status = "pending_approval"
    return "Approval requested"
```

## Parallel Execution

### Multiple Start Points

```python
class ParallelDataFlow(Flow[DataState]):
    @start()
    def fetch_source_a(self):
        return api_client.get_data_a()

    @start()
    def fetch_source_b(self):
        return api_client.get_data_b()

    @start()
    def fetch_source_c(self):
        return api_client.get_data_c()

    @listen(and_(fetch_source_a, fetch_source_b, fetch_source_c))
    def merge_all_sources(self, a, b, c):
        self.state.merged = {**a, **b, **c}
        return self.state.merged
```

### Async Within Methods

```python
@listen(previous_step)
async def parallel_processing(self, data):
    # Process multiple items concurrently
    tasks = [
        self.process_item(item)
        for item in data["items"]
    ]
    results = await asyncio.gather(*tasks)
    return results

async def process_item(self, item):
    agent = Agent(role="Processor", ...)
    return await agent.kickoff_async(f"Process: {item}")
```

### Crew Batch Processing

```python
@listen(gather_inputs)
async def batch_analyze(self, inputs):
    crew = AnalysisCrew().crew()

    # Concurrent batch processing
    results = await crew.akickoff_for_each([
        {"item": item} for item in inputs
    ])

    self.state.analyses = [r.raw for r in results]
    return results
```

## Error Handling and Retries

### Try-Catch Pattern

```python
class RobustFlow(Flow[State]):
    @start()
    def risky_operation(self):
        try:
            result = external_api.call()
            self.state.result = result
            return "success"
        except APIError as e:
            self.state.error = str(e)
            self.state.retry_count = getattr(self.state, 'retry_count', 0) + 1
            return "error"

    @router(risky_operation)
    def handle_result(self):
        if hasattr(self.state, 'error'):
            if self.state.retry_count < 3:
                return "retry"
            return "failed"
        return "success"

    @listen("retry")
    def retry_operation(self):
        import time
        time.sleep(2 ** self.state.retry_count)  # Exponential backoff
        return self.risky_operation()

    @listen("failed")
    def handle_failure(self):
        return {"error": self.state.error, "retries": self.state.retry_count}
```

### Graceful Degradation

```python
@listen(process_step)
def with_fallback(self, primary_result):
    if not primary_result or primary_result.get("status") == "failed":
        # Fallback to simpler approach
        llm = LLM(model="gpt-4o-mini")
        return llm.call([{"role": "user", "content": "Simple fallback..."}])
    return primary_result
```

## State Persistence

### Pydantic State with Validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class PersistentState(BaseModel):
    flow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"
    data: dict = {}
    error: Optional[str] = None
    retry_count: int = 0

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid = ['pending', 'processing', 'completed', 'failed', 'paused']
        if v not in valid:
            raise ValueError(f'Status must be one of {valid}')
        return v

    def save(self):
        with open(f"state_{self.flow_id}.json", "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, flow_id: str):
        with open(f"state_{flow_id}.json") as f:
            return cls.model_validate_json(f.read())
```

### Using Persistent State

```python
class PersistentFlow(Flow[PersistentState]):
    @start()
    def begin_processing(self):
        self.state.status = "processing"
        self.state.save()
        return self.do_work()

    @listen(begin_processing)
    def checkpoint(self, result):
        self.state.data["checkpoint_1"] = result
        self.state.save()
        return result

# Resume from saved state
saved_state = PersistentState.load("flow-123")
flow = PersistentFlow()
flow.state = saved_state
flow.kickoff(start_method="checkpoint")
```

## Flow Composition

### Nested Flows

```python
class SubFlow(Flow[SubState]):
    @start()
    def sub_process(self):
        return "sub result"

class MainFlow(Flow[MainState]):
    @listen(setup)
    def run_subflow(self):
        sub = SubFlow()
        sub.state.input = self.state.data
        result = sub.kickoff()
        self.state.sub_result = result
        return result
```

### Flow Factory Pattern

```python
def create_flow(flow_type: str) -> Flow:
    flows = {
        "analysis": AnalysisFlow,
        "research": ResearchFlow,
        "writing": WritingFlow
    }
    return flows[flow_type]()

# Usage
flow = create_flow(config["flow_type"])
flow.state.input = data
result = flow.kickoff()
```
