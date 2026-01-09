# Multi-Agent Design Patterns

Advanced patterns for orchestrating multiple agents in complex applications. These patterns address common challenges: quality assurance, dynamic scaling, fault tolerance, and human oversight.

## Pattern Selection Guide

### Decision Tree

```
What is your primary concern?
├── Output Quality Critical → Generator-Critic or Task Guardrails
├── Iterative Improvement → Iterative Refinement
├── Unknown Task Complexity → Orchestrator-Worker (Dynamic Spawning)
├── Independent Parallel Tasks → Parallel Fan-Out
├── Coordinator Needed → Hierarchical with Custom Manager
├── Crash Recovery / Long Tasks → State Persistence
├── Human Approval Required → HITL Webhooks
└── Enterprise Production → Composite (combine patterns)
```

### Quick Reference

| Pattern | Best For | Key CrewAI Construct |
|---------|----------|---------------------|
| **Generator-Critic** | Quality gates, validation loops | Flow + @router + "revise" loop |
| **Iterative Refinement** | Progressive improvement | Flow + iteration counter + exit condition |
| **Orchestrator-Worker** | Dynamic task decomposition | Flow + inline Agent() + asyncio.gather |
| **Parallel Fan-Out** | Concurrent independent tasks | asyncio.gather + akickoff() |
| **Task Guardrails** | Output validation | Task(guardrail=func, guardrail_max_retries=N) |
| **State Persistence** | Crash recovery, HITL | @persist() on Flow class |
| **HITL Webhooks** | Enterprise approval flows | humanInputWebhook + /resume API |
| **Custom Manager** | Coordinated delegation | Crew(manager_agent=..., process=Process.hierarchical) |
| **Composite** | Enterprise systems | Nested patterns combined |

---

## Generator-Critic Pattern

**Use when:** Output quality is critical and requires validation before acceptance (legal, medical, financial content).

**Concept:** One agent/crew generates content, another critiques it. Based on quality score, either approve or loop back for revision.

```python
from crewai.flow.flow import Flow, start, listen, router
from crewai import Crew
from pydantic import BaseModel

class GeneratorCriticState(BaseModel):
    content: str = ""
    critique: str = ""
    quality_score: float = 0.0
    iteration: int = 0
    max_iterations: int = 3

class GeneratorCriticFlow(Flow[GeneratorCriticState]):

    @start()
    def generate(self):
        """Generator crew creates content"""
        crew = GeneratorCrew().crew()
        result = crew.kickoff(inputs={"topic": self.state.topic})
        self.state.content = result.raw
        self.state.iteration += 1
        return result

    @listen(generate)
    def critique(self, generated):
        """Critic crew evaluates quality"""
        crew = CriticCrew().crew()
        result = crew.kickoff(inputs={"content": self.state.content})
        self.state.critique = result.raw
        self.state.quality_score = result.pydantic.score  # Assumes structured output
        return result

    @router(critique)
    def check_quality(self):
        if self.state.quality_score >= 0.8:
            return "approved"
        elif self.state.iteration >= self.state.max_iterations:
            return "max_iterations"
        return "revise"

    @listen("revise")
    def revise_content(self):
        """Feed critique back to generator"""
        crew = GeneratorCrew().crew()
        result = crew.kickoff(inputs={
            "topic": self.state.topic,
            "previous_attempt": self.state.content,
            "feedback": self.state.critique
        })
        self.state.content = result.raw
        self.state.iteration += 1
        return result

    @listen(revise_content)
    def re_critique(self, revised):
        """Loop back to critique"""
        return self.critique(revised)

    @listen("approved")
    def finalize(self):
        return {"content": self.state.content, "iterations": self.state.iteration}

    @listen("max_iterations")
    def handle_max(self):
        return {"content": self.state.content, "warning": "Max iterations reached"}
```

**When NOT to use:** Simple tasks where quality is easily validated, or when iteration cost is too high.

---

## Iterative Refinement Pattern

**Use when:** Output requires progressive improvement toward a quality threshold (optimization, polishing).

**Difference from Generator-Critic:** Focuses on continuous improvement rather than pass/fail validation.

```python
from crewai.flow.flow import Flow, start, listen, router
from crewai import Agent, LLM
from pydantic import BaseModel

class RefinementState(BaseModel):
    draft: str = ""
    quality_score: float = 0.0
    iteration: int = 0
    max_iterations: int = 5
    target_quality: float = 0.9

class IterativeRefinementFlow(Flow[RefinementState]):

    @start()
    def create_initial(self):
        """Create initial draft"""
        writer = Agent(
            role="Content Writer",
            goal="Create high-quality initial draft",
            llm=LLM(model="gpt-4o")
        )
        result = writer.kickoff(f"Write about: {self.state.topic}")
        self.state.draft = result.raw
        return result

    @listen(create_initial)
    def assess_and_refine(self, draft):
        """Assess quality and refine if needed"""
        self.state.iteration += 1

        # Assess quality
        llm = LLM(model="gpt-4o", response_format=QualityAssessment)
        assessment = llm.call([
            {"role": "user", "content": f"Score 0-1 and suggest improvements:\n\n{self.state.draft}"}
        ])
        self.state.quality_score = assessment.score

        if self.state.quality_score >= self.state.target_quality:
            return "converged"
        if self.state.iteration >= self.state.max_iterations:
            return "max_reached"

        # Refine
        refiner = Agent(role="Editor", goal="Improve based on feedback", llm=LLM(model="gpt-4o"))
        result = refiner.kickoff(f"Improve:\n{self.state.draft}\n\nFeedback:\n{assessment.improvements}")
        self.state.draft = result.raw
        return "continue"

    @router(assess_and_refine)
    def route(self):
        # Router returns the string from assess_and_refine
        pass  # The return from assess_and_refine is the route

    @listen("continue")
    def continue_refinement(self):
        """Loop back for another iteration"""
        return self.assess_and_refine(self.state.draft)

    @listen("converged")
    def output_final(self):
        return {"content": self.state.draft, "iterations": self.state.iteration}
```

---

## Orchestrator-Worker Pattern (Dynamic Spawning)

**Use when:** Task complexity is unknown upfront and requires dynamic scaling of subagents.

**Concept:** Lead agent analyzes complexity, spawns appropriate number of specialized workers, then synthesizes results.

```python
from crewai.flow.flow import Flow, start, listen, router
from crewai import Agent, LLM
from pydantic import BaseModel
import asyncio

class OrchestratorState(BaseModel):
    query: str = ""
    complexity: str = "moderate"
    worker_count: int = 1
    worker_results: list[str] = []
    synthesis: str = ""

class OrchestratorWorkerFlow(Flow[OrchestratorState]):

    @start()
    def analyze_complexity(self):
        """Orchestrator analyzes task and plans"""
        llm = LLM(model="gpt-4o", response_format=TaskPlan)
        plan = llm.call([
            {"role": "system", "content": "Analyze query complexity. Return subtasks."},
            {"role": "user", "content": f"Query: {self.state.query}"}
        ])

        self.state.complexity = plan.complexity
        # Scale workers to complexity: simple=1, moderate=3, complex=5
        complexity_map = {"simple": 1, "moderate": 3, "complex": 5}
        self.state.worker_count = complexity_map.get(plan.complexity, 3)
        return plan

    @listen(analyze_complexity)
    async def spawn_workers(self, plan):
        """Dynamically create and run worker agents in parallel"""
        worker_tasks = []

        for i, subtask in enumerate(plan.subtasks[:self.state.worker_count]):
            # Create fresh agent per subtask (clean context)
            agent = Agent(
                role=f"Research Specialist #{i+1}",
                goal=subtask.objective,
                backstory=f"Expert in {subtask.domain}",
                tools=[SearchTool()],
                llm=LLM(model="gpt-4o-mini")  # Cheaper model for workers
            )
            worker_tasks.append(agent.akickoff(subtask.instructions))

        # Run all workers concurrently
        results = await asyncio.gather(*worker_tasks)
        self.state.worker_results = [r.raw for r in results]
        return results

    @listen(spawn_workers)
    def synthesize(self, worker_outputs):
        """Orchestrator synthesizes all findings"""
        synthesizer = Agent(
            role="Lead Researcher",
            goal="Synthesize findings into coherent answer",
            llm=LLM(model="gpt-4o")  # Best model for synthesis
        )

        combined = "\n\n---\n\n".join([
            f"Worker {i+1}:\n{result}" for i, result in enumerate(self.state.worker_results)
        ])

        result = synthesizer.kickoff(
            f"Query: {self.state.query}\n\nFindings:\n{combined}\n\nSynthesize into answer."
        )
        self.state.synthesis = result.raw
        return result
```

**Scaling guidance:**
- Simple queries: 1 worker, 3-10 tool calls
- Moderate: 3 workers in parallel
- Complex: 5+ workers, may need multiple rounds

---

## Parallel Fan-Out Pattern

**Use when:** Multiple independent subtasks can run concurrently for speed.

```python
from crewai.flow.flow import Flow, start, listen, and_
from crewai import Crew
import asyncio

class ParallelState(BaseModel):
    data: dict = {}
    security_result: str = ""
    performance_result: str = ""
    style_result: str = ""
    final_report: str = ""

class ParallelAnalysisFlow(Flow[ParallelState]):

    @start()
    async def fan_out_analysis(self):
        """Run multiple crews in parallel"""
        results = await asyncio.gather(
            SecurityCrew().crew().akickoff(inputs=self.state.data),
            PerformanceCrew().crew().akickoff(inputs=self.state.data),
            StyleCrew().crew().akickoff(inputs=self.state.data)
        )

        self.state.security_result = results[0].raw
        self.state.performance_result = results[1].raw
        self.state.style_result = results[2].raw
        return results

    @listen(fan_out_analysis)
    def gather_and_synthesize(self, parallel_results):
        """Aggregate all parallel results"""
        llm = LLM(model="gpt-4o")
        self.state.final_report = llm.call([
            {"role": "user", "content": f"""Combine analyses:
            Security: {self.state.security_result}
            Performance: {self.state.performance_result}
            Style: {self.state.style_result}
            """}
        ])
        return self.state.final_report
```

**Batch processing with akickoff_for_each:**
```python
async def batch_analysis(self):
    datasets = [{"id": 1, ...}, {"id": 2, ...}, {"id": 3, ...}]
    results = await AnalysisCrew().crew().akickoff_for_each(datasets)
    return results
```

---

## Task Guardrails Pattern

**Use when:** Task outputs must meet specific validation criteria before acceptance.

**Key feature:** Automatic retry with feedback when validation fails.

```python
from typing import Tuple, Any
from crewai import Task, TaskOutput

def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate output is valid JSON"""
    try:
        data = json.loads(result.raw)
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON: {e}")

def validate_length(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate minimum length"""
    if len(result.raw) < 100:
        return (False, "Output too short, needs more detail")
    return (True, result.raw)

def validate_no_pii(result: TaskOutput) -> Tuple[bool, Any]:
    """Check for PII"""
    pii_patterns = ["SSN:", "credit card"]
    for pattern in pii_patterns:
        if pattern.lower() in result.raw.lower():
            return (False, f"Contains PII pattern: {pattern}")
    return (True, result.raw)

# Single guardrail
task = Task(
    description="Generate JSON report",
    expected_output="Valid JSON object",
    agent=analyst,
    guardrail=validate_json_output,
    guardrail_max_retries=3  # Retry up to 3x if validation fails
)

# Multiple sequential guardrails
task = Task(
    description="Generate customer report",
    expected_output="Detailed report without PII",
    agent=writer,
    guardrails=[validate_length, validate_no_pii],  # Run in order
    guardrail_max_retries=3
)
```

**When validation fails:** The agent receives feedback and retries automatically.

---

## State Persistence Pattern

**Use when:** Long-running flows need crash recovery, or HITL requires resume capability.

```python
from crewai.flow.flow import Flow, start, listen
from crewai.flow.persistence import persist
from pydantic import BaseModel

class LongRunningState(BaseModel):
    step: str = "initialized"
    checkpoint_data: dict = {}
    results: list[str] = []

@persist()  # Saves state after EVERY method
class ResilientFlow(Flow[LongRunningState]):

    @start()
    def phase_one(self):
        self.state.step = "phase_one_complete"
        self.state.checkpoint_data["phase1"] = "data"
        # If crash here, flow resumes from this state
        return "Phase 1 done"

    @listen(phase_one)
    def phase_two(self, prev):
        self.state.step = "phase_two_complete"
        # State automatically persisted
        return "Phase 2 done"

    @listen(phase_two)
    def phase_three(self, prev):
        self.state.step = "complete"
        return "All phases done"

# First run (crashes at phase_two)
flow1 = ResilientFlow()
result1 = flow1.kickoff()

# Second run - automatically resumes from persisted state
flow2 = ResilientFlow()
result2 = flow2.kickoff()  # Continues from last checkpoint
```

**Integration with HITL:**
```python
@persist()
class ApprovalFlow(Flow[ApprovalState]):

    @listen(generate_content)
    def await_approval(self, content):
        self.state.awaiting_approval = True
        self.state.pending_content = content
        # State persisted - can resume after human approves
        return "Awaiting human approval"
```

---

## HITL Webhooks Pattern (Enterprise)

**Use when:** Enterprise deployments require human approval workflows with external system integration.

```python
import requests

BASE_URL = "https://your-crewai-deployment.com"

# Kickoff with human input webhook
response = requests.post(f"{BASE_URL}/kickoff", json={
    "inputs": {"topic": "Quarterly Report"},
    "humanInputWebhook": {
        "url": "https://your-app.com/hitl-callback",
        "authentication": {
            "strategy": "bearer",
            "token": "your-secret-token"
        }
    }
})

execution_id = response.json()["execution_id"]

# When human reviews (via your webhook handler)
# Resume with approval/feedback
def resume_after_human_review(execution_id, task_id, approved, feedback=""):
    response = requests.post(f"{BASE_URL}/resume", json={
        "execution_id": execution_id,
        "task_id": task_id,
        "human_feedback": feedback,
        "is_approve": approved
    })
    return response.json()

# Approve and continue
resume_after_human_review(execution_id, "review_task", True, "Looks good!")

# Reject and retry
resume_after_human_review(execution_id, "review_task", False, "Needs more data on Q3")
```

---

## Custom Manager Pattern

**Use when:** Hierarchical coordination requires specialized management behavior.

**`manager_agent` vs `manager_llm`:**
- Use `manager_llm` for simple coordination with default behavior
- Use `manager_agent` for custom role, goals, and delegation rules

```python
from crewai import Agent, Crew, Task, Process

# Custom manager with specific behavior
manager = Agent(
    role="Senior Project Manager",
    goal="Coordinate team efficiently, prioritize quality over speed",
    backstory="""Experienced PM who excels at delegation.
    Always validates work before final delivery.
    Escalates blockers immediately.""",
    allow_delegation=True,  # Required for manager
    verbose=True
)

# Specialist agents
researcher = Agent(
    role="Research Analyst",
    goal="Provide accurate, thorough research",
    allow_delegation=False  # Specialists don't delegate
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear, accurate documentation",
    allow_delegation=False
)

# Hierarchical crew with custom manager
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[
        Task(description="Research and document API changes", agent=manager)
    ],
    process=Process.hierarchical,
    manager_agent=manager  # Use custom manager
)

# Alternative: Use LLM as manager (simpler)
crew_simple = Crew(
    agents=[researcher, writer],
    tasks=[...],
    process=Process.hierarchical,
    manager_llm="gpt-4o"  # Default manager behavior
)
```

---

## Post-Processing Pattern

**Use when:** Final output requires dedicated processing (citations, formatting, compliance).

```python
class PostProcessingFlow(Flow[ContentState]):

    @listen(research_complete)
    def add_citations(self, raw_research):
        """Dedicated agent for citations"""
        citation_agent = Agent(
            role="Citation Specialist",
            goal="Add proper academic citations",
            backstory="Editor with expertise in attribution",
            llm=LLM(model="gpt-4o")
        )

        result = citation_agent.kickoff(
            f"Add inline citations [Author, Year] to:\n{raw_research.raw}\n"
            f"Sources:\n{json.dumps(self.state.sources)}"
        )
        self.state.cited_content = result.raw
        return result

    @listen(add_citations)
    def format_for_publication(self, cited):
        """Formatting specialist"""
        formatter = Agent(
            role="Publication Formatter",
            goal="Format for target publication",
            llm=LLM(model="gpt-4o")
        )
        return formatter.kickoff(f"Format for {self.state.target_format}:\n{cited.raw}")
```

---

## Composite Patterns

**Use when:** Enterprise applications require multiple patterns working together.

### Example: Customer Support System

Combines: Coordinator, Parallel Fan-Out, Generator-Critic, Task Guardrails, State Persistence, HITL.

```python
from crewai.flow.flow import Flow, start, listen, router, or_
from crewai.flow.persistence import persist
from crewai import Agent, Crew, Task, Process
import asyncio

class SupportState(BaseModel):
    ticket: dict = {}
    category: str = ""
    parallel_analyses: dict = {}
    draft_response: str = ""
    quality_score: float = 0.0
    iteration: int = 0
    escalated: bool = False

@persist()  # Crash recovery
class CustomerSupportFlow(Flow[SupportState]):

    @start()
    def classify_ticket(self):
        """Route to appropriate specialist"""
        llm = LLM(model="gpt-4o", response_format=TicketClassification)
        result = llm.call([
            {"role": "user", "content": f"Classify: {self.state.ticket}"}
        ])
        self.state.category = result.category
        return result

    @router(classify_ticket)
    def route_to_specialist(self):
        if self.state.category in ["billing", "technical", "account"]:
            return self.state.category
        return "general"

    @listen("billing")
    async def handle_billing(self):
        """Parallel analysis for billing issues"""
        results = await asyncio.gather(
            BillingCrew().crew().akickoff(inputs=self.state.ticket),
            ComplianceCrew().crew().akickoff(inputs=self.state.ticket)
        )
        self.state.parallel_analyses = {
            "billing": results[0].raw,
            "compliance": results[1].raw
        }
        return results

    @listen("technical")
    def handle_technical(self):
        """Hierarchical crew for technical issues"""
        manager = Agent(role="Tech Lead", allow_delegation=True)
        crew = Crew(
            agents=[manager, Agent(role="Backend Expert"), Agent(role="Frontend Expert")],
            tasks=[Task(description=f"Resolve: {self.state.ticket}", agent=manager)],
            process=Process.hierarchical,
            manager_agent=manager
        )
        result = crew.kickoff()
        self.state.parallel_analyses["technical"] = result.raw
        return result

    @listen(or_("billing", "technical", "account", "general"))
    def generate_response(self):
        """Generator creates response"""
        crew = ResponseCrew().crew()
        result = crew.kickoff(inputs={
            "ticket": self.state.ticket,
            "analyses": self.state.parallel_analyses
        })
        self.state.draft_response = result.raw
        self.state.iteration += 1
        return result

    @listen(generate_response)
    def critique_response(self, draft):
        """Critic evaluates quality"""
        # Task guardrail validates tone
        task = Task(
            description="Critique response for tone and accuracy",
            agent=QACritic(),
            guardrail=validate_professional_tone,
            guardrail_max_retries=2
        )
        crew = Crew(agents=[QACritic()], tasks=[task])
        result = crew.kickoff(inputs={"response": self.state.draft_response})
        self.state.quality_score = result.pydantic.score
        return result

    @router(critique_response)
    def quality_gate(self):
        if self.state.quality_score >= 0.85:
            return "approved"
        elif self.state.iteration >= 2:
            return "escalate"  # HITL escalation
        return "revise"

    @listen("revise")
    def revise_response(self):
        # Loop back to generator with feedback
        return self.generate_response()

    @listen("escalate")
    def escalate_to_human(self):
        """HITL escalation"""
        self.state.escalated = True
        return {"status": "escalated", "draft": self.state.draft_response}

    @listen("approved")
    def send_response(self):
        return {"status": "sent", "response": self.state.draft_response}
```

---

## Anti-Patterns

### 1. Over-Engineering Simple Tasks
**Wrong:** Using Orchestrator-Worker for a simple classification task.
**Right:** Use direct LLM call with structured output.

### 2. Missing Exit Conditions
**Wrong:** Iterative loop without max_iterations.
**Right:** Always include `max_iterations` and check in router.

### 3. Not Using @persist() in Production
**Wrong:** Long-running flow without persistence.
**Right:** Add `@persist()` to any flow that could fail mid-execution.

### 4. Synchronous When Async Available
**Wrong:** `crew.kickoff()` for multiple independent crews.
**Right:** `await asyncio.gather(*[crew.akickoff() for crew in crews])`.

### 5. Forgetting Guardrails for Critical Outputs
**Wrong:** Financial report task without validation.
**Right:** Add `guardrail` for compliance and accuracy checks.

### 6. Skipping Post-Processing
**Wrong:** Sending raw LLM output to customers.
**Right:** Add citation, formatting, and compliance agents.
