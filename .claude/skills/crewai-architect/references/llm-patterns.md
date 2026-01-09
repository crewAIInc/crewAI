# LLM Integration Patterns

## Table of Contents
- [Built-in LLM Configuration](#built-in-llm-configuration)
- [Provider-Specific Setup](#provider-specific-setup)
- [Custom LLM Implementation](#custom-llm-implementation)
- [Direct LLM Calls in Flows](#direct-llm-calls-in-flows)
- [Structured Responses](#structured-responses)

## Built-in LLM Configuration

### Basic LLM Class

```python
from crewai import LLM

# OpenAI
llm = LLM(model="gpt-4o", temperature=0.7)

# Anthropic
llm = LLM(model="anthropic/claude-sonnet-4-20250514")

# Google
llm = LLM(model="gemini/gemini-2.0-flash")

# Local (Ollama)
llm = LLM(model="ollama/llama3.2")
```

### LLM Parameters

```python
llm = LLM(
    model="gpt-4o",
    temperature=0.7,          # Creativity (0-1)
    max_tokens=4096,          # Max response length
    top_p=0.9,                # Nucleus sampling
    frequency_penalty=0.0,    # Reduce repetition
    presence_penalty=0.0,     # Encourage new topics
    seed=42,                  # Reproducibility
    response_format=MyModel,  # Pydantic model for structured output
)
```

## Provider-Specific Setup

### OpenAI

```bash
# .env
OPENAI_API_KEY=sk-...
```

```python
llm = LLM(model="gpt-4o")
# or
llm = LLM(model="gpt-4o-mini")  # Faster, cheaper
```

### Anthropic

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

```python
llm = LLM(model="anthropic/claude-sonnet-4-20250514")
```

### Google Gemini

```bash
# .env
GOOGLE_API_KEY=AIza...
```

```python
# Via LiteLLM
llm = LLM(model="gemini/gemini-2.0-flash")

# Via OpenAI-compatible endpoint
llm = LLM(
    model="openai/gemini-2.0-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="your-gemini-key"
)
```

### Azure OpenAI

```bash
# .env
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-01
```

```python
llm = LLM(
    model="azure/your-deployment-name",
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_API_BASE"),
)
```

### Local Models (Ollama)

```bash
# Start Ollama
ollama serve
ollama pull llama3.2
```

```python
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)
```

## Direct LLM Calls in Flows

### Simple Call

```python
class MyFlow(Flow[State]):
    def __init__(self):
        super().__init__()
        self.llm = LLM(model="gpt-4o")

    @start()
    def process(self):
        response = self.llm.call(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Process: {self.state.input}"}
        ])
        return response
```

### With Message History

```python
@listen(previous_step)
def conversation(self, context):
    messages = [
        {"role": "system", "content": "Expert analyst."},
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": context},  # Previous response
        {"role": "user", "content": "Follow-up question"}
    ]
    return self.llm.call(messages=messages)
```

## Structured Responses

### With LLM Class

```python
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    summary: str = Field(description="Brief summary")
    key_points: list[str] = Field(description="Main findings")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    recommendation: str

@start()
def analyze(self):
    llm = LLM(model="gpt-4o", response_format=Analysis)
    result = llm.call(messages=[
        {"role": "user", "content": f"Analyze: {self.state.data}"}
    ])
    self.state.analysis = result  # LLM.call() returns model directly
    return result
```

### Complex Nested Models

```python
from pydantic import BaseModel
from typing import Optional

class Finding(BaseModel):
    title: str
    description: str
    severity: str  # "low", "medium", "high"

class RiskAssessment(BaseModel):
    overall_risk: str
    findings: list[Finding]
    mitigations: list[str]

class FullReport(BaseModel):
    executive_summary: str
    risk_assessment: RiskAssessment
    recommendations: list[str]
    next_steps: Optional[list[str]] = None

llm = LLM(model="gpt-4o", response_format=FullReport)
result = llm.call(messages=[...])
report = result  # Returns fully typed FullReport directly
```

### Validation and Error Handling

```python
@start()
def safe_extraction(self):
    llm = LLM(model="gpt-4o", response_format=DataModel)
    try:
        result = llm.call(messages=[...])
        if result:  # LLM returns model directly when response_format is set
            self.state.data = result
            return "success"
        else:
            self.state.error = "No structured output"
            return "failed"
    except Exception as e:
        self.state.error = str(e)
        return "failed"
```

## Model Selection Guide

| Use Case | Recommended Model | Notes |
|----------|------------------|-------|
| Complex reasoning | gpt-4o, claude-sonnet-4-20250514 | Best quality |
| Fast responses | gpt-4o-mini, gemini-2.0-flash | Good balance |
| Cost-sensitive | gpt-4o-mini, ollama/llama3.2 | Lowest cost |
| Long context | gpt-4o (128k), claude (200k) | Large documents |
| Structured output | gpt-4o, gpt-4o-mini | Best JSON mode |
| Privacy-sensitive | ollama/*, local models | No data leaves |

## Caching and Optimization

### LLM Instance Reuse

```python
class OptimizedFlow(Flow[State]):
    def __init__(self):
        super().__init__()
        # Initialize once, reuse across methods
        self.fast_llm = LLM(model="gpt-4o-mini")
        self.smart_llm = LLM(model="gpt-4o")

    @start()
    def quick_classify(self):
        # Use fast model for simple tasks
        return self.fast_llm.call([...])

    @listen(quick_classify)
    def deep_analysis(self, classification):
        # Use smart model for complex tasks
        return self.smart_llm.call([...])
```

### Temperature by Task

```python
# Classification: low temperature for consistency
classifier = LLM(model="gpt-4o", temperature=0.1)

# Creative: higher temperature for variety
creative = LLM(model="gpt-4o", temperature=0.9)

# Structured extraction: zero temperature
extractor = LLM(model="gpt-4o", temperature=0, response_format=DataModel)
```
