# First, import the core base classes without AgentEvaluator
from crewai.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationScore,
    MetricCategory,
    AgentEvaluationResult
)

# Now import the evaluators which depend on base classes
from crewai.evaluation.evaluators import (
    GoalAlignmentEvaluator,
    KnowledgeRetrievalEvaluator,
    SemanticQualityEvaluator
)

from crewai.evaluation.evaluators_tools import (
    ToolUsageEvaluator,
    StepEfficiencyEvaluator
)

from crewai.evaluation.meta_evaluator import MetaEvaluator

# Next import integration which uses the base classes but not AgentEvaluator
from crewai.evaluation.integration import (
    EvaluationTraceCallback,
    create_evaluation_callbacks,
    evaluate_agent,
    evaluate_crew,
    create_default_evaluator,
    evaluate_execution_from_callbacks
)

# Finally import AgentEvaluator after everything else is loaded
from crewai.evaluation.base_evaluator import AgentEvaluator
