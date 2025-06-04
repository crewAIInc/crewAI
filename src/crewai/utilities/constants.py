TRAINING_DATA_FILE = "training_data.pkl"
TRAINED_AGENTS_DATA_FILE = "trained_agents_data.pkl"
DEFAULT_SCORE_THRESHOLD = 0.35
KNOWLEDGE_DIRECTORY = "knowledge"
MAX_LLM_RETRY = 3
MAX_FILE_NAME_LENGTH = 255
EMITTER_COLOR = "bold_blue"

# Tool usage and validation constants
DEFAULT_TOOL_RESULTS_LIMIT = 3
MAX_REASONING_ATTEMPTS_DEFAULT = 5
DEFAULT_CONTEXT_SCORE_THRESHOLD = 0.35

# File and collection name constants  
MIN_COLLECTION_NAME_LENGTH = 3
MAX_COLLECTION_NAME_LENGTH = 63


class _NotSpecified:
    def __repr__(self):
        return "NOT_SPECIFIED"


# Sentinel value used to detect when no value has been explicitly provided.
# Unlike `None`, which might be a valid value from the user, `NOT_SPECIFIED` allows
# us to distinguish between "not passed at all" and "explicitly passed None" or "[]".
NOT_SPECIFIED = _NotSpecified()
