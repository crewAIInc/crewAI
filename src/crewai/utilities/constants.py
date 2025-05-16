TRAINING_DATA_FILE = "training_data.pkl"
TRAINED_AGENTS_DATA_FILE = "trained_agents_data.pkl"
DEFAULT_SCORE_THRESHOLD = 0.35
KNOWLEDGE_DIRECTORY = "knowledge"
MAX_LLM_RETRY = 3
MAX_FILE_NAME_LENGTH = 255
EMITTER_COLOR = "bold_blue"


class _NotSpecified:
    def __repr__(self):
        return "NOT_SPECIFIED"


# Sentinel value used to detect when no value has been explicitly provided.
# Unlike `None`, which might be a valid value from the user, `NOT_SPECIFIED` allows
# us to distinguish between "not passed at all" and "explicitly passed None" or "[]".
NOT_SPECIFIED = _NotSpecified()
