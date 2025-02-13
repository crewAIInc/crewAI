from enum import Enum


class TraceType(Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    FLOW_STEP = "flow_step"
    START_CALL = "start_call"


class RunType(Enum):
    KICKOFF = "kickoff"
    TRAIN = "train"
    TEST = "test"


class CrewType(Enum):
    CREW = "crew"
    FLOW = "flow"
