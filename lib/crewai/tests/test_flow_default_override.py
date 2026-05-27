"""Test that persisted state properly overrides default values."""

from crewai.flow.flow import Flow, FlowState, listen, start
from crewai.flow.persistence import persist


class PoemState(FlowState):
    """Test state model with default values that should be overridden."""
    sentence_count: int = 1000  # Default that should be overridden
    has_set_count: bool = False
    poem_type: str = ""


def test_default_value_override():
    """Test that persisted state values override class defaults."""

    @persist()
    class PoemFlow(Flow[PoemState]):
        initial_state = PoemState

        @start()
        def set_sentence_count(self):
            if self.state.has_set_count and self.state.sentence_count == 2:
                self.state.sentence_count = 3

            elif self.state.has_set_count and self.state.sentence_count == 1000:
                self.state.sentence_count = 1000

            elif self.state.has_set_count and self.state.sentence_count == 5:
                self.state.sentence_count = 5

            else:
                self.state.sentence_count = 2
                self.state.has_set_count = True

    flow1 = PoemFlow()
    flow1.kickoff()
    original_uuid = flow1.state.id
    assert flow1.state.sentence_count == 2

    flow2 = PoemFlow()
    flow2.kickoff(inputs={"id": original_uuid})
    assert flow2.state.sentence_count == 3

    # Fourth run - explicit override should work
    flow3 = PoemFlow()
    flow3.kickoff(inputs={
        "id": original_uuid,
        "has_set_count": True,
        "sentence_count": 5,
    })
    assert flow3.state.sentence_count == 5

    flow4 = PoemFlow()
    flow4.kickoff(inputs={"has_set_count": True})
    assert flow4.state.sentence_count == 1000


def test_multi_step_default_override():
    """Test default value override with multiple start methods."""

    @persist()
    class MultiStepPoemFlow(Flow[PoemState]):
        initial_state = PoemState

        @start()
        def set_sentence_count(self):
            print("Setting sentence count")
            if not self.state.has_set_count:
                self.state.sentence_count = 3
                self.state.has_set_count = True

        @listen(set_sentence_count)
        def set_poem_type(self):
            print("Setting poem type")
            if self.state.sentence_count == 3:
                self.state.poem_type = "haiku"
            elif self.state.sentence_count == 5:
                self.state.poem_type = "limerick"
            else:
                self.state.poem_type = "free_verse"

        @listen(set_poem_type)
        def finished(self):
            print("finished")

    flow1 = MultiStepPoemFlow()
    flow1.kickoff()
    original_uuid = flow1.state.id
    assert flow1.state.sentence_count == 3
    assert flow1.state.poem_type == "haiku"

    flow2 = MultiStepPoemFlow()
    flow2.kickoff(inputs={
        "id": original_uuid,
        "sentence_count": 5
    })
    assert flow2.state.sentence_count == 5
    assert flow2.state.poem_type == "limerick"

    flow3 = MultiStepPoemFlow()
    flow3.kickoff(inputs={
        "id": original_uuid
    })
    assert flow3.state.sentence_count == 5
    assert flow3.state.poem_type == "limerick"