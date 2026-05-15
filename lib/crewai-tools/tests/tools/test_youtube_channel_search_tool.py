from crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool import YoutubeChannelSearchTool


class _DummyAdapter:
    """Minimal adapter stub that records add() calls."""

    def __init__(self):
        self.calls = []

    def query(self, *args, **kwargs):
        return ""

    def add(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_add_converts_handle_to_full_url():
    """YoutubeChannelSearchTool.add must convert a bare handle to a full URL.

    Regression test for https://github.com/crewAIInc/crewAI/issues/5429
    where passing a handle like '@krishnaik06' resulted in an invalid URL
    because the loader received the bare handle instead of a proper URL.
    """
    tool = YoutubeChannelSearchTool()
    dummy = _DummyAdapter()
    tool.adapter = dummy

    tool.add("krishnaik06")

    assert len(dummy.calls) == 1
    pos_args, _kw = dummy.calls[0]
    assert pos_args[0] == "https://www.youtube.com/@krishnaik06"


def test_add_keeps_existing_at_prefix():
    tool = YoutubeChannelSearchTool()
    dummy = _DummyAdapter()
    tool.adapter = dummy

    tool.add("@test")

    assert len(dummy.calls) == 1
    pos_args, _kw = dummy.calls[0]
    assert pos_args[0] == "https://www.youtube.com/@test"
