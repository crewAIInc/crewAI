from crewai.llms.providers.azure.tool_arguments import parse_tool_arguments


def test_parse_tool_arguments_accepts_dict_null_and_json_string():
    assert parse_tool_arguments({"city": "London"}) == {"city": "London"}
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments('{"city": "London"}') == {"city": "London"}
    assert parse_tool_arguments("not-json") == {}
    assert parse_tool_arguments(0) == {}
    assert parse_tool_arguments(["x"]) == {}
