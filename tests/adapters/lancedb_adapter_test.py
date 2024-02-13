from crewai_tools.adapters.lancedb_adapter import LanceDBAdapter


def test_lancedb_adapter(helpers):
    adapter = LanceDBAdapter(
        uri="tests/data/lancedb",
        table_name="requirements",
        embedding_function=helpers.get_embedding_function(),
        top_k=2,
        vector_column_name="vector",
        text_column_name="text",
    )

    assert (
        adapter.query("What are the requirements for the task?")
        == """Technical requirements

The system should be able to process 1000 transactions per second. The code must be written in Ruby.
Problem

Currently, we are not able to find out palindromes in a given string. We need a solution to this problem."""
    )
