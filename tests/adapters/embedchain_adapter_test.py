from typing import Callable

from chromadb import Documents, EmbeddingFunction, Embeddings
from embedchain import App
from embedchain.config import AppConfig, ChromaDbConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.vectordb.chroma import ChromaDB

from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter


class MockEmbeddingFunction(EmbeddingFunction):
    fn: Callable

    def __init__(self, embedding_fn: Callable):
        self.fn = embedding_fn

    def __call__(self, input: Documents) -> Embeddings:
        return self.fn(input)


def test_embedchain_adapter(helpers):
    embedding_function = MockEmbeddingFunction(
        embedding_fn=helpers.get_embedding_function()
    )
    embedder = BaseEmbedder()
    embedder.set_embedding_fn(embedding_function)  # type: ignore

    db = ChromaDB(
        config=ChromaDbConfig(
            dir="tests/data/chromadb",
            collection_name="requirements",
        )
    )

    app = App(
        config=AppConfig(
            id="test",
        ),
        db=db,
        embedding_model=embedder,
    )

    adapter = EmbedchainAdapter(
        dry_run=True,
        embedchain_app=app,
    )

    assert (
        adapter.query("What are the requirements for the task?")
        == """
  Use the following pieces of context to answer the query at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  Technical requirements

The system should be able to process 1000 transactions per second. The code must be written in Ruby. | Problem

Currently, we are not able to find out palindromes in a given string. We need a solution to this problem. | Solution

We need a function that takes a string as input and returns true if the string is a palindrome, otherwise false.

  Query: What are the requirements for the task?

  Helpful Answer:
"""
    )
