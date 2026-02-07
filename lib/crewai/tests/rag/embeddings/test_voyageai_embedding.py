"""Tests for VoyageAI embedding function."""

import os

import pytest

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    VoyageAIEmbeddingFunction,
)

voyageai = pytest.importorskip("voyageai", reason="voyageai not installed")

# Standard models that support basic embedding and custom dimensions
STANDARD_MODELS = [
    "voyage-3.5",
    "voyage-4",
    "voyage-4-lite",
    "voyage-4-large",
]


@pytest.mark.parametrize("model", STANDARD_MODELS)
def test_basic_embedding(model: str) -> None:
    """Test basic embedding generation for standard models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


@pytest.mark.parametrize("model", STANDARD_MODELS)
def test_with_embedding_dimensions(model: str) -> None:
    """Test embedding generation with custom dimensions for standard models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
        output_dimension=2048,
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 2048


def test_with_contextual_embedding() -> None:
    """Test contextual embedding generation."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-context-3",
        output_dimension=2048,
    )
    embeddings = ef(["hello world", "in chroma"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 2048


def test_count_tokens() -> None:
    """Test token counting functionality."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    texts = ["hello world", "this is a longer text with more tokens"]
    token_counts = ef.count_tokens(texts)
    assert len(token_counts) == 2
    assert token_counts[0] > 0
    assert token_counts[1] > token_counts[0]  # Longer text should have more tokens


def test_count_tokens_empty_list() -> None:
    """Test token counting with empty list."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    token_counts = ef.count_tokens([])
    assert token_counts == []


def test_count_tokens_single_text() -> None:
    """Test token counting with single text."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-2",
    )
    token_counts = ef.count_tokens(["hello"])
    assert len(token_counts) == 1
    assert token_counts[0] > 0


# Token limit test cases: (model, expected_limit)
TOKEN_LIMIT_CASES = [
    ("voyage-2", 320_000),
    ("voyage-context-3", 32_000),
    ("voyage-3-large", 120_000),
    # Voyage-4 family
    ("voyage-4", 320_000),
    ("voyage-4-lite", 1_000_000),
    ("voyage-4-large", 120_000),
]


@pytest.mark.parametrize("model,expected_limit", TOKEN_LIMIT_CASES)
def test_get_token_limit(model: str, expected_limit: int) -> None:
    """Test getting token limit for different models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")

    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    assert ef.get_token_limit() == expected_limit


@pytest.mark.parametrize("model", STANDARD_MODELS)
def test_batching_with_multiple_texts(model: str) -> None:
    """Test that batching works with multiple texts for standard models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    texts = ["text1", "text2", "text3", "text4", "text5"]
    embeddings = ef(texts)
    assert len(embeddings) == 5
    assert all(len(emb) > 0 for emb in embeddings)


def test_build_batches() -> None:
    """Test the _build_batches method."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-2",
    )
    texts = ["short", "text", "here", "now"]
    batches = list(ef._build_batches(texts))
    # Should create at least one batch
    assert len(batches) >= 1
    # Total texts should be preserved
    total_texts = sum(len(batch) for batch in batches)
    assert total_texts == len(texts)


def test_batching_with_large_texts() -> None:
    """Test batching with texts that may exceed token limits."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    # Create long texts
    long_text = "This is a long text with many words. " * 100
    texts = [long_text, long_text, long_text]
    embeddings = ef(texts)
    assert len(embeddings) == 3
    assert all(len(emb) > 0 for emb in embeddings)


def test_contextual_batching() -> None:
    """Test that contextual models support batching."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-context-3",
    )
    texts = ["text1", "text2", "text3", "text4"]
    embeddings = ef(texts)
    assert len(embeddings) == 4
    assert all(len(emb) > 0 for emb in embeddings)


def test_contextual_build_batches() -> None:
    """Test that contextual models use _build_batches correctly."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-context-3",
    )
    texts = ["short", "text", "here", "now", "more"]
    batches = list(ef._build_batches(texts))
    # Should create at least one batch
    assert len(batches) >= 1
    # Total texts should be preserved
    total_texts = sum(len(batch) for batch in batches)
    assert total_texts == len(texts)


def test_contextual_with_large_batch() -> None:
    """Test contextual model with large batch that should be split."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-context-3",
    )
    # Create many texts
    texts = [f"Document number {i} with some content" for i in range(15)]
    embeddings = ef(texts)
    assert len(embeddings) == 15
    assert all(len(emb) > 0 for emb in embeddings)


def test_empty_input() -> None:
    """Test with empty input."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    # ChromaDB's EmbeddingFunction validates that embeddings are non-empty
    with pytest.raises(ValueError, match="Expected Embeddings to be non-empty"):
        ef([])


def test_single_string_input() -> None:
    """Test with single string input (not in a list)."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    embeddings = ef("hello world")
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


def test_is_context_model() -> None:
    """Test the _is_context_model helper method."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")

    # Test with context model
    ef_context = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-context-3",
    )
    assert ef_context._is_context_model() is True

    # Test with regular model
    ef_regular = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )
    assert ef_regular._is_context_model() is False


def test_name() -> None:
    """Test the static name method."""
    assert VoyageAIEmbeddingFunction.name() == "voyageai"


def test_voyage_multimodal_3() -> None:
    """Test voyage-multimodal-3 embedding generation."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-multimodal-3",
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


def test_voyage_multimodal_3_5() -> None:
    """Test voyage-multimodal-3.5 embedding generation."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-multimodal-3.5",
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


@pytest.mark.skip(reason="output_dimension for multimodal requires voyageai SDK >=0.3.6")
def test_voyage_multimodal_3_5_with_dimensions() -> None:
    """Test voyage-multimodal-3.5 embedding with custom dimensions.

    Note: This test requires voyageai SDK >=0.3.6 which adds ffmpeg-python
    as a required dependency. Skipped to avoid unnecessary dependencies.
    """
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-multimodal-3.5",
        output_dimension=2048,
    )
    embeddings = ef(["hello world"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 2048


def test_voyage_multimodal_3_5_token_limit() -> None:
    """Test voyage-multimodal-3.5 token limit is correctly set."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-multimodal-3.5",
    )
    assert ef.get_token_limit() == 32_000


def test_voyage_multimodal_batching() -> None:
    """Test multimodal model batching with multiple texts."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-multimodal-3.5",
    )
    texts = ["text1", "text2", "text3", "text4", "text5"]
    embeddings = ef(texts)
    assert len(embeddings) == 5
    assert all(len(emb) > 0 for emb in embeddings)


@pytest.mark.parametrize("model", STANDARD_MODELS)
def test_is_not_context_model(model: str) -> None:
    """Test that standard models are not detected as context models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    assert ef._is_context_model() is False


@pytest.mark.parametrize("model", STANDARD_MODELS)
def test_is_not_multimodal_model(model: str) -> None:
    """Test that standard models are not detected as multimodal models."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    assert ef._is_multimodal_model() is False


# Multimodal models list
MULTIMODAL_MODELS = [
    "voyage-multimodal-3",
    "voyage-multimodal-3.5",
]


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_is_multimodal_model(model: str) -> None:
    """Test that multimodal models are correctly detected."""
    if os.environ.get("VOYAGE_API_KEY") is None:
        pytest.skip("VOYAGE_API_KEY not set")
    ef = VoyageAIEmbeddingFunction(
        api_key=os.environ["VOYAGE_API_KEY"],
        model=model,
    )
    assert ef._is_multimodal_model() is True
