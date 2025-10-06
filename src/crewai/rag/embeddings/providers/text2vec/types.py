"""Type definitions for Text2Vec embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class Text2VecProviderConfig(TypedDict, total=False):
    """Configuration for Text2Vec provider."""

    model_name: Annotated[str, "shibing624/text2vec-base-chinese"]


class Text2VecProviderSpec(TypedDict):
    """Text2Vec provider specification."""

    provider: Required[Literal["text2vec"]]
    config: Text2VecProviderConfig
