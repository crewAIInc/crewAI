"""Type definitions for IBM Watson embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class WatsonProviderConfig(TypedDict, total=False):
    """Configuration for Watson provider."""

    model_id: str
    url: str
    params: dict[str, str | dict[str, str]]
    credentials: Any
    project_id: str
    space_id: str
    api_client: Any
    verify: bool | str
    persistent_connection: Annotated[bool, True]
    batch_size: Annotated[int, 100]
    concurrency_limit: Annotated[int, 10]
    max_retries: int
    delay_time: float
    retry_status_codes: list[int]
    api_key: str
    name: str
    iam_serviceid_crn: str
    trusted_profile_id: str
    token: str
    projects_token: str
    username: str
    password: str
    instance_id: str
    version: str
    bedrock_url: str
    platform_url: str
    proxies: dict


class WatsonProviderSpec(TypedDict, total=False):
    """Watson provider specification."""

    provider: Required[Literal["watson"]]
    config: WatsonProviderConfig
