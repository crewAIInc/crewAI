from enum import Enum
import os
from typing import Annotated, Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    StringConstraints,
)
import requests

from crewai_tools.tools.companyenrich_tool._core import core_arguments
from crewai_tools.tools.companyenrich_tool._validation import prepare_request


class CompanyEnrichOperation(str, Enum):
    """Fixed approved routes; email operations are beta-gated."""

    get_companies_enrich = "get_companies_enrich"
    post_companies_enrich = "post_companies_enrich"
    post_companies_enrich_batch = "post_companies_enrich_batch"
    get_companies_workforce = "get_companies_workforce"
    get_companies_autocomplete = "get_companies_autocomplete"
    post_companies_enrich_bulk = "post_companies_enrich_bulk"
    get_companies_enrich_bulk_jobid = "get_companies_enrich_bulk_jobid"
    get_companies_enrich_bulk_jobs = "get_companies_enrich_bulk_jobs"
    post_companies_search = "post_companies_search"
    post_companies_search_preview = "post_companies_search_preview"
    post_companies_search_count = "post_companies_search_count"
    post_companies_search_scroll = "post_companies_search_scroll"
    get_companies = "get_companies"
    post_companies_search_async = "post_companies_search_async"
    get_companies_search_async_jobid = "get_companies_search_async_jobid"
    get_companies_search_async_jobs = "get_companies_search_async_jobs"
    post_companies_similar = "post_companies_similar"
    post_companies_similar_preview = "post_companies_similar_preview"
    post_companies_similar_count = "post_companies_similar_count"
    post_companies_similar_scroll = "post_companies_similar_scroll"
    post_people_lookup = "post_people_lookup"
    get_people = "get_people"
    post_people_search = "post_people_search"
    post_people_search_scroll = "post_people_search_scroll"
    get_people_email = "get_people_email"
    post_people_email_bulk = "post_people_email_bulk"
    get_people_email_bulk_jobid = "get_people_email_bulk_jobid"
    get_people_education = "get_people_education"
    post_people_search_async = "post_people_search_async"
    get_people_search_async_jobid = "get_people_search_async_jobid"
    get_people_search_async_jobs = "get_people_search_async_jobs"
    get_jobs = "get_jobs"
    get_jobs_jobid = "get_jobs_jobid"
    get_geo_regions = "get_geo_regions"
    get_geo_countries_countrycode = "get_geo_countries_countrycode"
    post_geo_countries = "post_geo_countries"
    post_geo_states = "post_geo_states"
    post_geo_cities = "post_geo_cities"
    get_industries = "get_industries"
    get_keywords_autocomplete = "get_keywords_autocomplete"
    get_technologies_autocomplete = "get_technologies_autocomplete"
    get_positions_autocomplete = "get_positions_autocomplete"


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ExpansionInput(_Input):
    expand: list[str] | None = Field(
        default=None,
        description="Optional expansion: workforce for companies; education for people.",
    )


class CompanyEnrichCompanyInput(_ExpansionInput):
    domain: NonEmptyString = Field(min_length=1, max_length=1024)
    wait_for_enrichment: bool | None = None


class CompanyEnrichSearchInput(_ExpansionInput):
    semantic_query: NonEmptyString | None = Field(default=None, max_length=500)
    query: str | None = Field(
        default=None, max_length=250, description="Literal company name/domain query."
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Documented API filter fields. Dedicated inputs take precedence.",
    )
    page: StrictInt = Field(default=1, ge=1)
    page_size: StrictInt = Field(default=5, ge=1, le=100)


class CompanyEnrichSimilarInput(_ExpansionInput):
    domains: list[NonEmptyString] = Field(min_length=1, max_length=10)
    filters: dict[str, Any] | None = None
    page: StrictInt = Field(default=1, ge=1)
    page_size: StrictInt = Field(default=5, ge=1, le=100)
    wait_for_enrichment: bool | None = None


class CompanyEnrichPeopleSearchInput(_ExpansionInput):
    filters: dict[str, Any] = Field(
        description="People Search API body filters, including nested companyFilter."
    )
    page: StrictInt = Field(default=1, ge=1)
    page_size: StrictInt = Field(default=5, ge=1, le=100)


class CompanyEnrichLookupInput(_ExpansionInput):
    email: str


class CompanyEnrichWorkforceInput(_Input):
    company_id: str | None = None
    domain: str | None = None


class CompanyEnrichPropertiesInput(_ExpansionInput):
    properties: dict[str, Any] = Field(
        description="At least one of name, linkedinUrl, linkedinId, twitterUrl, facebookUrl, instagramUrl, youTubeUrl."
    )


class CompanyEnrichBatchInput(_ExpansionInput):
    domains: list[NonEmptyString] = Field(min_length=1, max_length=50)


class CompanyEnrichCompanyIdInput(_ExpansionInput):
    company_id: str


class CompanyEnrichPersonIdInput(_ExpansionInput):
    person_id: str | StrictInt = Field(
        description="Positive int64 id; decimal strings preserve precision."
    )


class CompanyEnrichSearchCountInput(_Input):
    semantic_query: NonEmptyString | None = Field(default=None, max_length=500)
    query: str | None = Field(default=None, max_length=250)
    filters: dict[str, Any] | None = None


class CompanyEnrichSimilarCountInput(_Input):
    domains: list[NonEmptyString] = Field(min_length=1, max_length=10)
    filters: dict[str, Any] | None = None
    wait_for_enrichment: bool | None = None


class CompanyEnrichAdvancedInput(_Input):
    operation: CompanyEnrichOperation = Field(
        description="One approved operation; email routes require allow_beta.",
        strict=False,
    )
    body: dict[str, Any] | None = None
    query: dict[str, Any] | None = None
    path_parameters: dict[str, Any] | None = None
    allow_beta: bool = False


class _CompanyEnrichTool(BaseTool):
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    operation: ClassVar[str]
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="COMPANYENRICH_API_KEY",
                description="CompanyEnrich API key",
                required=True,
            )
        ]
    )

    def _request(self, operation: str, **kwargs: Any) -> Any:
        method, path, query, body = prepare_request(operation, **kwargs)
        key = (
            self.api_key.get_secret_value()
            if self.api_key
            else os.getenv("COMPANYENRICH_API_KEY")
        )
        if not key or not key.strip():
            raise ValueError("Set COMPANYENRICH_API_KEY or pass api_key to the tool")
        try:
            response = requests.request(
                method,
                "https://api.companyenrich.com" + path,
                headers={
                    "Authorization": "Bearer " + key,
                    "Accept": "application/json",
                },
                params=query or None,
                json=body,
                timeout=self.timeout,
                allow_redirects=False,
            )
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ValueError("Unexpected redirect from CompanyEnrich")
            return response.json()
        except requests.RequestException as exc:
            raise type(exc)(
                "CompanyEnrich request failed; check connectivity, credentials and credits"
            ) from None

    def _invoke(self, values: dict[str, Any]) -> Any:
        values = self.args_schema.model_validate(values).model_dump()
        body, query = core_arguments(self.operation, values)
        result = self._request(self.operation, body=body, query=query)
        if self.operation in {
            "get_companies_enrich",
            "post_companies_search",
            "post_companies_similar",
        } and not isinstance(result, dict):
            raise TypeError("Expected a JSON object from CompanyEnrich")
        return result

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise ValueError("Use named inputs for this CompanyEnrich tool")
        return self._invoke(kwargs)


class CompanyEnrichCompanyTool(_CompanyEnrichTool):
    """CompanyEnrich company enrichment."""

    name: str = "CompanyEnrich company enrichment"
    description: str = "Company enrichment. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "get_companies_enrich"
    args_schema: type[BaseModel] = CompanyEnrichCompanyInput

    def _run(self, domain: str, **kwargs: Any) -> Any:
        return self._invoke({"domain": domain, **kwargs})


class CompanyEnrichSearchTool(_CompanyEnrichTool):
    """CompanyEnrich company search."""

    name: str = "CompanyEnrich company search"
    description: str = "Company search. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_search"
    args_schema: type[BaseModel] = CompanyEnrichSearchInput

    def _run(
        self,
        semantic_query: NonEmptyString | None = None,
        page: int = 1,
        page_size: int = 5,
        **kwargs: Any,
    ) -> Any:
        return self._invoke(
            {
                "semantic_query": semantic_query,
                "page": page,
                "page_size": page_size,
                **kwargs,
            }
        )


class CompanyEnrichSimilarTool(_CompanyEnrichTool):
    """CompanyEnrich similar companies."""

    name: str = "CompanyEnrich similar companies"
    description: str = "Similar companies. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_similar"
    args_schema: type[BaseModel] = CompanyEnrichSimilarInput

    def _run(
        self, domains: list[str], page: int = 1, page_size: int = 5, **kwargs: Any
    ) -> Any:
        return self._invoke(
            {"domains": domains, "page": page, "page_size": page_size, **kwargs}
        )


class CompanyEnrichPeopleSearchTool(_CompanyEnrichTool):
    """CompanyEnrich people search."""

    name: str = "CompanyEnrich people search"
    description: str = "People search. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_people_search"
    args_schema: type[BaseModel] = CompanyEnrichPeopleSearchInput


class CompanyEnrichPersonLookupTool(_CompanyEnrichTool):
    """CompanyEnrich person lookup by email."""

    name: str = "CompanyEnrich person lookup by email"
    description: str = "Person lookup by email. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_people_lookup"
    args_schema: type[BaseModel] = CompanyEnrichLookupInput


class CompanyEnrichWorkforceTool(_CompanyEnrichTool):
    """CompanyEnrich company workforce."""

    name: str = "CompanyEnrich company workforce"
    description: str = "Company workforce. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "get_companies_workforce"
    args_schema: type[BaseModel] = CompanyEnrichWorkforceInput


class CompanyEnrichPropertiesTool(_CompanyEnrichTool):
    """CompanyEnrich company match by properties."""

    name: str = "CompanyEnrich company match by properties"
    description: str = "Company match by properties. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_enrich"
    args_schema: type[BaseModel] = CompanyEnrichPropertiesInput


class CompanyEnrichBatchTool(_CompanyEnrichTool):
    """CompanyEnrich batch company enrichment."""

    name: str = "CompanyEnrich batch company enrichment"
    description: str = "Batch company enrichment. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_enrich_batch"
    args_schema: type[BaseModel] = CompanyEnrichBatchInput


class CompanyEnrichCompanyByIdTool(_CompanyEnrichTool):
    """CompanyEnrich company by ID."""

    name: str = "CompanyEnrich company by ID"
    description: str = "Company by id. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "get_companies"
    args_schema: type[BaseModel] = CompanyEnrichCompanyIdInput


class CompanyEnrichPersonByIdTool(_CompanyEnrichTool):
    """CompanyEnrich person by ID."""

    name: str = "CompanyEnrich person by ID"
    description: str = "Person by id. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "get_people"
    args_schema: type[BaseModel] = CompanyEnrichPersonIdInput


class CompanyEnrichSearchCountTool(_CompanyEnrichTool):
    """CompanyEnrich company search count."""

    name: str = "CompanyEnrich company search count"
    description: str = "Company search count. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_search_count"
    args_schema: type[BaseModel] = CompanyEnrichSearchCountInput


class CompanyEnrichSimilarCountTool(_CompanyEnrichTool):
    """CompanyEnrich similar companies count."""

    name: str = "CompanyEnrich similar companies count"
    description: str = "Similar companies count. One API request; may consume credits. Preserves the complete response."
    operation: ClassVar[str] = "post_companies_similar_count"
    args_schema: type[BaseModel] = CompanyEnrichSimilarCountInput


class CompanyEnrichAdvancedRequestTool(_CompanyEnrichTool):
    """One fixed-catalog API operation, including explicit job and beta controls."""

    name: str = "CompanyEnrich advanced request"
    description: str = "Call one of 42 approved data, lookup or job operations. Email operations are beta and require allow_beta. No retries, polling or result downloads. Excludes account secrets and saved-list administration."
    args_schema: type[BaseModel] = CompanyEnrichAdvancedInput

    def _run(
        self,
        operation: str,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        path_parameters: dict[str, Any] | None = None,
        allow_beta: bool = False,
    ) -> Any:
        return self._request(
            operation,
            body=body,
            query=query,
            path_parameters=path_parameters,
            allow_beta=allow_beta,
        )
