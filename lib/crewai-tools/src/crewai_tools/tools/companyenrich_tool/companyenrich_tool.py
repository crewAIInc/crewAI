import os
from typing import Annotated, Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, SecretStr, StringConstraints
import requests


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CompanyEnrichCompanyInput(BaseModel):
    """Look up a company by its domain."""

    domain: NonEmptyString = Field(
        max_length=1024, description="Company domain, for example stripe.com"
    )


class CompanyEnrichSearchInput(BaseModel):
    """Search for companies using a natural-language description."""

    semantic_query: NonEmptyString = Field(
        max_length=500,
        description="Description of the companies to find, up to 500 characters",
    )
    page: int = Field(default=1, ge=1, description="Page number, starting at 1")
    page_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Results per page; each result consumes credits",
    )


class CompanyEnrichSimilarInput(BaseModel):
    """Find companies similar to seed domains."""

    domains: list[NonEmptyString] = Field(
        min_length=1, max_length=10, description="One to ten seed company domains"
    )
    page: int = Field(default=1, ge=1, description="Page number, starting at 1")
    page_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Results per page; each result consumes credits",
    )


class _CompanyEnrichTool(BaseTool):
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="COMPANYENRICH_API_KEY",
                description="CompanyEnrich API key",
                required=True,
            )
        ]
    )

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = (
            self.api_key.get_secret_value()
            if self.api_key
            else os.getenv("COMPANYENRICH_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise ValueError("Set COMPANYENRICH_API_KEY or pass api_key to the tool")
        response = requests.request(
            method,
            f"https://api.companyenrich.com{endpoint}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            params=params,
            json=payload,
            timeout=self.timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError("Unexpected redirect from CompanyEnrich")
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError("Expected a JSON object from CompanyEnrich")
        return result


class CompanyEnrichCompanyTool(_CompanyEnrichTool):
    """Enrich a company domain using the CompanyEnrich API."""

    name: str = "CompanyEnrich company enrichment"
    description: str = (
        "Get a company profile by domain. Costs 1 CompanyEnrich credit per call."
    )
    args_schema: type[BaseModel] = CompanyEnrichCompanyInput

    def _run(self, domain: str) -> dict[str, Any]:
        inputs = CompanyEnrichCompanyInput(domain=domain)
        return self._request(
            "GET", "/companies/enrich", params={"domain": inputs.domain}
        )


class CompanyEnrichSearchTool(_CompanyEnrichTool):
    """Search companies with a natural-language description."""

    name: str = "CompanyEnrich company search"
    description: str = "Find companies matching a natural-language description. Costs 1 credit per result, with a 1-credit minimum. Returns one page only."
    args_schema: type[BaseModel] = CompanyEnrichSearchInput

    def _run(
        self, semantic_query: str, page: int = 1, page_size: int = 5
    ) -> dict[str, Any]:
        inputs = CompanyEnrichSearchInput(
            semantic_query=semantic_query, page=page, page_size=page_size
        )
        if inputs.page * inputs.page_size > 10000:
            raise ValueError("Company search requires page * page_size <= 10000")
        return self._request(
            "POST",
            "/companies/search",
            payload={
                "semanticQuery": inputs.semantic_query,
                "page": inputs.page,
                "pageSize": inputs.page_size,
            },
        )


class CompanyEnrichSimilarTool(_CompanyEnrichTool):
    """Discover companies similar to one or more seed companies."""

    name: str = "CompanyEnrich similar companies"
    description: str = "Find companies similar to one to ten seed domains. Costs 5 credits per result, with a 5-credit minimum. Returns one page only."
    args_schema: type[BaseModel] = CompanyEnrichSimilarInput

    def _run(
        self, domains: list[str], page: int = 1, page_size: int = 5
    ) -> dict[str, Any]:
        inputs = CompanyEnrichSimilarInput(
            domains=domains, page=page, page_size=page_size
        )
        return self._request(
            "POST",
            "/companies/similar",
            payload={
                "domains": inputs.domains,
                "page": inputs.page,
                "pageSize": inputs.page_size,
            },
        )
