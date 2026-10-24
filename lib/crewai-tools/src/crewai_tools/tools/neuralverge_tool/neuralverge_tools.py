"""NeuralVerge tools: company and person business intelligence for CrewAI agents."""

from __future__ import annotations

from typing import Any, ClassVar

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.neuralverge_tool.base import (
    NeuralVergeAsyncTaskTool,
    NeuralVergeBaseTool,
    drop_none,
)


_DOMAIN_DESCRIPTION = (
    "Amazon marketplace domain, e.g. 'amazon.com', 'amazon.de', 'amazon.co.uk'."
)


# ---------------------------------------------------------------- contact enrichment
class NeuralVergeEmailSchema(BaseModel):
    """Input schema for tools that take one email address."""

    email: str = Field(..., description="Email address, e.g. 'jane.doe@example.com'.")


class NeuralVergePersonByEmailTool(NeuralVergeBaseTool):
    """Reverse email lookup."""

    name: str = "NeuralVerge Find Person by Email"
    description: str = (
        "Find the person behind an email address: full name, phones, locations, "
        "company, position and social profiles including LinkedIn. Costs 10 points."
    )
    args_schema: type[BaseModel] = NeuralVergeEmailSchema
    endpoint: ClassVar[str] = "run-email-enrichment"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"email": kwargs["email"]}


class NeuralVergeEmailValidationTool(NeuralVergeBaseTool):
    """Email deliverability check."""

    name: str = "NeuralVerge Validate Email"
    description: str = (
        "Verify an email address: valid / risky / invalid, catch-all flag, mail "
        "provider and confidence. Costs 1 point."
    )
    args_schema: type[BaseModel] = NeuralVergeEmailSchema
    endpoint: ClassVar[str] = "run-email-validation"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"email": kwargs["email"]}


class NeuralVergeEmailFinderSchema(BaseModel):
    """Input schema for NeuralVergeEmailFinderTool."""

    first_name: str = Field(..., description="Person's first name.")
    last_name: str = Field(..., description="Person's last name.")
    domain: str = Field(..., description="Company domain, e.g. 'example.com'.")


class NeuralVergeEmailFinderTool(NeuralVergeBaseTool):
    """Work email finder."""

    name: str = "NeuralVerge Find Email by Name"
    description: str = (
        "Find and verify a professional work email from a person's first name, "
        "last name and company domain. Costs 10 points."
    )
    args_schema: type[BaseModel] = NeuralVergeEmailFinderSchema
    endpoint: ClassVar[str] = "run-email-finder"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "first_name": kwargs["first_name"],
            "last_name": kwargs["last_name"],
            "domain": kwargs["domain"],
        }


class NeuralVergePhoneSchema(BaseModel):
    """Input schema for phone lookup tools."""

    phone: str = Field(
        ...,
        description="Phone number with country code, e.g. '+15555550100'.",
    )


class NeuralVergePhoneLookupTool(NeuralVergeBaseTool):
    """Global reverse phone lookup."""

    name: str = "NeuralVerge Find Person by Phone"
    description: str = (
        "Reverse phone lookup (global): name, emails, locations, company and "
        "social profiles for a phone number. Costs 10 points."
    )
    args_schema: type[BaseModel] = NeuralVergePhoneSchema
    endpoint: ClassVar[str] = "run-phone-enrichment"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"phone": kwargs["phone"]}


class NeuralVergeUSPhoneLookupTool(NeuralVergeBaseTool):
    """US reverse phone lookup with carrier data."""

    name: str = "NeuralVerge Find Person by US Phone"
    description: str = (
        "US-only reverse phone lookup with carrier, line type and activity/validity "
        "signals. Costs 100 points; use the global phone tool unless carrier data is needed."
    )
    args_schema: type[BaseModel] = NeuralVergePhoneSchema
    endpoint: ClassVar[str] = "run-phone-enrichment-us"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"phone": kwargs["phone"]}


# ---------------------------------------------------------------------------- company
class NeuralVergeCompanyFundingSchema(BaseModel):
    """Input schema for NeuralVergeCompanyFundingTool."""

    crunchbase_url: str = Field(
        ...,
        description="Crunchbase organization URL, e.g. 'https://www.crunchbase.com/organization/example'.",
    )


class NeuralVergeCompanyFundingTool(NeuralVergeBaseTool):
    """Company funding and firmographics."""

    name: str = "NeuralVerge Company Funding"
    description: str = (
        "Company profile from a Crunchbase organization URL: website, HQ, founding "
        "year, headcount, industries, funding rounds and investors. Costs 15 points."
    )
    args_schema: type[BaseModel] = NeuralVergeCompanyFundingSchema
    endpoint: ClassVar[str] = "run-crunchbase-company"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"url": kwargs["crunchbase_url"]}


# ------------------------------------------------------------------ search / extract
class NeuralVergeWebSearchSchema(BaseModel):
    """Input schema for NeuralVergeWebSearchTool."""

    query: str = Field(..., description="Search query.")
    max_results: int = Field(default=10, description="Number of results.")
    country: str = Field(default="us", description="Two-letter country code.")
    language: str = Field(default="en", description="Two-letter language code.")


class NeuralVergeWebSearchTool(NeuralVergeBaseTool):
    """Web search."""

    name: str = "NeuralVerge Web Search"
    description: str = "Search the web. Returns ranked results with title, url and snippet. Costs 5 points."
    args_schema: type[BaseModel] = NeuralVergeWebSearchSchema
    endpoint: ClassVar[str] = "run-search"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "query": kwargs["query"],
            "settings": {
                "country": kwargs.get("country", "us"),
                "language": kwargs.get("language", "en"),
                "max_results": kwargs.get("max_results", 10),
            },
        }


class NeuralVergeExtractSchema(BaseModel):
    """Input schema for NeuralVergeExtractTool."""

    url: str = Field(..., description="Page URL to load.")
    instructions: str = Field(
        ...,
        description="What to extract, e.g. 'Collect company name and contact details.'",
    )
    extract_schema_json: str | None = Field(
        default=None,
        description="Optional JSON Schema, serialized as a string, that pins the output shape.",
    )
    country_code: str = Field(
        default="us", description="Two-letter country for the fetch."
    )


class NeuralVergeExtractTool(NeuralVergeBaseTool):
    """AI extraction from any URL."""

    name: str = "NeuralVerge Extract from URL"
    description: str = (
        "Load any web page and extract structured JSON from it following the "
        "instructions (optionally a JSON Schema). Costs about 5 points."
    )
    args_schema: type[BaseModel] = NeuralVergeExtractSchema
    endpoint: ClassVar[str] = "run-extract"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "url": kwargs["url"],
            "instructions": kwargs["instructions"],
            "settings": drop_none(
                {
                    "country_code": kwargs.get("country_code", "us"),
                    "extract_schema_json": kwargs.get("extract_schema_json"),
                }
            ),
        }


class NeuralVergeResearchSchema(BaseModel):
    """Input schema for NeuralVergeResearchTool."""

    instructions: str = Field(..., description="The research task or question.")
    country_code: str = Field(
        default="us", description="Two-letter country for search."
    )
    search_enabled: bool = Field(default=True, description="Allow web search.")
    deepsearch_model: str = Field(
        default="base", description="Depth tier for the research step."
    )
    finalizer_model: str | None = Field(
        default=None, description="Optional model that writes the final report."
    )
    extract_schema_json: str | None = Field(
        default=None,
        description="Optional JSON Schema, serialized as a string, for the structured result.",
    )


class NeuralVergeResearchTool(NeuralVergeAsyncTaskTool):
    """Multi-step AI research with polling."""

    name: str = "NeuralVerge Research"
    description: str = (
        "Run a multi-step AI web research task and return the report (Markdown plus "
        "structured JSON). Waits until the task completes; can take several minutes. "
        "Costs 20-400 points depending on depth."
    )
    args_schema: type[BaseModel] = NeuralVergeResearchSchema
    endpoint: ClassVar[str] = "run-research"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "instructions": kwargs["instructions"],
            "settings": drop_none(
                {
                    "country_code": kwargs.get("country_code", "us"),
                    "search_enabled": kwargs.get("search_enabled", True),
                    "deepsearch_model": kwargs.get("deepsearch_model", "base"),
                    "finalizer_model": kwargs.get("finalizer_model"),
                    "extract_schema_json": kwargs.get("extract_schema_json"),
                }
            ),
        }

    def _run(self, **kwargs: Any) -> str | ToolFailure:
        started = self._post(self.build_payload(**kwargs))
        if isinstance(started, ToolFailure):
            return started
        session_id = started.get("session_id")
        if not session_id:
            return ToolFailure(
                message="NeuralVerge run-research did not return a session_id.",
                code="missing_session_id",
            )
        return self._wait(str(session_id))


# --------------------------------------------------------------------------- LinkedIn
class NeuralVergeLinkedInProfileSchema(BaseModel):
    """Input schema for NeuralVergeLinkedInProfileEmailTool."""

    profile_url: str = Field(
        ...,
        description="Full LinkedIn profile URL, e.g. 'https://www.linkedin.com/in/janedoe/'.",
    )
    include_email: bool = Field(default=True, description="Also look up a work email.")


class NeuralVergeLinkedInProfileEmailTool(NeuralVergeBaseTool):
    """LinkedIn profile plus work email."""

    name: str = "NeuralVerge LinkedIn Profile with Email"
    description: str = (
        "Get a LinkedIn profile's details plus a work email when available, from the "
        "profile URL. Costs 10 points."
    )
    args_schema: type[BaseModel] = NeuralVergeLinkedInProfileSchema
    endpoint: ClassVar[str] = "run-linkedin-email"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "username": kwargs["profile_url"],
            "includeEmail": kwargs.get("include_email", True),
        }


class NeuralVergeLinkedInFinderSchema(BaseModel):
    """Input schema for NeuralVergeLinkedInProfileFinderTool."""

    full_name: str = Field(..., description="Person's full name, e.g. 'Jane Doe'.")
    company_or_domain: str = Field(
        ..., description="Company name or domain, e.g. 'example.com'."
    )


class NeuralVergeLinkedInProfileFinderTool(NeuralVergeBaseTool):
    """Find a LinkedIn profile by name and company."""

    name: str = "NeuralVerge Find LinkedIn Profile"
    description: str = (
        "Resolve a full name plus company name or domain to a LinkedIn profile URL, "
        "headline and company. Costs 10 points."
    )
    args_schema: type[BaseModel] = NeuralVergeLinkedInFinderSchema
    endpoint: ClassVar[str] = "run-linkedin-domain"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "full_name": kwargs["full_name"],
            "company_or_domain": kwargs["company_or_domain"],
        }


class NeuralVergeLinkedInCompanySearchSchema(BaseModel):
    """Input schema for NeuralVergeLinkedInCompanySearchTool."""

    search_query: str = Field(..., description="Keyword, e.g. 'AI companies'.")
    company_size: list[str] | None = Field(
        default=None, description="Size buckets, e.g. ['51-200']."
    )
    industry_ids: list[str] | None = Field(
        default=None, description="LinkedIn industry IDs."
    )
    locations: list[str] | None = Field(
        default=None, description="Locations, e.g. ['Amsterdam']."
    )
    max_items: int = Field(default=10, description="Maximum companies to return.")
    start_page: int | None = Field(
        default=None, description="Result page to start from."
    )


class NeuralVergeLinkedInCompanySearchTool(NeuralVergeBaseTool):
    """LinkedIn company search."""

    name: str = "NeuralVerge LinkedIn Company Search"
    description: str = (
        "Search LinkedIn companies by keyword, size, industry and location. "
        "Costs 5 points per company."
    )
    args_schema: type[BaseModel] = NeuralVergeLinkedInCompanySearchSchema
    endpoint: ClassVar[str] = "run-linkedin-company-search"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "searchQuery": kwargs["search_query"],
            "companySize": kwargs.get("company_size"),
            "industryIds": kwargs.get("industry_ids"),
            "locations": kwargs.get("locations"),
            "maxItems": kwargs.get("max_items", 10),
            "startPage": kwargs.get("start_page"),
        }


class NeuralVergeLinkedInPeopleSearchSchema(BaseModel):
    """Input schema for NeuralVergeLinkedInPeopleSearchTool."""

    search_query: str | None = Field(default=None, description="Free-text keyword.")
    current_company: list[str] | None = Field(
        default=None, description="Current employers."
    )
    current_job_title: list[str] | None = Field(
        default=None, description="Current job titles."
    )
    locations: list[str] | None = Field(default=None, description="Locations.")
    seniority_level: list[str] | None = Field(
        default=None, description="Seniority levels."
    )
    function: list[str] | None = Field(default=None, description="Job functions.")
    past_company: list[str] | None = Field(default=None, description="Past employers.")
    max_results: int = Field(default=25, description="Maximum profiles to return.")
    start_page: int | None = Field(
        default=None, description="Result page to start from."
    )


class NeuralVergeLinkedInPeopleSearchTool(NeuralVergeBaseTool):
    """LinkedIn people search."""

    name: str = "NeuralVerge LinkedIn People Search"
    description: str = (
        "Search LinkedIn people by keyword, company, job title, seniority, function "
        "and location. Billed in blocks of 25 results (100 points per block)."
    )
    args_schema: type[BaseModel] = NeuralVergeLinkedInPeopleSearchSchema
    endpoint: ClassVar[str] = "run-linkedin-people-search"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "searchQuery": kwargs.get("search_query"),
            "currentCompany": kwargs.get("current_company"),
            "currentJobTitleFilter": kwargs.get("current_job_title"),
            "locations": kwargs.get("locations"),
            "seniorityLevelFilter": kwargs.get("seniority_level"),
            "functionFilter": kwargs.get("function"),
            "pastCompany": kwargs.get("past_company"),
            "maxResults": kwargs.get("max_results", 25),
            "startPage": kwargs.get("start_page"),
        }


class NeuralVergeLinkedInEmployeesSchema(BaseModel):
    """Input schema for NeuralVergeLinkedInCompanyEmployeesTool."""

    companies: list[str] = Field(
        ...,
        description="LinkedIn company URLs, e.g. ['https://www.linkedin.com/company/example/'].",
    )
    search_query: str | None = Field(
        default=None, description="Keyword filter, e.g. 'recruiter'."
    )
    current_job_title: list[str] | None = Field(
        default=None, description="Current job titles."
    )
    locations: list[str] | None = Field(default=None, description="Locations.")
    seniority_level: list[str] | None = Field(
        default=None, description="Seniority levels."
    )
    max_results: int = Field(default=10, description="Maximum profiles to return.")
    start_page: int | None = Field(
        default=None, description="Result page to start from."
    )


class NeuralVergeLinkedInCompanyEmployeesTool(NeuralVergeBaseTool):
    """Employees of LinkedIn companies."""

    name: str = "NeuralVerge LinkedIn Company Employees"
    description: str = (
        "List employees of one or more companies (LinkedIn company URLs), with "
        "title, location and seniority filters. Costs 30 points per run plus 5 per profile."
    )
    args_schema: type[BaseModel] = NeuralVergeLinkedInEmployeesSchema
    endpoint: ClassVar[str] = "run-linkedin-company-employee"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "companies": kwargs["companies"],
            "searchQuery": kwargs.get("search_query"),
            "currentJobTitleFilter": kwargs.get("current_job_title"),
            "locations": kwargs.get("locations"),
            "seniorityLevelFilter": kwargs.get("seniority_level"),
            "maxResults": kwargs.get("max_results", 10),
            "startPage": kwargs.get("start_page"),
        }


# ----------------------------------------------------------------------------- Amazon
class NeuralVergeAmazonSearchSchema(BaseModel):
    """Input schema for NeuralVergeAmazonProductSearchTool."""

    query: str = Field(..., description="Search keyword, e.g. 'wireless mouse'.")
    domain: str = Field(default="amazon.com", description=_DOMAIN_DESCRIPTION)
    max_items: int = Field(default=20, description="Maximum products (up to 250).")
    start_page: int | None = Field(
        default=None, description="Result page to start from."
    )


class NeuralVergeAmazonProductSearchTool(NeuralVergeBaseTool):
    """Amazon product search."""

    name: str = "NeuralVerge Amazon Product Search"
    description: str = (
        "Search Amazon product listings for a keyword on any of 19 marketplaces. "
        "Costs 1 point per product."
    )
    args_schema: type[BaseModel] = NeuralVergeAmazonSearchSchema
    endpoint: ClassVar[str] = "run-amazon-product-search"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "query": kwargs["query"],
            "domain": kwargs.get("domain", "amazon.com"),
            "max_items": kwargs.get("max_items", 20),
            "start_page": kwargs.get("start_page"),
        }


class NeuralVergeAsinSchema(BaseModel):
    """Input schema for tools that take one ASIN."""

    asin: str = Field(..., description="Amazon ASIN, e.g. 'B004YAVF8I'.")
    domain: str = Field(default="amazon.com", description=_DOMAIN_DESCRIPTION)


class NeuralVergeAmazonProductTool(NeuralVergeBaseTool):
    """Amazon product detail."""

    name: str = "NeuralVerge Amazon Product"
    description: str = (
        "Amazon product detail by ASIN: title, brand, price, rating, images, "
        "features, specs and variations. Costs 5 points."
    )
    args_schema: type[BaseModel] = NeuralVergeAsinSchema
    endpoint: ClassVar[str] = "run-amazon-product"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"asin": kwargs["asin"], "domain": kwargs.get("domain", "amazon.com")}


class NeuralVergeAmazonOfferTool(NeuralVergeBaseTool):
    """Amazon Buy Box offer."""

    name: str = "NeuralVerge Amazon Buy Box Offer"
    description: str = "Current Buy Box offer and its seller for an Amazon ASIN. Costs 5 points per offer."
    args_schema: type[BaseModel] = NeuralVergeAsinSchema
    endpoint: ClassVar[str] = "run-amazon-product-offers"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {"asin": kwargs["asin"], "domain": kwargs.get("domain", "amazon.com")}


class NeuralVergeAmazonSellerSchema(BaseModel):
    """Input schema for NeuralVergeAmazonSellerTool."""

    seller: str = Field(..., description="Amazon seller ID, e.g. 'A2L77EE7U53NWQ'.")
    domain: str = Field(default="amazon.com", description=_DOMAIN_DESCRIPTION)


class NeuralVergeAmazonSellerTool(NeuralVergeBaseTool):
    """Amazon seller profile."""

    name: str = "NeuralVerge Amazon Seller"
    description: str = (
        "Amazon seller profile: name, rating, 365-day rating count, positive feedback "
        "share and business details. Costs 5 points."
    )
    args_schema: type[BaseModel] = NeuralVergeAmazonSellerSchema
    endpoint: ClassVar[str] = "run-amazon-seller"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "seller": kwargs["seller"],
            "domain": kwargs.get("domain", "amazon.com"),
        }


class NeuralVergeAmazonSellerProductsSchema(BaseModel):
    """Input schema for NeuralVergeAmazonSellerProductsTool."""

    seller: str = Field(..., description="Amazon seller ID.")
    domain: str = Field(default="amazon.com", description=_DOMAIN_DESCRIPTION)
    max_items: int = Field(default=20, description="Maximum products to return.")
    start_page: int | None = Field(
        default=None, description="Result page to start from."
    )


class NeuralVergeAmazonSellerProductsTool(NeuralVergeBaseTool):
    """Amazon seller storefront listings."""

    name: str = "NeuralVerge Amazon Seller Products"
    description: str = (
        "Storefront product listings of an Amazon seller ID. Costs 1 point per product."
    )
    args_schema: type[BaseModel] = NeuralVergeAmazonSellerProductsSchema
    endpoint: ClassVar[str] = "run-amazon-seller-products"

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "seller": kwargs["seller"],
            "domain": kwargs.get("domain", "amazon.com"),
            "max_items": kwargs.get("max_items", 20),
            "start_page": kwargs.get("start_page"),
        }
