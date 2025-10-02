import asyncio
import os
from typing import Any

import aiohttp
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class BrightDataConfig(BaseModel):
    API_URL: str = "https://api.brightdata.com"
    DEFAULT_TIMEOUT: int = 600
    DEFAULT_POLLING_INTERVAL: int = 1

    @classmethod
    def from_env(cls):
        return cls(
            API_URL=os.environ.get("BRIGHTDATA_API_URL", "https://api.brightdata.com"),
            DEFAULT_TIMEOUT=int(os.environ.get("BRIGHTDATA_DEFAULT_TIMEOUT", "600")),
            DEFAULT_POLLING_INTERVAL=int(
                os.environ.get("BRIGHTDATA_DEFAULT_POLLING_INTERVAL", "1")
            ),
        )


class BrightDataDatasetToolException(Exception):  # noqa: N818
    """Exception raised for custom error in the application."""

    def __init__(self, message, error_code):
        self.message = message
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"{self.message} (Error Code: {self.error_code})"


class BrightDataDatasetToolSchema(BaseModel):
    """Schema for validating input parameters for the BrightDataDatasetTool.

    Attributes:
        dataset_type (str): Required Bright Data Dataset Type used to specify which dataset to access.
        format (str): Response format (json by default). Multiple formats exist - json, ndjson, jsonl, csv
        url (str): The URL from which structured data needs to be extracted.
        zipcode (Optional[str]): An optional ZIP code to narrow down the data geographically.
        additional_params (Optional[Dict]): Extra parameters for the Bright Data API call.
    """

    dataset_type: str = Field(..., description="The Bright Data Dataset Type")
    format: str | None = Field(
        default="json", description="Response format (json by default)"
    )
    url: str = Field(..., description="The URL to extract data from")
    zipcode: str | None = Field(default=None, description="Optional zipcode")
    additional_params: dict[str, Any] | None = Field(
        default=None, description="Additional params if any"
    )


config = BrightDataConfig.from_env()

BRIGHTDATA_API_URL = config.API_URL
timeout = config.DEFAULT_TIMEOUT

datasets = [
    {
        "id": "amazon_product",
        "dataset_id": "gd_l7q7dkf244hwjntr0",
        "description": "\n".join(
            [
                "Quickly read structured amazon product data.",
                "Requires a valid product URL with /dp/ in it.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "amazon_product_reviews",
        "dataset_id": "gd_le8e811kzy4ggddlq",
        "description": "\n".join(
            [
                "Quickly read structured amazon product review data.",
                "Requires a valid product URL with /dp/ in it.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "amazon_product_search",
        "dataset_id": "gd_lwdb4vjm1ehb499uxs",
        "description": "\n".join(
            [
                "Quickly read structured amazon product search data.",
                "Requires a valid search keyword and amazon domain URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["keyword", "url", "pages_to_search"],
        "defaults": {"pages_to_search": "1"},
    },
    {
        "id": "walmart_product",
        "dataset_id": "gd_l95fol7l1ru6rlo116",
        "description": "\n".join(
            [
                "Quickly read structured walmart product data.",
                "Requires a valid product URL with /ip/ in it.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "walmart_seller",
        "dataset_id": "gd_m7ke48w81ocyu4hhz0",
        "description": "\n".join(
            [
                "Quickly read structured walmart seller data.",
                "Requires a valid walmart seller URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "ebay_product",
        "dataset_id": "gd_ltr9mjt81n0zzdk1fb",
        "description": "\n".join(
            [
                "Quickly read structured ebay product data.",
                "Requires a valid ebay product URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "homedepot_products",
        "dataset_id": "gd_lmusivh019i7g97q2n",
        "description": "\n".join(
            [
                "Quickly read structured homedepot product data.",
                "Requires a valid homedepot product URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "zara_products",
        "dataset_id": "gd_lct4vafw1tgx27d4o0",
        "description": "\n".join(
            [
                "Quickly read structured zara product data.",
                "Requires a valid zara product URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "etsy_products",
        "dataset_id": "gd_ltppk0jdv1jqz25mz",
        "description": "\n".join(
            [
                "Quickly read structured etsy product data.",
                "Requires a valid etsy product URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "bestbuy_products",
        "dataset_id": "gd_ltre1jqe1jfr7cccf",
        "description": "\n".join(
            [
                "Quickly read structured bestbuy product data.",
                "Requires a valid bestbuy product URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "linkedin_person_profile",
        "dataset_id": "gd_l1viktl72bvl7bjuj0",
        "description": "\n".join(
            [
                "Quickly read structured linkedin people profile data.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "linkedin_company_profile",
        "dataset_id": "gd_l1vikfnt1wgvvqz95w",
        "description": "\n".join(
            [
                "Quickly read structured linkedin company profile data",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "linkedin_job_listings",
        "dataset_id": "gd_lpfll7v5hcqtkxl6l",
        "description": "\n".join(
            [
                "Quickly read structured linkedin job listings data",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "linkedin_posts",
        "dataset_id": "gd_lyy3tktm25m4avu764",
        "description": "\n".join(
            [
                "Quickly read structured linkedin posts data",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "linkedin_people_search",
        "dataset_id": "gd_m8d03he47z8nwb5xc",
        "description": "\n".join(
            [
                "Quickly read structured linkedin people search data",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url", "first_name", "last_name"],
    },
    {
        "id": "crunchbase_company",
        "dataset_id": "gd_l1vijqt9jfj7olije",
        "description": "\n".join(
            [
                "Quickly read structured crunchbase company data",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "zoominfo_company_profile",
        "dataset_id": "gd_m0ci4a4ivx3j5l6nx",
        "description": "\n".join(
            [
                "Quickly read structured ZoomInfo company profile data.",
                "Requires a valid ZoomInfo company URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "instagram_profiles",
        "dataset_id": "gd_l1vikfch901nx3by4",
        "description": "\n".join(
            [
                "Quickly read structured Instagram profile data.",
                "Requires a valid Instagram URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "instagram_posts",
        "dataset_id": "gd_lk5ns7kz21pck8jpis",
        "description": "\n".join(
            [
                "Quickly read structured Instagram post data.",
                "Requires a valid Instagram URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "instagram_reels",
        "dataset_id": "gd_lyclm20il4r5helnj",
        "description": "\n".join(
            [
                "Quickly read structured Instagram reel data.",
                "Requires a valid Instagram URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "instagram_comments",
        "dataset_id": "gd_ltppn085pokosxh13",
        "description": "\n".join(
            [
                "Quickly read structured Instagram comments data.",
                "Requires a valid Instagram URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "facebook_posts",
        "dataset_id": "gd_lyclm1571iy3mv57zw",
        "description": "\n".join(
            [
                "Quickly read structured Facebook post data.",
                "Requires a valid Facebook post URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "facebook_marketplace_listings",
        "dataset_id": "gd_lvt9iwuh6fbcwmx1a",
        "description": "\n".join(
            [
                "Quickly read structured Facebook marketplace listing data.",
                "Requires a valid Facebook marketplace listing URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "facebook_company_reviews",
        "dataset_id": "gd_m0dtqpiu1mbcyc2g86",
        "description": "\n".join(
            [
                "Quickly read structured Facebook company reviews data.",
                "Requires a valid Facebook company URL and number of reviews.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url", "num_of_reviews"],
    },
    {
        "id": "facebook_events",
        "dataset_id": "gd_m14sd0to1jz48ppm51",
        "description": "\n".join(
            [
                "Quickly read structured Facebook events data.",
                "Requires a valid Facebook event URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "tiktok_profiles",
        "dataset_id": "gd_l1villgoiiidt09ci",
        "description": "\n".join(
            [
                "Quickly read structured Tiktok profiles data.",
                "Requires a valid Tiktok profile URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "tiktok_posts",
        "dataset_id": "gd_lu702nij2f790tmv9h",
        "description": "\n".join(
            [
                "Quickly read structured Tiktok post data.",
                "Requires a valid Tiktok post URL.",
                "This can be a cache lookup, so it can be more reliable than scraping",
            ]
        ),
        "inputs": ["url"],
    },
    {
        "id": "tiktok_shop",
        "dataset_id": "gd_m45m1u911dsa4274pi",
        "description": "\n".join(
            [
                "Quickly read structured Tiktok shop data.",
                "Requires a valid Tiktok shop product URL.",
                "This can be a cache lookup...",
            ]
        ),
        "inputs": ["url"],
    },
]


class BrightDataDatasetTool(BaseTool):
    """CrewAI-compatible tool for scraping structured data using Bright Data Datasets.

    Attributes:
        name (str): Tool name displayed in the CrewAI environment.
        description (str): Tool description shown to agents or users.
        args_schema (Type[BaseModel]): Pydantic schema for validating input arguments.
    """

    name: str = "Bright Data Dataset Tool"
    description: str = "Scrapes structured data using Bright Data Dataset API from a URL and optional input parameters"
    args_schema: type[BaseModel] = BrightDataDatasetToolSchema
    dataset_type: str | None = None
    url: str | None = None
    format: str = "json"
    zipcode: str | None = None
    additional_params: dict[str, Any] | None = None
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BRIGHT_DATA_API_KEY",
                description="API key for Bright Data",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        dataset_type: str | None = None,
        url: str | None = None,
        format: str = "json",
        zipcode: str | None = None,
        additional_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.dataset_type = dataset_type
        self.url = url
        self.format = format
        self.zipcode = zipcode
        self.additional_params = additional_params

    def filter_dataset_by_id(self, target_id):
        return [dataset for dataset in datasets if dataset["id"] == target_id]

    async def get_dataset_data_async(
        self,
        dataset_type: str,
        output_format: str,
        url: str,
        zipcode: str | None = None,
        additional_params: dict[str, Any] | None = None,
        polling_interval: int = 1,
    ) -> str:
        """Asynchronously trigger and poll Bright Data dataset scraping.

        Args:
            dataset_type (str): Bright Data Dataset Type.
            url (str): Target URL to scrape.
            zipcode (Optional[str]): Optional ZIP code for geo-specific data.
            additional_params (Optional[Dict]): Extra API parameters.
            polling_interval (int): Time interval in seconds between polling attempts.

        Returns:
            Dict: Structured dataset result from Bright Data.

        Raises:
            Exception: If any API step fails or the job fails.
            TimeoutError: If polling times out before job completion.
        """
        request_data = {"url": url}
        if zipcode is not None:
            request_data["zipcode"] = zipcode

        # Set additional parameters dynamically depending upon the dataset that is being requested
        if additional_params:
            request_data.update(additional_params)

        api_key = os.getenv("BRIGHT_DATA_API_KEY")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        dataset_id = ""
        dataset = self.filter_dataset_by_id(dataset_type)

        if len(dataset) == 1:
            dataset_id = dataset[0]["dataset_id"]
        else:
            raise ValueError(
                f"Unable to find the dataset for {dataset_type}. Please make sure to pass a valid one"
            )

        async with aiohttp.ClientSession() as session:
            # Step 1: Trigger job
            async with session.post(
                f"{BRIGHTDATA_API_URL}/datasets/v3/trigger",
                params={"dataset_id": dataset_id, "include_errors": "true"},
                json=[request_data],
                headers=headers,
            ) as trigger_response:
                if trigger_response.status != 200:
                    raise BrightDataDatasetToolException(
                        f"Trigger failed: {await trigger_response.text()}",
                        trigger_response.status,
                    )
                trigger_data = await trigger_response.json()
                snapshot_id = trigger_data.get("snapshot_id")

            # Step 2: Poll for completion
            elapsed = 0
            while elapsed < timeout:
                await asyncio.sleep(polling_interval)
                elapsed += polling_interval

                async with session.get(
                    f"{BRIGHTDATA_API_URL}/datasets/v3/progress/{snapshot_id}",
                    headers=headers,
                ) as status_response:
                    if status_response.status != 200:
                        raise BrightDataDatasetToolException(
                            f"Status check failed: {await status_response.text()}",
                            status_response.status,
                        )
                    status_data = await status_response.json()
                    if status_data.get("status") == "ready":
                        break
                    if status_data.get("status") == "error":
                        raise BrightDataDatasetToolException(
                            f"Job failed: {status_data}", 0
                        )
            else:
                raise TimeoutError("Polling timed out before job completed.")

            # Step 3: Retrieve result
            async with session.get(
                f"{BRIGHTDATA_API_URL}/datasets/v3/snapshot/{snapshot_id}",
                params={"format": output_format},
                headers=headers,
            ) as snapshot_response:
                if snapshot_response.status != 200:
                    raise BrightDataDatasetToolException(
                        f"Result fetch failed: {await snapshot_response.text()}",
                        snapshot_response.status,
                    )

                return await snapshot_response.text()

    def _run(
        self,
        url: str | None = None,
        dataset_type: str | None = None,
        format: str | None = None,
        zipcode: str | None = None,
        additional_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        dataset_type = dataset_type or self.dataset_type
        output_format = format or self.format
        url = url or self.url
        zipcode = zipcode or self.zipcode
        additional_params = additional_params or self.additional_params

        if not dataset_type:
            raise ValueError(
                "dataset_type is required either in constructor or method call"
            )
        if not url:
            raise ValueError("url is required either in constructor or method call")

        valid_output_formats = {"json", "ndjson", "jsonl", "csv"}
        if output_format not in valid_output_formats:
            raise ValueError(
                f"Unsupported output format: {output_format}. Must be one of {', '.join(valid_output_formats)}."
            )

        api_key = os.getenv("BRIGHT_DATA_API_KEY")
        if not api_key:
            raise ValueError("BRIGHT_DATA_API_KEY environment variable is required.")

        try:
            return asyncio.run(
                self.get_dataset_data_async(
                    dataset_type=dataset_type,
                    output_format=output_format,
                    url=url,
                    zipcode=zipcode,
                    additional_params=additional_params,
                )
            )
        except TimeoutError as e:
            return f"Timeout Exception occured in method : get_dataset_data_async. Details - {e!s}"
        except BrightDataDatasetToolException as e:
            return (
                f"Exception occured in method : get_dataset_data_async. Details - {e!s}"
            )
        except Exception as e:
            return f"Bright Data API error: {e!s}"
