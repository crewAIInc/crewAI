import time
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


class FixedSeleniumScrapingToolSchema(BaseModel):
    """Input for SeleniumScrapingTool."""


class SeleniumScrapingToolSchema(FixedSeleniumScrapingToolSchema):
    """Input for SeleniumScrapingTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")
    css_element: str = Field(
        ...,
        description="Mandatory css reference for element to scrape from the website",
    )


class SeleniumScrapingTool(BaseTool):
    name: str = "Read a website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = SeleniumScrapingToolSchema
    website_url: Optional[str] = None
    driver: Optional[Any] = webdriver.Chrome
    cookie: Optional[dict] = None
    wait_time: Optional[int] = 3
    css_element: Optional[str] = None
    return_html: Optional[bool] = False

    def __init__(
        self,
        website_url: Optional[str] = None,
        cookie: Optional[dict] = None,
        css_element: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if cookie is not None:
            self.cookie = cookie

        if css_element is not None:
            self.css_element = css_element

        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedSeleniumScrapingToolSchema

        self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        website_url = kwargs.get("website_url", self.website_url)
        css_element = kwargs.get("css_element", self.css_element)
        return_html = kwargs.get("return_html", self.return_html)
        driver = self._create_driver(website_url, self.cookie, self.wait_time)

        content = self._get_content(driver, css_element, return_html)
        driver.close()

        return "\n".join(content)

    def _get_content(self, driver, css_element, return_html):
        content = []

        if self._is_css_element_empty(css_element):
            content.append(self._get_body_content(driver, return_html))
        else:
            content.extend(self._get_elements_content(driver, css_element, return_html))

        return content

    def _is_css_element_empty(self, css_element):
        return css_element is None or css_element.strip() == ""

    def _get_body_content(self, driver, return_html):
        body_element = driver.find_element(By.TAG_NAME, "body")

        return (
            body_element.get_attribute("outerHTML")
            if return_html
            else body_element.text
        )

    def _get_elements_content(self, driver, css_element, return_html):
        elements_content = []

        for element in driver.find_elements(By.CSS_SELECTOR, css_element):
            elements_content.append(
                element.get_attribute("outerHTML") if return_html else element.text
            )

        return elements_content

    def _create_driver(self, url, cookie, wait_time):
        options = Options()
        options.add_argument("--headless")
        driver = self.driver(options=options)
        driver.get(url)
        time.sleep(wait_time)
        if cookie:
            driver.add_cookie(cookie)
            time.sleep(wait_time)
            driver.get(url)
            time.sleep(wait_time)
        return driver

    def close(self):
        self.driver.close()
