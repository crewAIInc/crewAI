"""Request payload builders shared by MrScraper operations."""

import json
from typing import Any


def _include_if_present(
    payload: dict[str, Any], key: str, value: Any
) -> dict[str, Any]:
    """Add a payload value unless it is absent."""
    if value is not None:
        payload[key] = value
    return payload


def append_output_schema(
    prompt: str | None, output_schema: dict[str, Any] | None, label: str
) -> str | None:
    """Append a compact output schema exactly once using the API's label."""
    if output_schema is None:
        return prompt
    schema = json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
    schema_instruction = f"{label}\n{schema}"
    return (
        f"{prompt}\n\n{schema_instruction}"
        if prompt is not None
        else schema_instruction
    )


def general_payload(
    *,
    url: str,
    prompt: str | None,
    output_schema: dict[str, Any] | None,
    mode: str,
    proxy_country: str | None,
) -> dict[str, Any]:
    """Build a General AI extraction payload."""
    payload: dict[str, Any] = {"graph": "general", "url": url, "mode": mode}
    message = append_output_schema(
        prompt, output_schema, "Return the output as JSON matching this schema:"
    )
    _include_if_present(payload, "message", message)
    _include_if_present(payload, "proxyCountry", proxy_country)
    return payload


def listing_payload(
    *,
    url: str,
    prompt: str | None,
    output_schema: dict[str, Any] | None,
    max_pages: int,
    proxy_country: str | None,
) -> dict[str, Any]:
    """Build a Listing AI extraction payload."""
    payload: dict[str, Any] = {
        "graph": "listing",
        "url": url,
        "maxPages": max_pages,
    }
    message = append_output_schema(
        prompt, output_schema, "Return each item as JSON matching this schema:"
    )
    _include_if_present(payload, "message", message)
    _include_if_present(payload, "proxyCountry", proxy_country)
    return payload


def map_payload(
    *,
    url: str,
    max_depth: int,
    max_pages: int,
    limit: int,
    include_patterns: str | None,
    exclude_patterns: str | None,
) -> dict[str, Any]:
    """Build a Map AI crawl payload."""
    payload: dict[str, Any] = {
        "graph": "map",
        "url": url,
        "maxDepth": max_depth,
        "maxPages": max_pages,
        "limit": limit,
    }
    _include_if_present(payload, "includePatterns", include_patterns)
    _include_if_present(payload, "excludePatterns", exclude_patterns)
    return payload


def rendered_request(
    *,
    url: str,
    max_retries: int,
    timeout: int,
    geo_code: str,
    proxy_country: str,
    screenshot: bool,
    screenshot_mode: str | None,
    html: bool,
    markdown: bool,
    token_cap: int | None,
    wait_for_selector: str | None,
    wait_until: str | None,
    block_resources: bool,
    home_page: bool,
    return_cookie: bool,
    super_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split rendered-page options into query parameters and a JSON body."""

    def bool_text(value: bool) -> str:
        """Serialize a boolean for rendered-page query parameters."""
        return "true" if value else "false"

    params: dict[str, Any] = {
        "timeout": timeout,
        "geoCode": geo_code,
        "html": bool_text(html),
        "markdown": bool_text(markdown),
        "proxyCountry": proxy_country,
    }
    if screenshot:
        params["screenshot"] = screenshot_mode or "full"
    _include_if_present(params, "waitForSelector", wait_for_selector)
    _include_if_present(params, "waitUntil", wait_until)
    if block_resources:
        params["blockResources"] = bool_text(block_resources)
    if return_cookie:
        params["returnCookie"] = bool_text(return_cookie)
    if super_mode:
        params["super"] = bool_text(super_mode)
    body = {
        "url": url,
        "maxRetries": max_retries,
    }
    _include_if_present(body, "tokenCap", token_cap)
    if home_page:
        body["homePage"] = home_page
    return params, body


def existing_run_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Build an agent-specific rerun payload from validated public inputs."""
    payload: dict[str, Any] = {
        "scraperId": values["scraper_id"],
        "url": values["url"],
        "maxRetry": values["max_retry"],
    }
    _include_if_present(payload, "proxyCountry", values.get("proxy_country"))

    if values["scraper_type"] == "manual":
        mapping = {
            "bypass_proxy": "bypassProxy",
            "cookie_jar": "cookieJar",
            "cookies": "cookies",
            "home_page": "homePage",
            "home_page_timeout": "homePageTimeout",
            "html": "html",
            "markdown": "markdown",
            "paginator": "paginator",
            "proxy": "proxy",
            "record": "record",
            "return_cookie": "returnCookie",
            "stream": "stream",
            "timeout": "timeout",
            "token_cap": "tokenCap",
        }
        for source, target in mapping.items():
            _include_if_present(payload, target, values.get(source))
        screenshot = values.get("screenshot")
        if screenshot is not None:
            payload["screenshot"] = "true" if screenshot else "false"
        return payload

    agent_type = values["agent_type"]
    if agent_type == "map":
        for source, target in {
            "max_depth": "maxDepth",
            "max_pages": "maxPages",
            "limit": "limit",
            "include_patterns": "includePatterns",
            "exclude_patterns": "excludePatterns",
        }.items():
            _include_if_present(payload, target, values.get(source))
        return payload

    ai_mapping = {
        "bypass_proxy": "bypassProxy",
        "html": "html",
        "markdown": "markdown",
        "render_javascript": "renderJavascript",
        "return_cookies": "returnCookies",
        "screenshot": "screenshot",
        "use_home_page": "useHomePage",
        "wait_for_selector": "waitForSelector",
    }
    for source, target in ai_mapping.items():
        _include_if_present(payload, target, values.get(source))
    if agent_type == "listing":
        for source, target in {
            "max_pages": "maxPages",
            "timeout": "timeout",
            "stream": "stream",
        }.items():
            _include_if_present(payload, target, values.get(source))
    return payload


__all__ = [
    "append_output_schema",
    "existing_run_payload",
    "general_payload",
    "listing_payload",
    "map_payload",
    "rendered_request",
]
