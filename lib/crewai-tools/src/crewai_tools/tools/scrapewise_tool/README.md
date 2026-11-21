# ScrapewiseProductDataTool

## Description

This tool reads structured competitor product and pricing data collected by
[ScrapeWise](https://scrapewise.ai) scrapers. ScrapeWise runs managed scrapers
against e-commerce sites and stores the extracted rows — title, price,
availability, product identifiers — so the agent gets clean structured data
without fetching or parsing any HTML itself.

Use it when the data you need is already being monitored on a schedule
(competitor prices, stock levels, assortment changes) rather than when you need
to scrape an arbitrary one-off page.

## Installation

Install the ScrapeWise client alongside `crewai-tools`:

```shell
pip install 'crewai[tools]' scrapewise
```

## Example Usage

Let the agent choose which scraper to read:

```python
from crewai_tools import ScrapewiseProductDataTool

tool = ScrapewiseProductDataTool()

agent = Agent(
    role="Pricing Analyst",
    goal="Report where our prices sit against the competition",
    tools=[tool],
)
```

Pin the tool to one scraper so the agent cannot read the wrong site:

```python
tool = ScrapewiseProductDataTool(scraper_id="65f1a2b3c4d5e6f708192a3b")
```

Read it directly:

```python
print(tool.run(scraper_id="65f1a2b3c4d5e6f708192a3b", max_rows=10))
```

## Arguments

- `scraper_id`: Required. The id of the ScrapeWise scraper to read. Omit it when
  the tool was constructed with a fixed `scraper_id` — the input schema then
  drops the field entirely.
- `max_rows`: Optional, defaults to `25`. How many product rows to return, 1 to
  100. The ScrapeWise API caps sample data at 100 rows, so values above that are
  rejected.

Constructor arguments:

- `scraper_id`: Optional. Pins the tool to one scraper.
- `api_key`: Optional. Falls back to `SCRAPEWISE_API_KEY`.
- `base_url`: Optional. Points the tool at a staging or self-hosted deployment.
- `timeout`: Optional. Per-request timeout in seconds. The tool does not set one
  of its own — when omitted, the `scrapewise` client's own default applies
  (60 seconds, as of `scrapewise` 0.1.0).

## Environment Variables

- `SCRAPEWISE_API_KEY`: Required. Your ScrapeWise API key, created in the portal
  under Settings → API Keys. The tool raises `ValueError` at construction time
  when it is missing, rather than failing later with a 401.

## Rate Limiting

ScrapeWise rate-limits per API key. The tool makes exactly one request per
invocation, so an agent that loops over many scrapers can hit the limit —
prefer a pinned `scraper_id` or a single wider read over many narrow ones.

## Error Handling

The tool returns errors as text rather than raising, so a recoverable mistake
does not abort the crew:

- A missing or wrong `scraper_id` returns an `Error reading ScrapeWise product
  data: ...` string carrying the API's own message.
- A scraper that has never run successfully returns an explicit "No product rows
  stored for this scraper yet" message rather than an empty result, so the agent
  does not read silence as "the competitor has no products".

Three failures are raised rather than returned, because they are configuration
problems and no retry will fix them: a missing `scrapewise` package
(`ImportError`, carrying the install command), a missing API key (`ValueError`)
and a `max_rows` outside 1–100 (`ValueError` from the input schema).

## Best Practices

- Pin `scraper_id` in the constructor whenever the crew only ever reads one
  site. It removes a whole class of agent error and shortens the input schema.
- Keep `max_rows` small. Product rows are wide; 25 rows of 15 fields is already
  a large chunk of context.
- Platform bookkeeping columns (`_sw_scraper`, `_sw_group`, `_sw_run_date`,
  `_sw_run_time`) are stripped from the output — they are noise to a model.
- This tool reads *sample* data, which the API caps at 100 rows. It is for
  analysis and spot checks, not for exporting a full dataset; use the ScrapeWise
  export for that.
