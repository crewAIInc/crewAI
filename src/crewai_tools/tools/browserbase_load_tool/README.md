# BrowserbaseLoadTool

## Description

[Browserbase](https://browserbase.com) is a serverless platform for running headless browsers, it offers advanced debugging, session recordings, stealth mode, integrated proxies and captcha solving.

## Installation

- Get an API key from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk) along with `crewai[tools]` package:

```
pip install browserbase 'crewai[tools]'
```

## Example

Utilize the BrowserbaseLoadTool as follows to allow your agent to load websites:

```python
from crewai_tools import BrowserbaseLoadTool

tool = BrowserbaseLoadTool()
```

## Arguments

- `api_key`: Optional. Specifies Browserbase API key. Defaults is the `BROWSERBASE_API_KEY` environment variable.
- `text_content`: Optional. Load pages as readable text. Default is `False`.
