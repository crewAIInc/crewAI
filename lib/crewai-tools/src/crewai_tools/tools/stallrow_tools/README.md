# StallRowPageMetadataTool Documentation

## Description
This tool fetches one web page's declared metadata — title, description, canonical URL, OpenGraph and Twitter card fields, document language, favicon, the h1/h2 heading outline, and word count — from the [StallRow](https://stall-row-production.up.railway.app) `POST /v1/page-metadata` endpoint. It is the first pay-per-call tool in `crewai_tools`: there is no API key, account, or subscription. Each call costs **$0.005 USDC**, paid automatically over the [x402 protocol](https://www.x402.org) on Base (EIP-3009) from a wallet you configure. Use it when an agent needs a page's identity and shape before deciding whether the full content is worth fetching.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
pip install 'crewai-tools[stallrow]'   # adds eth-account, the payment signer
```

## Example
The following example demonstrates how to initialize the tool and fetch a page's metadata:
```python
from crewai_tools import StallRowPageMetadataTool

# Initialize the tool; the payer wallet is read from the environment
tool = StallRowPageMetadataTool()

result = tool.run(url="https://www.x402.org/")
print(result)
# {"url": ..., "final_url": ..., "title": "x402 — internet-native payments",
#  "description": ..., "canonical": ..., "lang": "en", "favicon": ...,
#  "og": {...}, "twitter": {...}, "headings": {"h1": [...], "h2": [...]},
#  "word_count": 120}
```

## Steps to Get Started
To effectively use the `StallRowPageMetadataTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` and `crewai-tools[stallrow]` packages are installed in your Python environment.
2. **Wallet Configuration**: Set the environment variable `STALLROW_WALLET_PRIVATE_KEY` to the private key of a Base wallet holding a small amount of USDC. The key is used only to sign payments locally; it is never sent anywhere. Without it, the tool fails closed and pays nothing.
3. **Price Cap (optional)**: The per-call price is $0.005 USDC. A hard cap refuses any higher price before anything is signed. The cap defaults to $0.005 and can be changed with the `STALLROW_MAX_PRICE_USDC` environment variable or the tool's `max_price_usdc` parameter.

## Conclusion
By integrating the `StallRowPageMetadataTool` into Python projects, users gain access to fast, structured page metadata inside their agent workflows, paid per call in USDC over x402 — payment is the auth, and the spend cap makes the cost predictable by construction.
