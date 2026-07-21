# Spraay Tools Documentation

## Description

A suite of CrewAI tools for batch cryptocurrency payments, escrow, and balance queries via the [Spraay x402 payment gateway](https://docs.spraay.app). These tools provide three capabilities:

- **SpraayBatchPaymentTool**: Validate, estimate, and execute batch payments to up to 200 recipients in a single transaction with ~80% gas savings
- **SpraayEscrowTool**: Create on-chain escrow contracts between two parties with programmable release conditions
- **SpraayBalanceTool**: Check token balances across 16 supported chains

The gateway uses the [x402 payment protocol](https://www.x402.org/) — no API key or signup required. Free endpoints (validate, estimate, balance) cost nothing. Paid endpoints (execute, escrow) are paid per request via x402 micropayment and require a funded wallet private key in the `SPRAAY_WALLET_PRIVATE_KEY` environment variable (see [Paid Endpoints and Wallet Setup](#paid-endpoints-and-wallet-setup)).

Supported chains include Base, Ethereum, Solana, Polygon, Arbitrum, Optimism, Avalanche, BNB Chain, and more.

## Installation

To incorporate these tools into your project, follow the installation instructions below:

```bash
pip install crewai[tools] requests
```

To use the paid endpoints (batch `execute` and escrow creation), also install the official [x402](https://pypi.org/project/x402/) package with its `requests` and EVM extras:

```bash
pip install 'x402[requests,evm]'
```

## Paid Endpoints and Wallet Setup

Free endpoints (validate, estimate, balance) need no setup. Paid endpoints — batch `execute` ($0.02/request) and escrow creation ($0.10/request) — are paid via x402 micropayment in USDC on Base, and require a funded wallet:

```bash
export SPRAAY_WALLET_PRIVATE_KEY="0xYourPrivateKey"
```

When a paid endpoint responds with an HTTP 402 payment challenge, the tool signs the payment requirements with this key via the official x402 client and retries the request with the payment header attached. The wallet must hold enough USDC on Base (chain ID 8453) to cover the per-request fee.

If `SPRAAY_WALLET_PRIVATE_KEY` is not set, the tool does not fail — it returns the gateway's parsed payment requirements as structured JSON (`"status": "payment_required"`) so the agent can report what payment is needed.

> **Security note:** The private key signs real on-chain payments. Use a dedicated wallet funded with only the amount you intend to spend, and never commit the key to source control.

## Examples

### Batch Payment - Validate Recipients

```python
from crewai_tools import SpraayBatchPaymentTool

tool = SpraayBatchPaymentTool()
result = tool.run(
    action="validate",
    token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC on Base
    recipients=[
        {"address": "0xAbc123...", "amount": "50.0"},
        {"address": "0xDef456...", "amount": "25.0"},
        {"address": "0x789Ghi...", "amount": "75.0"},
    ],
    chain_id=8453,
)
```

### Batch Payment - Estimate Gas Costs

```python
from crewai_tools import SpraayBatchPaymentTool

tool = SpraayBatchPaymentTool()
result = tool.run(
    action="estimate",
    token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    recipients=[
        {"address": "0xAbc123...", "amount": "50.0"},
        {"address": "0xDef456...", "amount": "25.0"},
    ],
    sender_address="0xYourWallet...",
    chain_id=8453,
)
```

### Batch Payment - Execute

```python
from crewai_tools import SpraayBatchPaymentTool

tool = SpraayBatchPaymentTool()
result = tool.run(
    action="execute",
    token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    recipients=[
        {"address": "0xAbc123...", "amount": "50.0"},
        {"address": "0xDef456...", "amount": "25.0"},
    ],
    sender_address="0xYourWallet...",
    chain_id=8453,
)
```

### Escrow - Create Contract

```python
from crewai_tools import SpraayEscrowTool

tool = SpraayEscrowTool()
result = tool.run(
    token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    amount="500.0",
    depositor="0xClientWallet...",
    beneficiary="0xFreelancerWallet...",
    chain_id=8453,
    conditions="Release upon delivery of completed project files",
)
```

### Balance - Check Wallet

```python
from crewai_tools import SpraayBalanceTool

tool = SpraayBalanceTool()
result = tool.run(
    wallet_address="0xYourWallet...",
    token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    chain_id=8453,
)
```

## Steps to Get Started

To effectively use the Spraay Tools, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.

2. **Tool Selection**: Choose the appropriate tool based on your needs:
   - Use **SpraayBatchPaymentTool** for sending payments to multiple recipients
   - Use **SpraayEscrowTool** for trustless escrow between two parties
   - Use **SpraayBalanceTool** for checking wallet balances before transactions

3. **Start with free endpoints**: The validate, estimate, and balance actions require no payment and no setup — call them immediately to test your integration.

## Batch Contract

The batch payment smart contract is deployed on Base at [`0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC`](https://basescan.org/address/0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC).

## Conclusion

By integrating Spraay Tools into your CrewAI agents, you give them the ability to handle real cryptocurrency payments — from payroll and grant distributions to escrow-protected freelance contracts. The free validation and estimation endpoints let agents plan transactions safely before committing funds, and the x402 protocol means there is no API key to manage — agents pay per request natively.
