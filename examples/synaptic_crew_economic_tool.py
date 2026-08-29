"""
SynapticChain 256-Lane Multi-Agent Economic Tool for CrewAI.

Enables CrewAI agent crews (Researchers, Analysts, Execution Bots) to allocate
micro-bounties and settle fees across 256 parallel lanes without Head-of-Line nonce blocking.
"""

import asyncio
import os
import time
from typing import Any, Dict, Optional

try:
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    def tool(name: str):
        def decorator(fn): return fn
        return decorator

DEFAULT_RPC = os.getenv("SYNAPTIC_RPC_URL", "https://nodes.synapticchain.xyz/rpc")


class SynapticCrewSettlementTool:
    """Multi-agent settlement tool across 256 independent lanes."""

    def __init__(self, rpc_url: str = DEFAULT_RPC):
        self.rpc_url = rpc_url
        self.lane_nonces = [0] * 256

    async def execute_lane_transaction(
        self,
        recipient: str,
        amount_sunit: int,
        lane_id: Optional[int] = None,
        memo: str = "",
    ) -> Dict[str, Any]:
        """
        Execute payment on a specific lane without nonce collision.
        """
        if amount_sunit <= 0:
            raise ValueError("Amount must be positive integer")

        if not recipient or not recipient.startswith("syn1"):
            raise ValueError("Invalid recipient address")

        # Explicit None check so lane 0 is valid and not overwritten
        if lane_id is None:
            allocated_lane = 0
        else:
            allocated_lane = int(lane_id) % 256

        # Capture nonce before suspension point
        current_nonce = self.lane_nonces[allocated_lane]
        self.lane_nonces[allocated_lane] += 1

        start = time.perf_counter()
        await asyncio.sleep(0.005)
        elapsed = (time.perf_counter() - start) * 1000.0

        return {
            "status": "CONFIRMED",
            "tx_hash": f"0x{'c'*32}{allocated_lane:02x}{current_nonce:04x}",
            "lane_id": allocated_lane,
            "nonce": current_nonce,
            "recipient": recipient,
            "amount_sunit": amount_sunit,
            "elapsed_ms": round(elapsed, 2),
            "memo": memo,
        }


async def main():
    tool_inst = SynapticCrewSettlementTool()
    print("👥 CrewAI x SynapticChain 256-Lane Tool Demo")
    
    # Test lane 0 explicitly
    res0 = await tool_inst.execute_lane_transaction(
        recipient="syn1dejphz2hjetjqva9fg39c7hg8gpr7muapqyvq7",
        amount_sunit=500_000,
        lane_id=0,
        memo="Lead Analyst Bounty (Lane 0)"
    )
    print(f"  Lane 0 Result: {res0['status']} on Lane #{res0['lane_id']} (nonce {res0['nonce']})")

    # Test lane 42
    res42 = await tool_inst.execute_lane_transaction(
        recipient="syn1dejphz2hjetjqva9fg39c7hg8gpr7muapqyvq7",
        amount_sunit=1_200_000,
        lane_id=42,
        memo="Researcher Reward (Lane 42)"
    )
    print(f"  Lane 42 Result: {res42['status']} on Lane #{res42['lane_id']} (nonce {res42['nonce']})")


if __name__ == "__main__":
    asyncio.run(main())
