"""TealTiger governance middleware — Dakera integration example.

This example shows how to wire DakeraCostStorage, DakeraDecisionStore, and
DakeraDelegationHelper into a TealTiger-governed agent loop.

All storage methods are async — use AsyncDakeraClient and run inside an
async context (asyncio.run).

Prerequisites:
    pip install dakera[tealtiger] tealtiger

Run:
    DAKERA_API_KEY=dk-mykey python examples/tealtiger_governance.py
"""

import asyncio
import os

from dakera.async_client import AsyncDakeraClient
from dakera.integrations.tealtiger import (
    DakeraCostStorage,
    DakeraDecisionStore,
    DakeraDelegationHelper,  # noqa: F401 — used in delegation chain (section 6, commented out)
)

# ---------------------------------------------------------------------------
# Mock helpers — replicate TealTiger types without requiring the package
# ---------------------------------------------------------------------------


class _MockProvider:
    value = "openai"


class _MockTokens:
    input_tokens = 800
    output_tokens = 200
    total_tokens = 1000


class _MockBreakdown:
    input_cost = 0.0400
    output_cost = 0.0080


class _MockRecord:
    id = "cost-demo-001"
    request_id = "req-demo-001"
    agent_id = "research-agent"
    model = "gpt-4o"
    provider = _MockProvider()
    actual_cost = 0.0480
    actual_tokens = _MockTokens()
    breakdown = _MockBreakdown()
    timestamp = "2026-06-14T12:00:00Z"

    def model_dump_json(self) -> str:
        import json

        return json.dumps(
            {
                "id": self.id,
                "request_id": self.request_id,
                "agent_id": self.agent_id,
                "model": self.model,
                "provider": self.provider.value,
                "actual_cost": self.actual_cost,
                "actual_tokens": {
                    "input_tokens": self.actual_tokens.input_tokens,
                    "output_tokens": self.actual_tokens.output_tokens,
                    "total_tokens": self.actual_tokens.total_tokens,
                },
                "breakdown": {
                    "input_cost": self.breakdown.input_cost,
                    "output_cost": self.breakdown.output_cost,
                },
                "timestamp": self.timestamp,
            }
        )


class _MockDecisionAction:
    value = "DENY"


class _MockDecision:
    action = _MockDecisionAction()
    correlation_id = "corr-demo-001"
    policy_id = "policy-demo-001"
    risk_score = 90
    reason = "Blocked by PII policy"

    def model_dump_json(self) -> str:
        import json

        return json.dumps(
            {
                "action": self.action.value,
                "correlation_id": self.correlation_id,
                "policy_id": self.policy_id,
                "risk_score": self.risk_score,
                "reason": self.reason,
            }
        )


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    # 1. Initialise the async Dakera client
    client = AsyncDakeraClient(
        base_url=os.environ.get("DAKERA_URL", "http://localhost:3000"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-devkey"),
    )

    # 2. Create Dakera-backed TealTiger storage objects
    cost_storage = DakeraCostStorage(client, dakera_agent_id="governance")
    decision_store = DakeraDecisionStore(client)
    # delegation_helper = DakeraDelegationHelper(client)  # used below for chain traversal

    print("DakeraCostStorage ready:", cost_storage._agent_id)

    # 3. Register cost_storage with a TealTiger client
    #
    #   from tealtiger import TealOpenAI, TealOpenAIConfig
    #   teal_client = TealOpenAI(config=TealOpenAIConfig(cost_storage=cost_storage))
    #
    #   # Or for Anthropic:
    #   from tealtiger import TealAnthropic, TealAnthropicConfig
    #   teal_client = TealAnthropic(config=TealAnthropicConfig(cost_storage=cost_storage))
    #
    # Every LLM call tracked by TealTiger will be persisted via Dakera asynchronously.

    # 4. Manual CostRecord example (works without TealTiger installed)
    record = _MockRecord()
    await cost_storage.store(record)
    print(
        f"Stored cost record: {record.id}"
        f" (agent={record.agent_id}, cost=${record.actual_cost:.4f})"
    )

    retrieved = await cost_storage.get("cost-demo-001")
    if retrieved:
        cost_val = (
            retrieved.get("actual_cost")
            if isinstance(retrieved, dict)
            else retrieved.actual_cost
        )
        print(f"Retrieved: cost=${cost_val:.4f}")

    # 5. Decision example (tealtiger.Decision API)
    decision = _MockDecision()
    decision_mem_id = await decision_store.store_receipt("research-agent", decision)
    print(f"Stored DENY decision: memory_id={decision_mem_id}")

    already_decided = await decision_store.is_terminal("research-agent", "corr-demo-001")
    print(f"Idempotency check — already decided: {already_decided}")

    # 6. Delegation chain example
    #
    #   await delegation_helper.link_delegation(
    #       child_id=child_receipt_mem_id,
    #       parent_id=parent_receipt_mem_id,
    #   )
    #   chain = await delegation_helper.get_delegation_chain(
    #       agent_id="research-agent",
    #       decision_id=parent_decision_mem_id,
    #       max_depth=5,
    #   )
    #   print("Delegation chain:", chain)

    print("\nDakera governance integration ready.")
    print("All cost records, decisions, and delegation chains are persisted in Dakera.")


if __name__ == "__main__":
    asyncio.run(main())
