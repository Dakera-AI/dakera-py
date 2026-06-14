"""TealTiger governance middleware — Dakera integration example.

This example shows how to wire DakeraCostStorage, DakeraDecisionStore, and
DakeraDelegationHelper into a TealTiger-governed agent loop.

Prerequisites:
    pip install dakera[tealtiger] tealtiger

Run:
    DAKERA_API_KEY=dk-mykey python examples/tealtiger_governance.py
"""

import os

from dakera import DakeraClient
from dakera.integrations.tealtiger import (
    DakeraCostStorage,
    DakeraDecisionStore,
    DakeraDelegationHelper,
)

# ---------------------------------------------------------------------------
# 1. Initialise the Dakera client
# ---------------------------------------------------------------------------

client = DakeraClient(
    base_url=os.environ.get("DAKERA_URL", "http://localhost:3000"),
    api_key=os.environ.get("DAKERA_API_KEY", "dk-devkey"),
)

# ---------------------------------------------------------------------------
# 2. Create Dakera-backed TealTiger storage objects
# ---------------------------------------------------------------------------

# All cost records go into the "governance" Dakera namespace by default.
cost_storage = DakeraCostStorage(client, dakera_agent_id="governance")

# Decision receipts are stored per-agent in the Dakera namespace that matches
# the TealTiger agent_id.
decision_store = DakeraDecisionStore(client)

# Delegation helper links decision receipt memories in the knowledge graph.
delegation_helper = DakeraDelegationHelper(client)

# ---------------------------------------------------------------------------
# 3. Register cost_storage with TealTiger middleware
# ---------------------------------------------------------------------------
#
# When TealTiger is installed, drop cost_storage into the middleware config:
#
#   from tealtiger import TealTigerMiddleware
#   middleware = TealTigerMiddleware(
#       cost_storage=cost_storage,
#       ...
#   )
#
# Every LLM call tracked by TealTiger will then be persisted via Dakera.

print("DakeraCostStorage ready:", cost_storage._agent_id)
print("DakeraDecisionStore ready:", decision_store._client.base_url)

# ---------------------------------------------------------------------------
# 4. Manual CostRecord example (works without TealTiger installed)
# ---------------------------------------------------------------------------


class _MockProvider:
    value = "openai"


class _MockRecord:
    id = "cost-demo-001"
    request_id = "req-demo-001"
    agent_id = "research-agent"
    model = "gpt-4o"
    provider = _MockProvider()
    actual_cost = 0.0480
    actual_tokens = None
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
                "timestamp": self.timestamp,
            }
        )


record = _MockRecord()
cost_storage.store(record)
print(f"Stored cost record: {record.id} (agent={record.agent_id}, cost=${record.actual_cost:.4f})")

retrieved = cost_storage.get("cost-demo-001")
if retrieved:
    if isinstance(retrieved, dict):
        cost_val = retrieved.get("actual_cost")
    else:
        cost_val = retrieved.actual_cost
    print(f"Retrieved: cost=${cost_val:.4f}")

# ---------------------------------------------------------------------------
# 5. Decision receipt example
# ---------------------------------------------------------------------------


class _MockDecision:
    value = "DENY"


class _MockReceipt:
    decision = _MockDecision()
    decision_id = "dec-demo-001"
    idempotency_key = "idem-demo-001"

    def model_dump_json(self) -> str:
        import json

        return json.dumps(
            {
                "decision": self.decision.value,
                "decision_id": self.decision_id,
                "idempotency_key": self.idempotency_key,
            }
        )


receipt = _MockReceipt()
receipt_mem_id = decision_store.store_receipt("research-agent", receipt)
print(f"Stored DENY receipt: memory_id={receipt_mem_id}")

already_decided = decision_store.is_terminal("research-agent", "idem-demo-001")
print(f"Idempotency check — already decided: {already_decided}")

# ---------------------------------------------------------------------------
# 6. Delegation chain example
# ---------------------------------------------------------------------------
#
# After storing two receipts (parent and child), link them and traverse:
#
#   delegation_helper.link_delegation(
#       child_id=child_receipt_mem_id,
#       parent_id=parent_receipt_mem_id,
#   )
#   chain = delegation_helper.get_delegation_chain(
#       agent_id="research-agent",
#       decision_id=parent_receipt_mem_id,
#       max_depth=5,
#   )
#   print("Delegation chain:", chain)
#
# The chain is stored in Dakera's knowledge graph and survives agent restarts.

print("\nDakera governance integration ready.")
print("All cost records, decisions, and delegation chains are persisted in Dakera.")
