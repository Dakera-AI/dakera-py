"""
LLM Chat Comparison — with and without Dakera memory

Demonstrates the pattern used by the Dakera playground: run the same user
query through two paths and compare responses.

  Path A (memory-augmented)  — recall relevant context, prepend to prompt
  Path B (baseline)          — send the raw prompt with no memory context

Requires:
  pip install dakera openai          # or any LLM client
  export DAKERA_URL=https://5-75-177-31.sslip.io
  export DAKERA_API_KEY=<your-key>
  export OPENAI_API_KEY=<your-key>   # only needed for the LLM call

Run:
  python examples/playground/chat_comparison.py
"""

import os

from dakera import DakeraClient
from dakera.session import ChatMemorySession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DAKERA_URL = os.environ.get("DAKERA_URL", "http://localhost:3000")
DAKERA_API_KEY = os.environ.get("DAKERA_API_KEY")
AGENT_ID = "playground-demo"


def build_context_prompt(memories: list, user_message: str) -> str:
    """Prepend recalled memory context to the user message."""
    if not memories:
        return user_message
    context_lines = "\n".join(
        f"- {m.content}" for m in memories
    )
    return (
        f"[Relevant context from memory]\n{context_lines}\n\n"
        f"[User message]\n{user_message}"
    )


def call_llm(prompt: str) -> str:
    """
    Placeholder for any LLM call.  Replace with your preferred provider.

    Returns a deterministic mock response so this file runs without
    credentials.
    """
    # Swap in real LLM call:
    #   from openai import OpenAI
    #   client = OpenAI()
    #   resp = client.chat.completions.create(
    #       model="gpt-4o-mini",
    #       messages=[{"role": "user", "content": prompt}],
    #   )
    #   return resp.choices[0].message.content
    if "[Relevant context from memory]" in prompt:
        return "I recall you mentioned this before. Here is a context-aware answer."
    return "I have no prior context. Here is a generic answer."


def main() -> None:
    client = DakeraClient(DAKERA_URL, api_key=DAKERA_API_KEY)

    print("=== Dakera Playground — LLM Chat Comparison Demo ===\n")

    # ------------------------------------------------------------------
    # Step 1: Seed some prior conversation turns
    # ------------------------------------------------------------------
    print("Seeding prior conversation turns into Dakera memory...")
    with ChatMemorySession.create(
        client,
        agent_id=AGENT_ID,
        metadata={"source": "playground-seed"},
    ) as seed_session:
        seed_session.store("user", "I'm building a chatbot in Python using LangChain.")
        seed_session.store(
            "assistant",
            "Great choice — LangChain has excellent memory integrations.",
        )
        seed_session.store("user", "My team prefers async code so we use FastAPI on the backend.")
        print(f"  Session {seed_session.session_id}: stored 3 turns\n")

    # ------------------------------------------------------------------
    # Step 2: Start a new session and compare responses
    # ------------------------------------------------------------------
    follow_up = "What framework should I use for the async background tasks?"

    with ChatMemorySession.create(
        client,
        agent_id=AGENT_ID,
        metadata={"source": "playground-compare"},
    ) as session:
        print(f"Comparison session: {session.session_id}")
        print(f"User: {follow_up}\n")

        # Path A — memory-augmented
        memories = session.recall(follow_up, top_k=5)
        augmented_prompt = build_context_prompt(memories, follow_up)
        response_with_memory = call_llm(augmented_prompt)

        # Path B — baseline (no memory)
        response_without_memory = call_llm(follow_up)

        # Store the actual exchange
        session.store("user", follow_up)
        session.store("assistant", response_with_memory)

    # ------------------------------------------------------------------
    # Step 3: Print side-by-side comparison
    # ------------------------------------------------------------------
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  WITHOUT Dakera memory                                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  {response_without_memory}")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  WITH Dakera memory                                         │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  {response_with_memory}")
    print("└─────────────────────────────────────────────────────────────┘")

    if memories:
        print(f"\n  Memory used: {len(memories)} relevant context item(s)")
        for m in memories:
            print(f"    • [{m.score:.2f}] {m.content[:80]}")


if __name__ == "__main__":
    main()
