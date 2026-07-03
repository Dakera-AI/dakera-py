#!/usr/bin/env python3
"""
Dakera Python SDK — persistent memory for a local Ollama chat loop.

Ollama's ``/api/chat`` endpoint is stateless: every call starts fresh. This
example wraps an Ollama chat with Dakera so the app transparently *recalls*
relevant long-term context before each turn and *stores* the turn afterwards.
Because memories are importance-scored and decay over time, stale context stops
competing with fresh, relevant facts.

Key Dakera features shown:
    * ``ChatMemorySession`` — a session-scoped helper that recalls across the
      agent's full memory (not just this session) and stores turns as episodic
      memories.
    * Importance weighting — durable preferences are stored at high importance
      so they outrank ordinary chatter during recall.
    * Semantic recall with relevance scores.

Prerequisites:
    * Ollama running locally with a model pulled, e.g.::

        ollama pull llama3.2

    * A Dakera server. The canonical self-hosted path is the public
      ``dakera-deploy`` docker-compose (server image + object store):
      https://github.com/dakera-ai/dakera-deploy
    * ``pip install "dakera" "ollama>=0.4"``

Run:
    OLLAMA_MODEL=llama3.2 python examples/ollama_memory_chat.py
"""

import os
import sys

try:
    import ollama
except ImportError:
    sys.exit("This example needs the Ollama client: pip install 'ollama>=0.4'")

from dakera import ChatMemorySession, DakeraClient

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
AGENT_ID = "ollama-demo-user"


def chat(session: ChatMemorySession, user_msg: str, top_k: int = 5) -> str:
    """Answer *user_msg* with an Ollama model grounded in Dakera memory."""
    # 1. Recall relevant long-term context for this agent.
    recalled = session.recall(user_msg, top_k=top_k)
    if recalled:
        context = "\n".join(f"- {m.content}" for m in recalled)
        system = f"Relevant context from earlier conversations:\n{context}"
    else:
        system = "You are a helpful assistant."

    # 2. Normal stateless Ollama call, now grounded in memory. Context is
    #    passed as a system *message* (Ollama /api/chat has no top-level
    #    "system" field — it takes a messages array).
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = response["message"]["content"]

    # 3. Persist the turn. Ordinary turns use the session default importance
    #    (0.6); ChatMemorySession tags each memory with its role automatically.
    session.store("user", user_msg)
    session.store("assistant", answer)
    return answer


def main() -> None:
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    session = ChatMemorySession.create(
        client, AGENT_ID, metadata={"app": "ollama-memory-chat"}
    )
    try:
        # Seed a durable preference at high importance so it reliably surfaces
        # on later, semantically related turns.
        session.store(
            "user",
            "My name is Sam and I always want answers in metric units.",
            importance=0.95,
        )

        for turn in [
            "What's a good outdoor temperature for a run?",
            "Remind me — what's my name and which units do I prefer?",
        ]:
            print(f"\n> {turn}")
            print(chat(session, turn))
    finally:
        # Closing summarizes the session; the stored turns remain recallable
        # by this agent in future sessions.
        session.close(summary="Ollama memory-chat demo session.")


if __name__ == "__main__":
    main()
