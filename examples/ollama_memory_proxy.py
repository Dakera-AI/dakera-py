#!/usr/bin/env python3
"""
Dakera Python SDK — a transparent persistent-memory proxy in front of Ollama.

This is a drop-in middleware: point any existing Ollama client at this proxy
(``http://localhost:8080``) instead of Ollama directly (``http://localhost:11434``)
and every ``/api/chat`` call transparently gains long-term memory. The proxy:

    1. recalls relevant context for the incoming conversation from Dakera,
    2. injects it as a system message and forwards the request to Ollama,
    3. stores the user turn and the assistant reply back into Dakera.

No changes to the calling application are required — it keeps speaking the
Ollama API. This mirrors the recipe requested in ollama/ollama#16987, with the
grounding corrected: Dakera listens on port 3000 (not 3300), self-hosting uses
the ``dakera-deploy`` compose (the server needs its object store), and recalled
context is injected as a system *message* rather than a non-existent top-level
``system`` field.

Prerequisites:
    * Ollama running locally (``ollama serve`` + e.g. ``ollama pull llama3.2``).
    * A Dakera server — self-host via https://github.com/dakera-ai/dakera-deploy
    * ``pip install "dakera[async]" fastapi uvicorn httpx``

Run:
    uvicorn examples.ollama_memory_proxy:app --port 8080
    # then, from any Ollama client:
    #   curl http://localhost:8080/api/chat -d '{
    #     "model": "llama3.2", "stream": false,
    #     "messages": [{"role": "user", "content": "Hi, I am Sam."}]}'
"""

import os
import sys

try:
    import httpx
    from fastapi import FastAPI, Request
except ImportError:
    sys.exit("This example needs: pip install fastapi uvicorn httpx 'dakera[async]'")

from dakera import AsyncChatMemorySession, AsyncDakeraClient

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
AGENT_ID = os.environ.get("DAKERA_AGENT_ID", "ollama-proxy-user")

app = FastAPI(title="Dakera → Ollama memory proxy")

_memory = AsyncDakeraClient(
    os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
    api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
)
_ollama = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=120.0)


def _last_user_message(messages: list[dict]) -> str:
    """Return the content of the most recent user message, if any."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


@app.post("/api/chat")
async def chat_with_memory(request: Request) -> dict:
    """Ollama-compatible /api/chat that transparently recalls and stores memory."""
    body = await request.json()
    messages: list[dict] = list(body.get("messages", []))
    user_msg = _last_user_message(messages)

    session = await AsyncChatMemorySession.create(_memory, AGENT_ID)
    try:
        # 1. Recall and inject relevant context as a leading system message.
        if user_msg:
            recalled = await session.recall(user_msg, top_k=5)
            if recalled:
                context = "\n".join(f"- {m.content}" for m in recalled)
                messages.insert(
                    0,
                    {"role": "system", "content": f"Relevant context:\n{context}"},
                )

        # 2. Forward to Ollama. Non-streaming so we can capture the full reply;
        #    a streaming variant would tee chunks and store on completion.
        body["messages"] = messages
        body["stream"] = False
        resp = await _ollama.post("/api/chat", json=body)
        data = resp.json()

        # 3. Store the exchange for future recall.
        if resp.status_code == 200 and user_msg:
            answer = data.get("message", {}).get("content", "")
            await session.store("user", user_msg)
            if answer:
                await session.store("assistant", answer)
        return data
    finally:
        await session.close()
