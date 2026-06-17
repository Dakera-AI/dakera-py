"""
Session management helpers for Dakera SDK.

Provides a high-level ``ChatMemorySession`` (sync) and
``AsyncChatMemorySession`` (async) that wrap the low-level session/memory
API into the three-step pattern used by the playground LLM chat comparison
feature:

1. Create a session (bound to an agent)
2. Store conversation turns in the session
3. Recall relevant context before generating the next response

Example::

    from dakera import DakeraClient
    from dakera.session import ChatMemorySession

    client = DakeraClient("http://localhost:3000", api_key="...")

    # Compare responses with / without Dakera memory
    with ChatMemorySession.create(client, agent_id="chat-agent") as session:
        session.store("user", "My name is Alice and I like Python.")
        context = session.recall("user preferences")
        # Pass `context` to your LLM call — or omit it for the baseline arm
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dakera.async_client import AsyncDakeraClient
    from dakera.client import DakeraClient
    from dakera.models import RecalledMemory


class ChatMemorySession:
    """High-level session helper for LLM chat comparison patterns.

    Groups conversation turns under a single Dakera session so that:

    * Every stored message is associated with ``session_id`` for scoped
      retrieval.
    * ``recall`` queries the agent's full memory — not just this session —
      so prior conversations inform the current exchange.
    * The context manager automatically ends the session on exit, even
      when an exception is raised.

    Prefer :meth:`create` over the constructor to avoid managing the raw
    session dict yourself.
    """

    def __init__(
        self,
        client: DakeraClient,
        agent_id: str,
        session_id: str,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        client: DakeraClient,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMemorySession:
        """Create a new Dakera session and return a ``ChatMemorySession``.

        Args:
            client: Configured :class:`~dakera.DakeraClient` instance.
            agent_id: Identifier for the agent whose memory to use.
            metadata: Optional metadata attached to the session record.

        Returns:
            A ``ChatMemorySession`` bound to the new session.

        Example::

            session = ChatMemorySession.create(client, "my-agent")
            try:
                session.store("user", "Hello, remember me?")
                context = session.recall("who am I")
            finally:
                session.close()
        """
        raw = client.start_session(agent_id, metadata=metadata)
        session_id = raw["id"] if isinstance(raw, dict) else raw.id
        return cls(client, agent_id, session_id)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store(
        self,
        role: str,
        content: str,
        importance: float = 0.6,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a conversation turn in the session.

        Args:
            role: Speaker role — e.g. ``"user"`` or ``"assistant"``.
            content: The message text to persist.
            importance: Importance score (0.0–1.0). Defaults to 0.6, which
                keeps conversation turns above the decay floor without
                crowding higher-importance procedural memories.
            tags: Optional tags; ``role`` is always appended automatically.

        Returns:
            The stored memory dict from the server.
        """
        effective_tags = list(tags or [])
        if role not in effective_tags:
            effective_tags.append(role)

        return self._client.store_memory(
            agent_id=self._agent_id,
            content=content,
            memory_type="episodic",
            importance=importance,
            session_id=self._session_id,
            tags=effective_tags,
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RecalledMemory]:
        """Recall memories relevant to *query* for this agent.

        Searches the agent's full memory (not just the current session) so
        that context from prior conversations is surfaced when relevant.

        Args:
            query: Natural-language query to find relevant memories.
            top_k: Maximum number of memories to return (default: 5).

        Returns:
            List of :class:`~dakera.RecalledMemory` objects ordered by
            relevance.
        """
        response = self._client.recall(
            agent_id=self._agent_id,
            query=query,
            top_k=top_k,
        )
        return response.memories

    def close(self, summary: str | None = None) -> dict[str, Any]:
        """End the Dakera session.

        Args:
            summary: Optional human-readable summary stored alongside the
                session record.

        Returns:
            Server response dict.
        """
        return self._client.end_session(self._session_id, summary=summary)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The underlying Dakera session ID."""
        return self._session_id

    @property
    def agent_id(self) -> str:
        """The agent ID this session is bound to."""
        return self._agent_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ChatMemorySession:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        with contextlib.suppress(Exception):
            self.close()


class AsyncChatMemorySession:
    """Async high-level session helper for LLM chat comparison patterns.

    Drop-in async counterpart to :class:`ChatMemorySession`.  Requires an
    :class:`~dakera.AsyncDakeraClient` instance.

    * Every stored message is associated with ``session_id`` for scoped
      retrieval.
    * :meth:`recall` queries the agent's full memory — not just this session —
      so prior conversations inform the current exchange.
    * The async context manager ends the session on exit automatically.

    Example::

        from dakera import AsyncDakeraClient
        from dakera.session import AsyncChatMemorySession

        async with await AsyncChatMemorySession.create(
            client, agent_id="chat-agent"
        ) as session:
            await session.store("user", "My name is Alice and I like Python.")
            context = await session.recall("user preferences")
            # pass context to your LLM — or skip for the baseline arm
    """

    def __init__(
        self,
        client: AsyncDakeraClient,
        agent_id: str,
        session_id: str,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        client: AsyncDakeraClient,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncChatMemorySession:
        """Create a new Dakera session and return an ``AsyncChatMemorySession``.

        Args:
            client: Configured :class:`~dakera.AsyncDakeraClient` instance.
            agent_id: Identifier for the agent whose memory to use.
            metadata: Optional metadata attached to the session record.

        Returns:
            An ``AsyncChatMemorySession`` bound to the new session.
        """
        raw = await client.start_session(agent_id, metadata=metadata)
        session_id = raw["id"] if isinstance(raw, dict) else raw.id
        return cls(client, agent_id, session_id)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def store(
        self,
        role: str,
        content: str,
        importance: float = 0.6,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a conversation turn in the session.

        Args:
            role: Speaker role — e.g. ``"user"`` or ``"assistant"``.
            content: The message text to persist.
            importance: Importance score (0.0–1.0). Defaults to 0.6.
            tags: Optional tags; ``role`` is always appended automatically.

        Returns:
            The stored memory dict from the server.
        """
        effective_tags = list(tags or [])
        if role not in effective_tags:
            effective_tags.append(role)
        return await self._client.store_memory(
            agent_id=self._agent_id,
            content=content,
            memory_type="episodic",
            importance=importance,
            session_id=self._session_id,
            tags=effective_tags,
        )

    async def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RecalledMemory]:
        """Recall memories relevant to *query* for this agent.

        Searches the agent's full memory (not just the current session) so
        that context from prior conversations is surfaced when relevant.

        Args:
            query: Natural-language query to find relevant memories.
            top_k: Maximum number of memories to return (default: 5).

        Returns:
            List of :class:`~dakera.RecalledMemory` objects ordered by
            relevance.
        """
        response = await self._client.recall(
            agent_id=self._agent_id,
            query=query,
            top_k=top_k,
        )
        return response.memories

    async def close(self, summary: str | None = None) -> dict[str, Any]:
        """End the Dakera session.

        Args:
            summary: Optional human-readable summary stored alongside the
                session record.

        Returns:
            Server response dict.
        """
        return await self._client.end_session(self._session_id, summary=summary)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The underlying Dakera session ID."""
        return self._session_id

    @property
    def agent_id(self) -> str:
        """The agent ID this session is bound to."""
        return self._agent_id

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncChatMemorySession:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        with contextlib.suppress(Exception):
            await self.close()
