"""Shared fixtures for CLI unit tests."""

import pytest

from redis_sre_agent.tools.mcp.pool import MCPConnectionPool


@pytest.fixture(autouse=True)
def stub_mcp_pool(monkeypatch):
    """Keep CLI unit tests hermetic by neutralizing the MCP connection pool.

    The ``query`` CLI command starts the singleton ``MCPConnectionPool``, which
    would otherwise dial whatever MCP server is configured in a developer's local
    ``config.yaml`` (e.g. the live Atlassian server) during a unit test - slow,
    network-dependent, and unrelated to the CLI logic under test. Reset the
    singleton and no-op its lifecycle so routing/behavior is exercised in isolation.
    """
    MCPConnectionPool.reset_instance()

    async def _noop_start(self):
        self._started = True
        return {}

    async def _noop_shutdown(self, force: bool = False):
        return None

    monkeypatch.setattr(MCPConnectionPool, "start", _noop_start)
    monkeypatch.setattr(MCPConnectionPool, "shutdown", _noop_shutdown)

    yield

    MCPConnectionPool.reset_instance()
