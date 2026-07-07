"""Resilience tests for MCP provider loading in the ToolManager.

A failure connecting to a remote MCP server (e.g. Confluence being slow,
rate-limited, or unreachable) must not fail the whole turn. The manager should
degrade gracefully and still expose the remaining (e.g. built-in RAG) tools.
"""

import asyncio

import pytest

from redis_sre_agent.core import config as config_module
from redis_sre_agent.tools.manager import ToolManager
from redis_sre_agent.tools.mcp.provider import MCPToolProvider
from redis_sre_agent.tools.protocols import ToolCapability


@pytest.fixture
def single_fake_remote_server(monkeypatch):
    """Point the manager at one fake remote MCP server (no real config/network)."""
    monkeypatch.setattr(
        config_module.settings,
        "mcp_servers",
        {
            "flaky_remote": {
                "url": "https://mcp.example.invalid/v1/mcp",
                "transport": "streamable_http",
            }
        },
        raising=False,
    )


@pytest.mark.asyncio
async def test_mcp_cancelled_error_degrades_to_builtin_tools(
    single_fake_remote_server, monkeypatch
):
    """A connect failure surfacing as CancelledError must be treated as the
    server being unavailable, not as a turn-ending cancellation."""

    async def _abort(self):
        # Mirrors how anyio's streamable-HTTP task group aborts a failed remote
        # connection: a CancelledError (a BaseException, not an Exception).
        raise asyncio.CancelledError("simulated remote connection abort")

    monkeypatch.setattr(MCPToolProvider, "_connect", _abort)

    async with ToolManager(redis_instance=None) as mgr:
        knowledge_tools = mgr.get_tools_for_capability(ToolCapability.KNOWLEDGE)
        assert knowledge_tools, "Expected built-in knowledge tools despite MCP failure"
        # The flaky remote contributed no tools.
        assert all(
            not t.name.startswith("mcp_flaky_remote") for t in knowledge_tools
        ), "Flaky remote MCP server should not have registered any tools"


@pytest.mark.asyncio
async def test_mcp_generic_exception_degrades_to_builtin_tools(
    single_fake_remote_server, monkeypatch
):
    """A plain Exception from connect is likewise contained per-provider."""

    async def _boom(self):
        raise RuntimeError("simulated remote connection error")

    monkeypatch.setattr(MCPToolProvider, "_connect", _boom)

    async with ToolManager(redis_instance=None) as mgr:
        knowledge_tools = mgr.get_tools_for_capability(ToolCapability.KNOWLEDGE)
        assert knowledge_tools, "Expected built-in knowledge tools despite MCP failure"
