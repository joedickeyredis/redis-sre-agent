"""Live integration test for the Atlassian (Confluence) MCP server.

Validates that the Atlassian server configured in config.yaml propagates into
settings.mcp_servers and that the agent discovers the Confluence read tools when
the server is loaded through the production ToolManager path (the same entry
point the agent and API use).

The server is located by its URL (mcp.atlassian.com), so the tests do not care
what key it is configured under (e.g. `atlassian` or `atlassian_rovo_mcp`).

Two tests:
  - test_config_yaml_propagates_atlassian_server: deterministic, no network. Proves
    config.yaml -> settings.mcp_servers carries the right URL, transport, and the
    ${ATLASSIAN_MCP_BASIC_AUTH} Authorization header. Runs whenever the server is
    configured.
  - test_atlassian_mcp_lists_confluence_tools: LIVE. Loads the Atlassian server
    through ToolManager (which authenticates via the Basic API-token header,
    connects, and discovers tools) and asserts the Confluence read tools are
    exposed via the public manager.get_tools() surface. Additionally requires
    ATLASSIAN_MCP_BASIC_AUTH.

    Atlassian's Rovo MCP endpoint intermittently serves only its base
    getTeamworkGraph* toolset (cloud resources not yet resolved server-side)
    instead of the full ~45-tool surface. This varies across connections over
    minutes and re-listing the same session does not recover it, so the test
    retries with fresh connects; if every attempt returns only the base set it
    SKIPS (an upstream condition, not a regression) rather than failing. A total
    absence of Atlassian tools still fails hard, since that signals an auth or
    connection problem.

The live test drives ToolManager via explicit __aenter__/__aexit__ so the exit can
be time-boxed with asyncio.wait_for. The Atlassian streamable-HTTP client's aclose
can block (a known MCP SDK / anyio quirk), and cancelling it after a short timeout
keeps a wedged teardown from hanging the suite without abandoning connections on a
throwaway thread.

tests/conftest.py loads .env on import, so ATLASSIAN_MCP_BASIC_AUTH in .env is
picked up automatically. Both tests live under tests/integration/, so conftest
treats them as integration tests (the `integration` directory contributes the
`integration` keyword regardless of markers); they are excluded from `make test`
and only run with --run-api-tests:

    uv run pytest tests/integration/tools/mcp/test_atlassian_mcp_integration.py \
      --run-api-tests -v
"""

import asyncio
import contextlib
import os
from typing import Optional, Set, Tuple

import pytest

from redis_sre_agent.core.config import MCPServerConfig, settings


def _find_atlassian_server() -> Tuple[Optional[str], Optional[MCPServerConfig]]:
    """Locate the Atlassian MCP server by URL, regardless of its config key.

    Handles both MCPServerConfig instances and raw dicts (mcp_servers values can
    deserialize either way from YAML), mirroring ToolManager._load_mcp_providers.
    """
    for name, cfg in (settings.mcp_servers or {}).items():
        if isinstance(cfg, dict):
            try:
                cfg = MCPServerConfig.model_validate(cfg)
            except Exception:
                continue
        if "mcp.atlassian.com" in (getattr(cfg, "url", None) or ""):
            return name, cfg
    return None, None


_ATLASSIAN_NAME, _atlassian_config = _find_atlassian_server()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _atlassian_config is None,
        reason="No Atlassian (mcp.atlassian.com) server in config.yaml; skipping",
    ),
]

_requires_token = pytest.mark.skipif(
    not os.getenv("ATLASSIAN_MCP_BASIC_AUTH"),
    reason="ATLASSIAN_MCP_BASIC_AUTH not set; skipping live Atlassian MCP test",
)


# Confluence read tools. Their presence distinguishes the full
# Atlassian tool surface from Rovo's degraded base set (getTeamworkGraph* only), and
# they are the exact set the config's `tools` whitelist should expose to the agent.
_CONFLUENCE_READ_TOOLS = {
    "searchConfluenceUsingCql",
    "getConfluencePage",
    "searchAtlassian",
    "fetchAtlassian",
}


def test_config_yaml_propagates_atlassian_server():
    """config.yaml should propagate into settings.mcp_servers with the right values."""
    cfg = _atlassian_config

    # URL propagated and points at the modern streamable-HTTP endpoint.
    assert os.path.expandvars(cfg.url) == "https://mcp.atlassian.com/v1/mcp"

    # Transport defaults to / is set to streamable_http (what the SDK client uses).
    assert (cfg.transport or "streamable_http").lower() == "streamable_http"

    # Auth header propagated as a ${ATLASSIAN_MCP_BASIC_AUTH} template (not hardcoded).
    assert cfg.headers, "Atlassian server config is missing headers"
    auth = cfg.headers.get("Authorization")
    assert auth, "Atlassian server config is missing an Authorization header"
    assert "ATLASSIAN_MCP_BASIC_AUTH" in auth, (
        "Authorization header should reference the ${ATLASSIAN_MCP_BASIC_AUTH} env var, "
        f"got: {auth!r}"
    )

    # When the token is present, it expands to a non-empty Basic credential. This is
    # exactly the expansion the provider/pool perform via os.path.expandvars.
    if os.getenv("ATLASSIAN_MCP_BASIC_AUTH"):
        expanded = os.path.expandvars(auth)
        assert expanded.startswith("Basic "), f"Expected 'Basic <token>', got: {expanded!r}"
        assert len(expanded) > len("Basic "), "ATLASSIAN_MCP_BASIC_AUTH expanded to empty"

    # Read whitelist: the config narrows the agent-callable surface to exactly the
    # four Confluence read tools, each surfaced as a read-only knowledge tool. This
    # is what keeps every Atlassian write tool (createConfluencePage, createJiraIssue,
    # ...) out of the agent's reach, so the assertion guards that safety property.
    assert cfg.tools is not None, (
        "Atlassian server config should define a `tools` whitelist so only the "
        "Confluence read tools are exposed to the agent (no write tools)"
    )
    assert set(cfg.tools) == _CONFLUENCE_READ_TOOLS, (
        f"Whitelist should be exactly {sorted(_CONFLUENCE_READ_TOOLS)}, "
        f"got {sorted(cfg.tools)}"
    )
    for tool_name, tool_cfg in cfg.tools.items():
        capability = getattr(tool_cfg.capability, "value", tool_cfg.capability)
        action_kind = getattr(tool_cfg.action_kind, "value", tool_cfg.action_kind)
        assert capability == "knowledge", (
            f"{tool_name} should be capability=knowledge, got {capability!r}"
        )
        assert action_kind == "read", (
            f"{tool_name} should be action_kind=read (auto-allow, no approval prompt), "
            f"got {action_kind!r}"
        )


async def _discover_tools_via_tool_manager(
    name: str, cfg: MCPServerConfig, teardown_timeout: float = 10.0
) -> Set[str]:
    """Discover tool names by loading the server through the production ToolManager.

    This exercises the same public path the agent uses: ToolManager loads the MCP
    provider from settings.mcp_servers, connects, and exposes tools via get_tools().
    We isolate settings.mcp_servers to just this server so ToolManager does not dial
    other configured MCP servers, and reset the process-wide connection-pool
    singleton so the run starts clean.

    Teardown is time-boxed: the Atlassian client's aclose can block, so we cancel it
    after teardown_timeout rather than let a wedged teardown hang the suite.
    """
    import redis_sre_agent.core.config as config_module
    from redis_sre_agent.tools.manager import ToolManager
    from redis_sre_agent.tools.mcp.pool import MCPConnectionPool

    original_mcp_servers = config_module.settings.mcp_servers
    config_module.settings.mcp_servers = {name: cfg}
    MCPConnectionPool.reset_instance()

    manager = ToolManager()
    await manager.__aenter__()
    try:
        return {tool.name for tool in manager.get_tools()}
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                manager.__aexit__(None, None, None), timeout=teardown_timeout
            )
        config_module.settings.mcp_servers = original_mcp_servers
        MCPConnectionPool.reset_instance()


def _atlassian_tool_names(tool_names: Set[str], server_name: str) -> Set[str]:
    """Filter to tools registered by the Atlassian MCP provider."""
    return {name for name in tool_names if name.startswith(f"mcp_{server_name}_")}


def _has_confluence_tools(atlassian_tools: Set[str]) -> bool:
    """True if the full Confluence read surface is present (not just the base set)."""
    return any(
        name.endswith(f"_{op}") for op in _CONFLUENCE_READ_TOOLS for name in atlassian_tools
    )


async def _discover_atlassian_tools_with_retry(
    name: str,
    cfg: MCPServerConfig,
    *,
    attempts: int = 3,
    backoff: float = 2.0,
) -> Set[str]:
    """Discover Atlassian tools, retrying fresh connects if Rovo is degraded.

    Atlassian's Rovo MCP endpoint intermittently serves only its base
    getTeamworkGraph* toolset (cloud resources not yet resolved server-side)
    before recovering to the full ~45-tool surface. This was verified to vary
    across connections over minutes, and re-listing on the same session does NOT
    recover it -- so each retry is a fresh ToolManager connect. Returns the best
    toolset observed across attempts; the caller decides assert vs skip.

    Discovery strips any read whitelist (`tools`) from the config so we observe
    the server's full advertised surface. The whitelist only governs which tools
    the agent may call -- not what the server exposes -- and keeping it here would
    filter out the getTeamworkGraph* base set, making a degraded upstream look
    like a total auth/connection failure instead of a skippable condition.
    """
    discovery_cfg = cfg.model_copy(update={"tools": None}) if hasattr(cfg, "model_copy") else cfg

    tool_names: Set[str] = set()
    for attempt in range(1, attempts + 1):
        tool_names = await _discover_tools_via_tool_manager(name, discovery_cfg)
        if _has_confluence_tools(_atlassian_tool_names(tool_names, name)):
            return tool_names
        if attempt < attempts:
            await asyncio.sleep(backoff)
    return tool_names


@_requires_token
@pytest.mark.asyncio
async def test_atlassian_mcp_lists_confluence_tools():
    """The configured Atlassian MCP server should expose the Confluence read tools."""
    tool_names = await _discover_atlassian_tools_with_retry(_ATLASSIAN_NAME, _atlassian_config)

    # MCP tools are namespaced as mcp_<server>_<hash>_<operation>. Match on the
    # trailing "_<operation>" so assertions are independent of the instance hash.
    atlassian_tools = _atlassian_tool_names(tool_names, _ATLASSIAN_NAME)

    # Hard failure: no Atlassian tools at all means an auth/connection problem,
    # not a degraded surface.
    assert atlassian_tools, (
        "No Atlassian MCP tools were discovered. This indicates an auth or "
        "connection failure (check ATLASSIAN_MCP_BASIC_AUTH and network), not a "
        "degraded upstream toolset."
    )

    missing = {
        op
        for op in _CONFLUENCE_READ_TOOLS
        if not any(name.endswith(f"_{op}") for name in atlassian_tools)
    }

    # Environmental degradation, not a regression: Rovo intermittently serves only
    # its base getTeamworkGraph* toolset (verified to vary across connections over
    # minutes; re-listing the same session does not recover it). Skip rather than
    # fail so this test doesn't flake red on an upstream condition we don't control.
    base_only = bool(atlassian_tools) and all("TeamworkGraph" in name for name in atlassian_tools)
    if missing and base_only:
        pytest.skip(
            "Atlassian Rovo MCP returned only its base getTeamworkGraph* toolset "
            f"({sorted(atlassian_tools)}) across all retries; the Confluence tools "
            "were not provisioned server-side this run. This is an intermittent "
            "upstream condition, not a code regression -- re-run to retry."
        )

    # The Confluence read tools must be present.
    assert not missing, f"Missing expected Confluence read tools: {sorted(missing)}"
