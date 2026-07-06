"""Unit tests for MCP tool provider."""

import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types as mcp_types
from pydantic import ValidationError

from redis_sre_agent.core.config import MCPResultShaping, MCPServerConfig, MCPToolConfig
from redis_sre_agent.evaluation.fake_mcp import build_fixture_mcp_runtime
from redis_sre_agent.evaluation.injection import eval_injection_scope
from redis_sre_agent.evaluation.scenarios import EvalScenario
from redis_sre_agent.tools.mcp.provider import MCPToolProvider
from redis_sre_agent.tools.models import ToolActionKind, ToolCapability


class TestMCPToolProvider:
    """Test MCPToolProvider functionality."""

    def test_provider_name(self):
        """Test that provider name is based on server name."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="memory", server_config=config)
        assert provider.provider_name == "mcp_memory"

    def test_provider_name_with_special_chars(self):
        """Test provider name with various server names."""
        config = MCPServerConfig(command="test")

        provider = MCPToolProvider(server_name="my_server", server_config=config)
        assert provider.provider_name == "mcp_my_server"

        provider = MCPToolProvider(server_name="test123", server_config=config)
        assert provider.provider_name == "mcp_test123"

    def test_should_include_tool_no_filter(self):
        """Test that all tools are included when no filter is specified."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._should_include_tool("any_tool") is True
        assert provider._should_include_tool("another_tool") is True

    def test_should_include_tool_with_filter(self):
        """Test that only specified tools are included when filter is set."""
        config = MCPServerConfig(
            command="test",
            tools={
                "allowed_tool": MCPToolConfig(),
                "another_allowed": MCPToolConfig(),
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._should_include_tool("allowed_tool") is True
        assert provider._should_include_tool("another_allowed") is True
        assert provider._should_include_tool("not_allowed") is False

    def test_get_capability_default(self):
        """Test that default capability is UTILITIES."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._get_capability("any_tool") == ToolCapability.UTILITIES

    def test_get_capability_with_override(self):
        """Test that capability override is respected."""
        config = MCPServerConfig(
            command="test",
            tools={
                "search_tool": MCPToolConfig(capability=ToolCapability.LOGS),
                "metrics_tool": MCPToolConfig(capability=ToolCapability.METRICS),
                "no_override": MCPToolConfig(),
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._get_capability("search_tool") == ToolCapability.LOGS
        assert provider._get_capability("metrics_tool") == ToolCapability.METRICS
        assert provider._get_capability("no_override") == ToolCapability.UTILITIES
        assert provider._get_capability("unknown_tool") == ToolCapability.UTILITIES

    def test_get_description_default(self):
        """Test that MCP description is used by default."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        mcp_desc = "Original MCP description"
        assert provider._get_description("any_tool", mcp_desc) == mcp_desc

    def test_get_description_with_override(self):
        """Test that description override is respected."""
        config = MCPServerConfig(
            command="test",
            tools={
                "custom_tool": MCPToolConfig(description="Custom description"),
                "no_override": MCPToolConfig(),
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._get_description("custom_tool", "MCP desc") == "Custom description"
        assert provider._get_description("no_override", "MCP desc") == "MCP desc"
        assert provider._get_description("unknown", "MCP desc") == "MCP desc"

    def test_get_description_with_original_template(self):
        """Test that {original} placeholder is replaced with MCP description."""
        config = MCPServerConfig(
            command="test",
            tools={
                "templated_tool": MCPToolConfig(description="Custom context. {original}"),
                "prepended": MCPToolConfig(
                    description="WARNING: Use carefully. {original} See docs for details."
                ),
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        # Template should replace {original} with the MCP description
        assert (
            provider._get_description("templated_tool", "Original MCP description")
            == "Custom context. Original MCP description"
        )
        assert (
            provider._get_description("prepended", "Search for files.")
            == "WARNING: Use carefully. Search for files. See docs for details."
        )

    def test_get_action_kind_with_override(self):
        """Test that action kind override is respected."""
        config = MCPServerConfig(
            command="test",
            tools={
                "query_tool": MCPToolConfig(action_kind=ToolActionKind.READ),
                "mutate_tool": MCPToolConfig(action_kind=ToolActionKind.WRITE),
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert (
            provider._get_action_kind("query_tool", "Read current state.", ToolCapability.UTILITIES)
            == ToolActionKind.READ
        )
        assert (
            provider._get_action_kind(
                "mutate_tool", "Write current state.", ToolCapability.UTILITIES
            )
            == ToolActionKind.WRITE
        )

    def test_get_action_kind_infers_from_mcp_tool_name_and_description(self):
        """Test that MCP tools infer action kinds when no override is configured."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="github", server_config=config)

        assert (
            provider._get_action_kind(
                "_create_branch",
                "Create a new branch in the given repository.",
                ToolCapability.REPOS,
            )
            == ToolActionKind.WRITE
        )
        assert (
            provider._get_action_kind(
                "_search_repositories",
                "Search for a repository by name or description.",
                ToolCapability.REPOS,
            )
            == ToolActionKind.READ
        )
        assert (
            provider._get_action_kind(
                "custom_operation",
                "Runs a custom operation for the current selection.",
                ToolCapability.REPOS,
            )
            == ToolActionKind.UNKNOWN
        )

    def test_get_action_kind_keeps_leading_verb_for_resolved_mcp_operation_names(self):
        """Test that already-resolved MCP operation names still infer write intent."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="github", server_config=config)

        assert (
            provider._get_action_kind(
                "_update_repository",
                "Operate on repository settings for the selected repo.",
                ToolCapability.UTILITIES,
            )
            == ToolActionKind.WRITE
        )

    def test_get_action_kind_keeps_read_verb_for_multi_part_mcp_operation_names(self):
        """Test that resolved MCP operation names like list_pull_requests stay intact."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="github", server_config=config)

        assert (
            provider._get_action_kind(
                "list_pull_requests",
                "Operate on the selected pull request collection.",
                ToolCapability.REPOS,
            )
            == ToolActionKind.READ
        )

    def test_get_tool_config(self):
        """Test getting tool config."""
        tool_config = MCPToolConfig(
            capability=ToolCapability.LOGS,
            description="Test description",
        )
        config = MCPServerConfig(
            command="test",
            tools={"my_tool": tool_config},
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._get_tool_config("my_tool") == tool_config
        assert provider._get_tool_config("unknown") is None

    def test_get_tool_config_no_tools_defined(self):
        """Test getting tool config when no tools are defined."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._get_tool_config("any_tool") is None

    def test_apply_arg_defaults_noop_when_not_configured(self):
        """A tool with no config entry returns args unchanged."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._apply_arg_defaults("any_tool", {"query": "x"}) == {"query": "x"}

    def test_apply_arg_defaults_noop_when_tool_has_no_arg_defaults(self):
        """A configured tool without arg_defaults returns args unchanged."""
        config = MCPServerConfig(command="test", tools={"searchAtlassian": MCPToolConfig()})
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._apply_arg_defaults("searchAtlassian", {"query": "x"}) == {"query": "x"}

    def test_apply_arg_defaults_injects_static_value_when_missing(self):
        """A static default is injected when the model omits it."""
        config = MCPServerConfig(
            command="test",
            tools={"searchAtlassian": MCPToolConfig(arg_defaults={"cloudId": "abc"})},
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        assert provider._apply_arg_defaults("searchAtlassian", {"query": "x"}) == {
            "query": "x",
            "cloudId": "abc",
        }

    def test_apply_arg_defaults_expands_env_var(self):
        """A ${VAR} default is expanded from the environment."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_CLOUD_ID}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {"TEST_ARG_DEFAULT_CLOUD_ID": "cloud-123"}, clear=False):
            result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert result["cloudId"] == "cloud-123"

    def test_apply_arg_defaults_pins_over_model_value(self):
        """A resolvable default overrides a model-supplied value (pinned)."""
        config = MCPServerConfig(
            command="test",
            tools={"searchAtlassian": MCPToolConfig(arg_defaults={"cloudId": "config-value"})},
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        result = provider._apply_arg_defaults(
            "searchAtlassian", {"cloudId": "model-guess", "query": "x"}
        )

        assert result["cloudId"] == "config-value"
        assert result["query"] == "x"

    def test_apply_arg_defaults_skips_unresolved_var_preserves_model_value(self):
        """An unresolved ${VAR} is skipped so the model-supplied value survives."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_MISSING}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("TEST_ARG_DEFAULT_MISSING", None)
            result = provider._apply_arg_defaults("searchAtlassian", {"cloudId": "model-guess"})

        assert result["cloudId"] == "model-guess"

    def test_apply_arg_defaults_skips_unresolved_var_leaves_key_absent(self):
        """An unresolved ${VAR} never injects a literal placeholder."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_MISSING}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("TEST_ARG_DEFAULT_MISSING", None)
            result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert result == {"query": "x"}
        assert "cloudId" not in result

    def test_apply_arg_defaults_skips_empty_string_value(self):
        """A default that resolves to an empty string is skipped."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_EMPTY}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {"TEST_ARG_DEFAULT_EMPTY": ""}, clear=False):
            result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert "cloudId" not in result

    def test_apply_arg_defaults_skips_embedded_unresolved_placeholder(self):
        """An embedded unresolved ${VAR} (not whole-string) is also skipped."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"path": "a-${TEST_ARG_DEFAULT_MISSING}-b"}
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("TEST_ARG_DEFAULT_MISSING", None)
            result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert "path" not in result

    def test_apply_arg_defaults_passes_through_non_string_values(self):
        """Non-string defaults are injected verbatim (no expansion attempted)."""
        config = MCPServerConfig(
            command="test",
            tools={"searchAtlassian": MCPToolConfig(arg_defaults={"limit": 10, "flag": True})},
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert result["limit"] == 10
        assert result["flag"] is True

    def test_apply_arg_defaults_handles_multiple_keys_independently(self):
        """A resolvable key is injected while an unresolved one is skipped."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={
                        "cloudId": "${TEST_ARG_DEFAULT_CLOUD_ID}",
                        "region": "${TEST_ARG_DEFAULT_MISSING}",
                    }
                )
            },
        )
        provider = MCPToolProvider(server_name="test", server_config=config)

        with patch.dict("os.environ", {"TEST_ARG_DEFAULT_CLOUD_ID": "cloud-123"}, clear=False):
            os.environ.pop("TEST_ARG_DEFAULT_MISSING", None)
            result = provider._apply_arg_defaults("searchAtlassian", {"query": "x"})

        assert result["cloudId"] == "cloud-123"
        assert "region" not in result

    def test_apply_arg_defaults_does_not_mutate_input_args(self):
        """The input args dict is not mutated; a new dict is returned."""
        config = MCPServerConfig(
            command="test",
            tools={"searchAtlassian": MCPToolConfig(arg_defaults={"cloudId": "abc"})},
        )
        provider = MCPToolProvider(server_name="test", server_config=config)
        original = {"query": "x"}

        result = provider._apply_arg_defaults("searchAtlassian", original)

        assert original == {"query": "x"}
        assert result is not original
        assert result["cloudId"] == "abc"


class TestMCPToolProviderAsync:
    """Test async functionality of MCPToolProvider."""

    @pytest.mark.asyncio
    async def test_tools_returns_empty_list_without_connection(self):
        """Test that tools() returns empty list when not connected."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        # Without connecting, tools should be empty
        tools = provider.tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_create_tool_schemas_empty_without_connection(self):
        """Test that create_tool_schemas returns empty when not connected."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        # Without connecting, schemas should be empty
        schemas = provider.create_tool_schemas()
        assert schemas == []

    @pytest.mark.asyncio
    async def test_get_input_schemas_returns_raw_schema_by_operation_name(self):
        """Test that raw MCP schemas stay keyed by original tool names."""
        config = MCPServerConfig(
            command="test",
            tools={"file_write": MCPToolConfig()},
        )
        provider = MCPToolProvider(server_name="afs_gateway", server_config=config)
        provider._mcp_tools = [
            SimpleNamespace(
                name="file_write",
                description="Write a file",
                inputSchema={
                    "title": "file_writeArguments",
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "title": "Path"},
                        "content": {"type": "string", "title": "Content"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            )
        ]

        schemas = provider.get_input_schemas()

        assert schemas == {
            "file_write": {
                "title": "file_writeArguments",
                "type": "object",
                "properties": {
                    "path": {"type": "string", "title": "Path"},
                    "content": {"type": "string", "title": "Content"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            }
        }

    @pytest.mark.asyncio
    async def test_get_input_schemas_coerces_model_like_input_schema(self):
        """Test that MCP model objects are serialized into plain dict schemas."""
        config = MCPServerConfig(
            command="test",
            tools={"analyzer_list_accounts": MCPToolConfig()},
        )
        provider = MCPToolProvider(server_name="re_analyzer", server_config=config)

        class _SchemaModel:
            def model_dump(self, mode: str = "json") -> dict[str, object]:
                assert mode == "json"
                return {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                }

        provider._mcp_tools = [
            SimpleNamespace(
                name="analyzer_list_accounts",
                description="List accounts",
                inputSchema=_SchemaModel(),
            )
        ]

        schemas = provider.get_input_schemas()

        assert schemas == {
            "analyzer_list_accounts": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            }
        }

    @pytest.mark.asyncio
    async def test_call_mcp_tool_not_connected(self):
        """Test that _call_mcp_tool returns error when not connected."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        result = await provider._call_mcp_tool("some_tool", {"arg": "value"})
        assert result["status"] == "error"
        assert "not connected" in result["error"]

    @pytest.mark.asyncio
    async def test_connect_uses_pool_when_available(self):
        """Test that _connect uses pooled connection when available."""
        from unittest.mock import MagicMock

        from redis_sre_agent.tools.mcp.pool import MCPConnectionPool, PooledConnection

        # Reset and set up the pool
        MCPConnectionPool.reset_instance()
        pool = MCPConnectionPool.get_instance()

        # Create a mock pooled connection
        mock_session = MagicMock()
        mock_tools = [MagicMock(name="tool1")]
        pool._connections["test-server"] = PooledConnection(
            server_name="test-server",
            session=mock_session,
            tools=mock_tools,
            exit_stack=MagicMock(),
        )
        pool._started = True

        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test-server", server_config=config, use_pool=True)

        await provider._connect()

        assert provider._session is mock_session
        assert provider._using_pooled_connection is True

        # Cleanup
        MCPConnectionPool.reset_instance()

    @pytest.mark.asyncio
    async def test_disconnect_does_not_close_pooled_connection(self):
        """Test that _disconnect doesn't close pooled connections."""
        from unittest.mock import MagicMock

        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="test", server_config=config)

        # Simulate using a pooled connection
        provider._session = MagicMock()
        provider._using_pooled_connection = True

        await provider._disconnect()

        # Session should be cleared but not closed
        assert provider._session is None
        assert provider._using_pooled_connection is False

    @pytest.mark.asyncio
    async def test_connect_uses_eval_fake_mcp_runtime_before_network(self):
        """Test that eval fake MCP catalogs bypass network transports."""
        scenario = EvalScenario.model_validate(
            {
                "id": "fake-mcp-provider",
                "name": "Fake MCP provider",
                "provenance": {
                    "source_kind": "synthetic",
                    "source_pack": "fixture-pack",
                    "source_pack_version": "2026-04-14",
                    "golden": {"expectation_basis": "human_authored"},
                },
                "execution": {
                    "lane": "full_turn",
                    "query": "Check memory pressure.",
                },
                "tools": {
                    "mcp_servers": {
                        "metrics_eval": {
                            "capability": "metrics",
                            "tools": {
                                "query_metrics": {
                                    "description": "Query fixture metrics.",
                                    "input_schema": {
                                        "properties": {
                                            "query": {"type": "string"},
                                        },
                                        "required": ["query"],
                                    },
                                    "result": {
                                        "series": "memory_pressure",
                                        "value": 91,
                                    },
                                }
                            },
                        }
                    }
                },
            }
        )
        runtime = build_fixture_mcp_runtime(scenario)
        assert runtime is not None

        provider = MCPToolProvider(
            server_name="metrics_eval",
            server_config=runtime.get_server_configs()["metrics_eval"],
            use_pool=False,
        )

        with (
            eval_injection_scope(
                mcp_servers=runtime.get_server_configs(),
                mcp_runtime=runtime,
            ),
            patch(
                "redis_sre_agent.tools.mcp.provider.streamablehttp_client",
                side_effect=AssertionError("network transport should not run"),
            ),
        ):
            await provider._connect()
            tools = provider.tools()
            result = await provider._call_mcp_tool(
                "query_metrics",
                {"query": "memory pressure"},
            )

        assert provider._using_eval_runtime is True
        assert len(tools) == 1
        assert tools[0].definition.name.endswith("_query_metrics")
        assert tools[0].definition.capability is ToolCapability.METRICS
        assert tools[0].metadata.action_kind is ToolActionKind.READ
        assert result["status"] == "success"
        assert result["data"]["series"] == "memory_pressure"

    @pytest.mark.asyncio
    async def test_connect_expands_environment_variables_in_url_and_headers(self):
        """Configured MCP URLs should expand ${VAR} placeholders before transport setup."""
        config = MCPServerConfig(
            url="${TEST_MCP_URL}",
            transport="streamable_http",
            headers={"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
        )
        provider = MCPToolProvider(server_name="afs_gateway", server_config=config, use_pool=False)
        seen: dict[str, object] = {}

        @asynccontextmanager
        async def _fake_streamablehttp_client(url: str, headers: dict[str, str] | None = None):
            seen["url"] = url
            seen["headers"] = headers
            yield ("read-stream", "write-stream", lambda: None)

        class _FakeSession:
            async def __aenter__(self) -> "_FakeSession":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def initialize(self) -> None:
                return None

            async def list_tools(self) -> SimpleNamespace:
                return SimpleNamespace(tools=[])

        with (
            patch.dict(
                "os.environ",
                {
                    "TEST_MCP_URL": "http://afs-gateway.example/mcp",
                    "TEST_MCP_TOKEN": "secret-token",
                },
                clear=False,
            ),
            patch(
                "redis_sre_agent.tools.mcp.provider.streamablehttp_client",
                _fake_streamablehttp_client,
            ),
            patch(
                "redis_sre_agent.tools.mcp.provider.ClientSession",
                side_effect=lambda read_stream, write_stream: _FakeSession(),
            ),
        ):
            await provider._connect()

        assert seen == {
            "url": "http://afs-gateway.example/mcp",
            "headers": {"Authorization": "Bearer secret-token"},
        }

    def test_create_tool_schemas_strips_resolved_arg_default_from_schema(self):
        """A resolvable arg_default key is stripped from the advertised schema.

        The model must not see a pinned deployment value (e.g. cloudId); otherwise it
        stalls asking the user for a value it cannot know. The value is injected at call
        time instead (see test_tool_invoke_applies_arg_defaults).
        """
        config = MCPServerConfig(
            command="test",
            tools={
                "searchConfluenceUsingCql": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_CLOUD_ID}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        provider._mcp_tools = [
            SimpleNamespace(
                name="searchConfluenceUsingCql",
                description="Search Confluence via CQL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cloudId": {"type": "string"},
                        "cql": {"type": "string"},
                    },
                    "required": ["cloudId", "cql"],
                },
            )
        ]

        with patch.dict("os.environ", {"TEST_ARG_DEFAULT_CLOUD_ID": "cloud-123"}, clear=False):
            schemas = provider.create_tool_schemas()

        assert len(schemas) == 1
        params = schemas[0].parameters
        assert "cloudId" not in params["properties"]
        assert "cloudId" not in params["required"]
        # Non-pinned args are untouched.
        assert "cql" in params["properties"]
        assert "cql" in params["required"]

    def test_create_tool_schemas_keeps_unresolved_arg_default_in_schema(self):
        """An unresolved ${VAR} arg_default is NOT stripped, so the model can supply it."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchConfluenceUsingCql": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_MISSING}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        provider._mcp_tools = [
            SimpleNamespace(
                name="searchConfluenceUsingCql",
                description="Search Confluence via CQL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cloudId": {"type": "string"},
                        "cql": {"type": "string"},
                    },
                    "required": ["cloudId", "cql"],
                },
            )
        ]

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("TEST_ARG_DEFAULT_MISSING", None)
            schemas = provider.create_tool_schemas()

        assert len(schemas) == 1
        params = schemas[0].parameters
        assert "cloudId" in params["properties"]
        assert "cloudId" in params["required"]
        assert "cql" in params["required"]

    @pytest.mark.asyncio
    async def test_tool_invoke_applies_arg_defaults(self):
        """The LLM-facing tool.invoke path resolves the raw tool name and injects arg_defaults.

        This drives the full production chain - tools() builds the mangled
        ``mcp_{server}_{hash}_{op}`` schema name, resolve_operation reverses it back to
        the raw ``searchAtlassian`` key, and only then does _get_tool_config match and
        _apply_arg_defaults inject. Calling _call_mcp_tool directly would bypass that
        resolution and hide a name-mismatch regression.
        """
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    arg_defaults={"cloudId": "${TEST_ARG_DEFAULT_CLOUD_ID}"}
                )
            },
        )
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        provider._mcp_tools = [
            SimpleNamespace(
                name="searchAtlassian",
                description="Rovo cross-product search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cloudId": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["cloudId", "query"],
                },
            )
        ]

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                isError=False,
                structuredContent={"ok": True},
                content=[],
            )
        )
        provider._session = mock_session

        tools = provider.tools()
        assert len(tools) == 1
        # The LLM-facing name is the mangled schema name, not the raw operation.
        assert tools[0].definition.name.endswith("_searchAtlassian")

        with patch.dict("os.environ", {"TEST_ARG_DEFAULT_CLOUD_ID": "cloud-123"}, clear=False):
            result = await tools[0].invoke({"query": "redis"})

        assert result["status"] == "success"
        mock_session.call_tool.assert_awaited_once()
        call_args, call_kwargs = mock_session.call_tool.call_args
        # resolve_operation recovered the raw name that _get_tool_config keys on.
        assert call_args[0] == "searchAtlassian"
        assert call_kwargs["arguments"] == {"query": "redis", "cloudId": "cloud-123"}

    def test_try_parse_json_object_returns_dict_for_object(self):
        """A JSON object string parses to the corresponding dict (whitespace-tolerant)."""
        assert MCPToolProvider._try_parse_json_object('{"a": 1}') == {"a": 1}
        assert MCPToolProvider._try_parse_json_object('  {"a": 1}\n') == {"a": 1}

    def test_try_parse_json_object_returns_none_for_non_object(self):
        """Arrays, scalars, malformed, and non-JSON text all return None (never hoisted)."""
        assert MCPToolProvider._try_parse_json_object("[1, 2, 3]") is None
        assert MCPToolProvider._try_parse_json_object('"just a string"') is None
        assert MCPToolProvider._try_parse_json_object("42") is None
        assert MCPToolProvider._try_parse_json_object("plain text") is None
        assert MCPToolProvider._try_parse_json_object("") is None
        assert MCPToolProvider._try_parse_json_object("   ") is None
        assert MCPToolProvider._try_parse_json_object('{"a": 1') is None

    @staticmethod
    def _make_session(*, structured_content, text):
        """Build a mock MCP session returning one text block and optional structuredContent."""
        session = MagicMock()
        session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                isError=False,
                structuredContent=structured_content,
                content=[mcp_types.TextContent(type="text", text=text)],
            )
        )
        return session

    @pytest.mark.asyncio
    async def test_call_mcp_tool_hoists_json_text_payload(self):
        """A stringified JSON object with no structuredContent is merged onto the response."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="atlassian", server_config=config)
        payload = {"results": [{"title": "T", "url": "u"}], "total": 1}
        provider._session = self._make_session(
            structured_content=None, text=json.dumps(payload)
        )

        result = await provider._call_mcp_tool("searchConfluenceUsingCql", {})

        assert result["status"] == "success"
        assert result["results"] == payload["results"]
        assert result["total"] == 1
        # Raw text is preserved alongside the hoisted keys.
        assert result["text"] == json.dumps(payload)

    @pytest.mark.asyncio
    async def test_call_mcp_tool_skips_hoist_when_structured_content_present(self):
        """JSON merging does not run when the server already provided structuredContent."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="atlassian", server_config=config)
        payload = {"results": [{"title": "T"}]}
        provider._session = self._make_session(
            structured_content={"ok": True}, text=json.dumps(payload)
        )

        result = await provider._call_mcp_tool("searchConfluenceUsingCql", {})

        assert result["data"] == {"ok": True}
        assert "results" not in result
        assert result["text"] == json.dumps(payload)

    @pytest.mark.asyncio
    async def test_call_mcp_tool_hoist_does_not_overwrite_reserved_keys(self):
        """Hoisting never clobbers provider-owned keys (status/text/...)."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="atlassian", server_config=config)
        payload = {"status": "hacked", "text": "hacked", "results": [1]}
        provider._session = self._make_session(
            structured_content=None, text=json.dumps(payload)
        )

        result = await provider._call_mcp_tool("searchConfluenceUsingCql", {})

        assert result["status"] == "success"
        assert result["text"] == json.dumps(payload)
        assert result["results"] == [1]

    @pytest.mark.asyncio
    async def test_call_mcp_tool_leaves_non_json_text_untouched(self):
        """Plain-text responses are left as-is (no hoisting, text preserved)."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="atlassian", server_config=config)
        provider._session = self._make_session(
            structured_content=None, text="just some prose"
        )

        result = await provider._call_mcp_tool("searchConfluenceUsingCql", {})

        assert result == {"status": "success", "text": "just some prose"}

    @pytest.mark.asyncio
    async def test_call_mcp_tool_applies_configured_result_shaping_end_to_end(self):
        """A tool with result_shaping is filtered + trimmed in BOTH hoisted results and text."""
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    result_shaping=MCPResultShaping(
                        include_types=["page", "blogpost"],
                        max_results=8,
                    )
                )
            },
        )
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        payload = {
            "results": [_result("page", 1), _result("issue", 1), _result("page", 2)],
            "total": 3,
        }
        provider._session = self._make_session(
            structured_content=None, text=json.dumps(payload)
        )

        result = await provider._call_mcp_tool("searchAtlassian", {})

        # Out-of-scope "issue" dropped from the hoisted, citation-facing results.
        assert [r["type"] for r in result["results"]] == ["page", "page"]
        # The raw text blob the LLM sees is rewritten to the same shaped set...
        reserialized = json.loads(result["text"])
        assert reserialized["results"] == result["results"]
        # ...while non-results keys are preserved.
        assert reserialized["total"] == 3

    @pytest.mark.asyncio
    async def test_call_mcp_tool_leaves_results_untouched_without_shaping_config(self):
        """A tool with no result_shaping config keeps every result and the raw text."""
        config = MCPServerConfig(command="test")
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        payload = {"results": [_result("page", 1), _result("issue", 1)]}
        provider._session = self._make_session(
            structured_content=None, text=json.dumps(payload)
        )

        result = await provider._call_mcp_tool("searchAtlassian", {})

        assert result["results"] == payload["results"]
        assert result["text"] == json.dumps(payload)

    @pytest.mark.asyncio
    async def test_call_mcp_tool_does_not_shape_structured_content(self):
        """result_shaping is text-path only; structuredContent passes through untouched.

        Shaping deliberately does not run on the structuredContent branch: results there
        live nested under response['data'] (not hoisted to top level), so they are neither
        cited nor consumed the way the text-path payload is, and trimming only 'data' while
        leaving the full list in 'text' would present two inconsistent sets to the LLM.
        """
        config = MCPServerConfig(
            command="test",
            tools={
                "searchAtlassian": MCPToolConfig(
                    result_shaping=MCPResultShaping(include_types=["page"])
                )
            },
        )
        provider = MCPToolProvider(server_name="atlassian_rovo_mcp", server_config=config)
        provider._session = self._make_session(
            structured_content={"results": [_result("page", 1), _result("issue", 1)]},
            text="{}",
        )

        result = await provider._call_mcp_tool("searchAtlassian", {})

        assert [r["type"] for r in result["data"]["results"]] == ["page", "issue"]


def _result(type_: str, index: int) -> dict:
    return {"id": f"{type_}/{index}", "title": f"{type_} {index}", "type": type_}


class TestMCPResultShaping:
    """Unit tests for the generic, config-driven result shaping."""

    def test_filters_by_include_types(self):
        shaping = MCPResultShaping(include_types=["page", "blogpost"])
        parsed = {"results": [_result("page", 1), _result("issue", 1), _result("blogpost", 1)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert [r["type"] for r in parsed["results"]] == ["page", "blogpost"]

    def test_filter_then_cap_compose(self):
        # Cap counts POST-filter: filter to pages, then keep the first 2 of those.
        shaping = MCPResultShaping(include_types=["page"], max_results=2)
        parsed = {
            "results": [
                _result("page", 1),
                _result("issue", 1),
                _result("page", 2),
                _result("page", 3),
            ]
        }

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert [r["title"] for r in parsed["results"]] == ["page 1", "page 2"]

    def test_max_results_is_a_cap(self):
        shaping = MCPResultShaping(max_results=2)
        parsed = {"results": [_result("page", i) for i in range(5)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert len(parsed["results"]) == 2

    def test_all_filtered_out_yields_empty(self):
        # Nothing in scope -> empty rather than a weak out-of-scope hit.
        shaping = MCPResultShaping(include_types=["page"])
        parsed = {"results": [_result("issue", 1), _result("issue", 2)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert parsed["results"] == []

    def test_custom_results_path_and_type_field(self):
        # Generic: not tied to "results"/"type" key names.
        shaping = MCPResultShaping(
            results_path="items", type_field="kind", include_types=["doc"]
        )
        parsed = {"items": [{"kind": "doc"}, {"kind": "ticket"}, {"kind": "doc"}]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert parsed["items"] == [{"kind": "doc"}, {"kind": "doc"}]

    def test_noop_when_nothing_changes(self):
        # A config that removes nothing reports no change (so text is not rewritten).
        shaping = MCPResultShaping(include_types=["page"])
        parsed = {"results": [_result("page", 1), _result("page", 2)]}

        assert MCPToolProvider._apply_result_shaping(parsed, shaping) is False
        assert len(parsed["results"]) == 2

    def test_missing_or_empty_results_are_noops(self):
        shaping = MCPResultShaping(include_types=["page"])
        assert MCPToolProvider._apply_result_shaping({}, shaping) is False
        assert MCPToolProvider._apply_result_shaping({"results": []}, shaping) is False
        assert MCPToolProvider._apply_result_shaping({"results": "nope"}, shaping) is False

    def test_non_dict_item_dropped_only_when_filtering(self):
        # With a type filter, a non-dict item can't match and is dropped.
        shaping = MCPResultShaping(include_types=["page"])
        parsed = {"results": ["not-a-dict", _result("page", 1)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert parsed["results"] == [_result("page", 1)]

    def test_non_dict_item_kept_when_not_filtering(self):
        # Without include_types, items are not inspected, so non-dicts survive trimming.
        shaping = MCPResultShaping(max_results=3)
        parsed = {"results": ["a", {"whatever": 1}, 42, "d"]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert parsed["results"] == ["a", {"whatever": 1}, 42]

    def test_generic_shape_unlike_atlassian(self):
        # An Elasticsearch/OpenSearch-style shape with entirely different key names and
        # unrelated item fields. Proves the trimmer is not coupled to Atlassian at all,
        # and that filter -> cap compose on a foreign shape.
        shaping = MCPResultShaping(
            results_path="hits",
            type_field="doctype",
            include_types=["article"],
            max_results=2,
        )
        parsed = {
            "hits": [
                {"doctype": "article", "score": 9.1, "body": "keep-1"},
                {"doctype": "faq", "score": 8.0},
                {"doctype": "article", "score": 7.2, "body": "keep-2"},
                {"doctype": "article", "score": 6.0, "body": "capped-out"},
            ],
            "took_ms": 12,
        }

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        # faq filtered; cap=2 -> first two articles.
        assert [h["body"] for h in parsed["hits"]] == ["keep-1", "keep-2"]
        # Unrelated top-level keys are untouched.
        assert parsed["took_ms"] == 12

    def test_missing_configured_path_is_safe_noop(self):
        # If the configured results_path is absent (shape mismatch), do nothing rather
        # than guess or drop data. This is the core safety property for unknown shapes.
        shaping = MCPResultShaping(results_path="hits", include_types=["article"])
        parsed = {"results": [_result("page", 1)], "other": "x"}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is False
        assert parsed == {"results": [_result("page", 1)], "other": "x"}

    def test_nested_path_is_not_traversed_and_is_safe_noop(self):
        # KNOWN LIMITATION: results_path is a single top-level key, not a dotted path.
        # A nested result list is left untouched (safe) rather than silently mangled.
        shaping = MCPResultShaping(results_path="results", include_types=["page"])
        parsed = {"data": {"results": [_result("page", 1), _result("issue", 1)]}}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is False
        assert parsed["data"]["results"] == [_result("page", 1), _result("issue", 1)]

    def test_items_missing_type_field_are_dropped_under_filter(self):
        # A scope filter (include_types) intentionally drops items lacking the type field
        # (item.get(type_field) is None, not in include_types). This keeps a Jira-like
        # item that omits the field from slipping into a Confluence-only result set.
        shaping = MCPResultShaping(include_types=["page"])
        parsed = {"results": [{"title": "no type here"}, _result("page", 1)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert parsed["results"] == [_result("page", 1)]

    def test_trim_only_without_filter_keeps_all_kinds(self):
        # include_types unset: pure cap, no scope filtering, so every kind is eligible.
        shaping = MCPResultShaping(max_results=2)
        parsed = {"results": [_result("page", 1), _result("issue", 1), _result("faq", 1)]}

        applied = MCPToolProvider._apply_result_shaping(parsed, shaping)

        assert applied is True
        assert [r["type"] for r in parsed["results"]] == ["page", "issue"]

    @pytest.mark.parametrize(
        "reserved", ["status", "text", "error", "images", "resources", "data"]
    )
    def test_results_path_colliding_with_reserved_key_is_rejected(self, reserved):
        # A results_path pointing at an envelope-owned key would be shaped and then
        # silently dropped by the text-path hoist guard. Fail fast at config load.
        with pytest.raises(ValidationError):
            MCPResultShaping(results_path=reserved)

    def test_default_results_path_is_accepted(self):
        # The default and ordinary domain keys pass validation unchanged.
        assert MCPResultShaping().results_path == "results"
        assert MCPResultShaping(results_path="items").results_path == "items"
