"""MCP (Model Context Protocol) tool provider.

This module provides a dynamic tool provider that connects to an MCP server
and exposes its tools to the agent. It supports tool filtering and description
overrides based on the MCPServerConfig.
"""

import json
import logging
import os
import re
from collections import Counter
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from redis_sre_agent.core.config import (
    RESERVED_RESPONSE_KEYS,
    MCPResultShaping,
    MCPServerConfig,
    MCPToolConfig,
)
from redis_sre_agent.core.runtime_overrides import get_active_mcp_runtime
from redis_sre_agent.tools.models import (
    Tool,
    ToolActionKind,
    ToolCapability,
    ToolDefinition,
    ToolMetadata,
    infer_tool_action_kind,
)
from redis_sre_agent.tools.protocols import ToolProvider

if TYPE_CHECKING:
    from redis_sre_agent.core.instances import RedisInstance

logger = logging.getLogger(__name__)

# Matches an unresolved ${VAR} placeholder left behind when os.path.expandvars
# cannot resolve an environment variable (i.e. it is unset).
_UNRESOLVED_VAR_RE = re.compile(r"\$\{[^}]*\}")

def _coerce_input_schema_dict(input_schema: Any) -> Optional[Dict[str, Any]]:
    """Normalize MCP input schemas into plain JSON-serializable dicts."""
    if isinstance(input_schema, dict):
        return dict(input_schema)
    if hasattr(input_schema, "model_dump"):
        try:
            payload = input_schema.model_dump(mode="json")
        except TypeError:
            payload = input_schema.model_dump()
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(input_schema, "dict"):
        payload = input_schema.dict()
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(input_schema, "items"):
        try:
            return dict(input_schema.items())
        except Exception:
            return None
    return None


class MCPToolProvider(ToolProvider):
    """Dynamic tool provider that connects to an MCP server.

    This provider:
    1. Connects to an MCP server using the configured transport (stdio or HTTP)
    2. Discovers available tools from the server
    3. Optionally filters tools based on the config's `tools` mapping
    4. Applies capability and description overrides from the config
    5. Exposes the tools to the agent

    Example:
        config = MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            tools={
                "search_memories": MCPToolConfig(capability=ToolCapability.LOGS),
            }
        )
        provider = MCPToolProvider(
            server_name="memory",
            server_config=config,
        )
        async with provider:
            tools = provider.tools()
    """

    # Default capability for MCP tools if not specified
    DEFAULT_CAPABILITY = ToolCapability.UTILITIES

    def __init__(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        redis_instance: Optional["RedisInstance"] = None,
        use_pool: bool = True,
    ):
        """Initialize the MCP tool provider.

        Args:
            server_name: Name of the MCP server (used in tool naming)
            server_config: Configuration for the MCP server
            redis_instance: Optional Redis instance (not typically used by MCP)
            use_pool: If True, use pooled connection when available (default: True)
        """
        super().__init__(redis_instance=redis_instance)
        self._server_name = server_name
        self._server_config = server_config
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._mcp_tools: List[mcp_types.Tool] = []
        self._tool_cache: List[Tool] = []
        self._use_pool = use_pool
        self._using_pooled_connection = False
        self._using_eval_runtime = False

    @property
    def provider_name(self) -> str:
        """Return the provider name based on the server name."""
        return f"mcp_{self._server_name}"

    async def __aenter__(self) -> "MCPToolProvider":
        """Enter async context and connect to the MCP server."""
        await self._connect()
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context and disconnect from the MCP server."""
        await self._disconnect()

    async def _connect(self) -> None:
        """Connect to the MCP server and discover tools.

        This method first tries to use a pooled connection if available.
        Falls back to creating a new connection if pool is empty or disabled.
        """
        eval_runtime = get_active_mcp_runtime()
        if eval_runtime is not None:
            eval_session = eval_runtime.get_server_session(self._server_name)
            if eval_session is not None:
                self._session = eval_session
                tools_result = await eval_session.list_tools()
                self._mcp_tools = tools_result.tools
                self._tool_cache = []
                self._using_eval_runtime = True
                logger.info(
                    "Using eval fake MCP runtime for server '%s' (%s tools)",
                    self._server_name,
                    len(self._mcp_tools),
                )
                return

        # Try to use pooled connection first
        if self._use_pool:
            try:
                from redis_sre_agent.tools.mcp.pool import MCPConnectionPool

                pool = MCPConnectionPool.get_instance()
                pooled_conn = pool.get_connection(self._server_name)
                if pooled_conn:
                    self._session = pooled_conn.session
                    self._mcp_tools = pooled_conn.tools
                    self._tool_cache = []
                    self._using_pooled_connection = True
                    logger.info(
                        f"Using pooled connection for MCP server '{self._server_name}' "
                        f"({len(self._mcp_tools)} tools)"
                    )
                    return
            except Exception as e:
                logger.debug(f"Pool not available for '{self._server_name}': {e}")

        # Fall back to creating a new connection
        try:
            logger.info(
                f"Creating new connection to MCP server '{self._server_name}' "
                f"(command={self._server_config.command}, url={self._server_config.url})"
            )

            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            # Determine transport type and connect
            if self._server_config.command:
                # Stdio transport - spawn a subprocess
                # Merge parent environment with config-specified env so that
                # env vars like OPENAI_API_KEY are inherited by the subprocess.
                # Expand ${VAR} patterns in config env values (e.g., ${GITHUB_PERSONAL_ACCESS_TOKEN})
                config_env = {}
                for key, value in (self._server_config.env or {}).items():
                    config_env[key] = os.path.expandvars(value)
                merged_env = {**os.environ, **config_env}
                server_params = StdioServerParameters(
                    command=self._server_config.command,
                    args=self._server_config.args or [],
                    env=merged_env,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            elif self._server_config.url:
                # URL-based transport (SSE or Streamable HTTP)
                expanded_url = os.path.expandvars(self._server_config.url)
                # Expand environment variables in headers (e.g., ${GITHUB_TOKEN})
                headers = None
                if self._server_config.headers:
                    headers = {}
                    for key, value in self._server_config.headers.items():
                        # Expand ${VAR} patterns from environment
                        expanded_value = os.path.expandvars(value)
                        headers[key] = expanded_value

                # Determine transport type - default to streamable_http for modern servers
                transport_type = (self._server_config.transport or "streamable_http").lower()

                if transport_type == "sse":
                    # Legacy SSE transport
                    logger.info(f"Using SSE transport for '{self._server_name}'")
                    read_stream, write_stream = await self._exit_stack.enter_async_context(
                        sse_client(expanded_url, headers=headers)
                    )
                else:
                    # Streamable HTTP transport (default, works with GitHub remote MCP, etc.)
                    logger.info(f"Using Streamable HTTP transport for '{self._server_name}'")
                    (
                        read_stream,
                        write_stream,
                        _get_session_id,
                    ) = await self._exit_stack.enter_async_context(
                        streamablehttp_client(expanded_url, headers=headers)
                    )
            else:
                raise ValueError(
                    f"MCP server '{self._server_name}' must have either 'command' or 'url' configured"
                )

            # Create and initialize the session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            # Discover tools from the server
            tools_result = await self._session.list_tools()
            self._mcp_tools = tools_result.tools
            self._tool_cache = []

            logger.info(
                f"MCP server '{self._server_name}' connected with {len(self._mcp_tools)} tools: "
                f"{[t.name for t in self._mcp_tools]}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self._server_name}': {e}")
            # Clean up on failure
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            raise

    async def _disconnect(self) -> None:
        """Disconnect from the MCP server.

        If using a pooled connection, just clears local references without
        closing the shared session. Only closes connections we own.
        """
        if self._using_pooled_connection:
            # Don't close pooled connections - they're managed by the pool
            logger.debug(f"Releasing pooled connection for '{self._server_name}'")
            self._session = None
            self._using_pooled_connection = False
            return

        if self._using_eval_runtime:
            logger.debug(f"Releasing eval fake MCP runtime session for '{self._server_name}'")
            self._session = None
            self._using_eval_runtime = False
            return

        try:
            if self._exit_stack:
                logger.info(f"Disconnecting from MCP server '{self._server_name}'")
                await self._exit_stack.aclose()
                self._exit_stack = None
                self._session = None
        except Exception as e:
            logger.warning(f"Error disconnecting from MCP server '{self._server_name}': {e}")

    def _get_tool_config(self, tool_name: str) -> Optional[MCPToolConfig]:
        """Get the configuration for a specific tool, if any."""
        if self._server_config.tools:
            return self._server_config.tools.get(tool_name)
        return None

    def _should_include_tool(self, tool_name: str) -> bool:
        """Check if a tool should be included based on the config.

        If `tools` is specified in the config, only those tools are included.
        If `tools` is None, all tools from the server are included.
        """
        if self._server_config.tools is None:
            return True
        return tool_name in self._server_config.tools

    def _get_capability(self, tool_name: str) -> ToolCapability:
        """Get the capability for a tool, with config override support."""
        config = self._get_tool_config(tool_name)
        if config and config.capability:
            return config.capability
        return self.DEFAULT_CAPABILITY

    def _get_description(self, tool_name: str, mcp_description: str) -> str:
        """Get the description for a tool, with config override/template support.

        If the config provides a description, it can use {original} as a placeholder
        for the MCP tool's original description. This allows adding context while
        preserving the original tool documentation.

        Examples:
            - No override: uses original MCP description
            - Override without placeholder: "Custom description" -> replaces entirely
            - Override with placeholder: "Context. {original}" -> prepends context

        Args:
            tool_name: Name of the MCP tool
            mcp_description: Original description from the MCP server

        Returns:
            Final description (original, override, or templated)
        """
        config = self._get_tool_config(tool_name)
        if config and config.description:
            # Support templating: {original} gets replaced with the MCP description
            if "{original}" in config.description:
                return config.description.replace("{original}", mcp_description)
            return config.description
        return mcp_description

    def _get_action_kind(
        self,
        tool_name: str,
        description: str,
        capability: ToolCapability,
    ) -> ToolActionKind:
        """Get the approval action kind for a tool, with config override support."""
        config = self._get_tool_config(tool_name)
        if config and config.action_kind is not None:
            return config.action_kind
        return infer_tool_action_kind(
            name=tool_name,
            description=description,
            capability=capability,
            provider_name=self.provider_name,
        )

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """Create tool schemas from the MCP server's tools.

        This method transforms MCP tool definitions into ToolDefinition objects,
        applying any configured filters, capability overrides, and description
        overrides.
        """
        schemas: List[ToolDefinition] = []

        for mcp_tool in self._mcp_tools:
            tool_name = mcp_tool.name
            if not tool_name:
                continue

            # Check if tool should be included
            if not self._should_include_tool(tool_name):
                continue

            # Get description (with potential override)
            mcp_description = mcp_tool.description or f"MCP tool: {tool_name}"
            description = self._get_description(tool_name, mcp_description)

            # Get capability (with potential override)
            capability = self._get_capability(tool_name)

            # Build parameters schema from MCP tool input schema.
            # Keys that have a resolvable arg_default are stripped from the advertised
            # schema (properties + required): they are pinned deployment values injected
            # at call time, so exposing them only causes the model to stall asking the
            # user for a value it cannot know (e.g. Atlassian cloudId).
            input_schema = _coerce_input_schema_dict(mcp_tool.inputSchema) or {}
            input_schema = self._strip_pinned_keys(
                input_schema, self._resolved_arg_default_keys(tool_name)
            )
            parameters = {
                "type": "object",
                "properties": input_schema.get("properties") or {},
                "required": input_schema.get("required") or [],
            }

            schema = ToolDefinition(
                name=self._make_tool_name(tool_name),
                description=description,
                capability=capability,
                parameters=parameters,
            )
            schemas.append(schema)

        return schemas

    def get_input_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return raw input schemas keyed by original MCP tool name."""
        schemas: Dict[str, Dict[str, Any]] = {}

        for mcp_tool in self._mcp_tools:
            tool_name = mcp_tool.name
            if not tool_name or not self._should_include_tool(tool_name):
                continue

            input_schema = _coerce_input_schema_dict(getattr(mcp_tool, "inputSchema", None))
            if input_schema is not None:
                schemas[tool_name] = self._strip_pinned_keys(
                    input_schema, self._resolved_arg_default_keys(tool_name)
                )

        return schemas

    def tools(self) -> List[Tool]:
        """Return the concrete tools exposed by this provider.

        This caches the tools list to avoid rebuilding on every call.
        """
        if self._tool_cache:
            return self._tool_cache

        schemas = self.create_tool_schemas()
        tools: List[Tool] = []

        for schema in schemas:
            # Extract the original MCP tool name from our tool name
            mcp_tool_name = self.resolve_operation(schema.name, {}) or ""

            meta = ToolMetadata(
                name=schema.name,
                description=schema.description,
                capability=schema.capability,
                provider_name=self.provider_name,
                requires_instance=False,  # MCP tools typically don't require Redis instance
                action_kind=self._get_action_kind(
                    mcp_tool_name,
                    schema.description,
                    schema.capability,
                ),
            )

            # Create the invoke closure that calls the MCP server
            async def _invoke(
                args: Dict[str, Any],
                _tool_name: str = mcp_tool_name,
            ) -> Any:
                return await self._call_mcp_tool(_tool_name, args)

            tools.append(Tool(metadata=meta, definition=schema, invoke=_invoke))

        self._tool_cache = tools
        return tools

    @staticmethod
    def _resolve_arg_default(value: Any) -> Any:
        """Resolve a single arg_default value, or return ``None`` if it should be skipped.

        String values undergo ${VAR} environment expansion. A resolved value is skipped
        (returns ``None``) when it is an empty string or still contains an unresolved
        ${VAR} placeholder, so a literal placeholder is never used and the model-supplied
        value can serve as the fallback. Non-string values pass through verbatim.
        """
        if not isinstance(value, str):
            return value
        resolved = os.path.expandvars(value)
        if resolved == "" or _UNRESOLVED_VAR_RE.search(resolved):
            return None
        return resolved

    def _resolved_arg_default_keys(self, tool_name: str) -> set:
        """Return the arg_default keys for a tool whose values resolve to a concrete value.

        These keys are pinned deployment config injected at call time, so they are
        stripped from the advertised schema. Keys with unresolved/empty defaults are
        NOT included, so they remain visible for the model to supply as a fallback.
        """
        config = self._get_tool_config(tool_name)
        if not config or not config.arg_defaults:
            return set()
        return {
            key
            for key, value in config.arg_defaults.items()
            if self._resolve_arg_default(value) is not None
        }

    @staticmethod
    def _strip_pinned_keys(schema: Dict[str, Any], pinned_keys: set) -> Dict[str, Any]:
        """Return a copy of ``schema`` with ``pinned_keys`` removed from properties + required.

        Pinned keys are arg_defaults that resolve to a concrete value; they are injected
        at call time, so they are hidden from the advertised schema to stop the model from
        stalling to ask the user for a value it cannot know. The rest of the raw schema
        (title, additionalProperties, ...) is preserved. Returns the input unchanged when
        there are no pinned keys.
        """
        if not pinned_keys:
            return schema
        result = dict(schema)
        if isinstance(result.get("properties"), dict):
            result["properties"] = {
                key: value
                for key, value in result["properties"].items()
                if key not in pinned_keys
            }
        if isinstance(result.get("required"), list):
            result["required"] = [key for key in result["required"] if key not in pinned_keys]
        return result

    def _apply_arg_defaults(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Inject configured ``arg_defaults`` into a tool call at invocation time.

        This does NOT modify the tool's advertised schema - the arguments the MCP
        server declared (including ``required``) are left intact. Configured defaults
        are applied here, per-call, for fixed deployment values (e.g. account/tenant/
        region/resource ids) the model cannot know. String values support ${VAR}
        environment expansion; entries that are empty or still contain an unresolved
        ${VAR} placeholder are skipped (and logged) so a literal placeholder is never
        sent, and the model's own value is left as the fallback in that case.

        Configured defaults take precedence over the model-supplied value because they
        represent pinned deployment configuration; each override is logged for
        traceability.
        """
        config = self._get_tool_config(tool_name)
        if not config or not config.arg_defaults:
            return args

        merged = dict(args or {})
        for key, value in config.arg_defaults.items():
            resolved = self._resolve_arg_default(value)
            if resolved is None:
                logger.warning(
                    "Skipping arg_default '%s' for tool '%s': value is empty or has an "
                    "unresolved ${VAR} placeholder; using the model-supplied value (if any).",
                    key,
                    tool_name,
                )
                continue
            if key in merged and merged[key] != resolved:
                logger.debug(
                    "Overriding model-supplied '%s' for tool '%s' with configured arg_default.",
                    key,
                    tool_name,
                )
            merged[key] = resolved
        return merged

    @staticmethod
    def _prune_in_place(obj: Any, drop_names: set) -> "Counter[str]":
        """Recursively remove any dict key whose NAME is in ``drop_names``; return a
        ``Counter`` of the names actually removed (empty when nothing changed).

        Matches by exact key name at any depth (path-agnostic), not by dotted path.
        Removing a key drops its entire subtree and counts as ONE removal for that name, so
        the tally reflects dropped keys, not every descendant that vanished with them (e.g.
        dropping ``history`` removes the whole author-profile subtree but counts once).
        Recurses into surviving dict values and list items; non-container values are left
        as-is. Matched keys short-circuit descent (the dropped subtree is not traversed).

        Returning the actual names (not just a total) lets the caller log which fields were
        stripped, rather than echoing the configured denylist. The walk trusts
        ``drop_names`` verbatim; the protected-name floor that keeps citation/identity
        fields undroppable is enforced upstream at config load
        (MCPResultShaping.reject_protected_drop_fields).
        """
        removed: "Counter[str]" = Counter()
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in drop_names:
                    del obj[key]
                    removed[key] += 1
                else:
                    removed.update(MCPToolProvider._prune_in_place(obj[key], drop_names))
        elif isinstance(obj, list):
            for item in obj:
                removed.update(MCPToolProvider._prune_in_place(item, drop_names))
        return removed

    @staticmethod
    def _apply_result_shaping(
        container: Dict[str, Any],
        shaping: MCPResultShaping,
        tool_name: str = "",
    ) -> bool:
        """Trim/filter/prune a tool's result list in-place; return True if it changed.

        Vendor-neutral: some MCP servers expose no server-side control over how many
        results come back, which kinds are included, or how verbose each result is, so a
        single search returns a large, mixed, low-signal set. This optionally (a) keeps only
        results whose ``type_field`` value is in ``include_types``, (b) limits the final
        number of results to ``max_results``, and (c) recursively prunes ``drop_fields`` key
        names from each kept result. What qualifies as in-scope (the allowed type values, the
        result limit, the pruned field names) is controlled entirely by configuration, so no
        provider- or product-specific logic is embedded here. Only tools that explicitly
        enable ``result_shaping`` in their configuration are shaped.

        Runs before the response is serialized for the LLM and before citation
        extraction, so the shaped set is the only one either ever sees.
        """
        # Phase 0 - locate: shaping is a no-op unless results_path holds a non-empty list.
        results = container.get(shaping.results_path)
        if not isinstance(results, list) or not results:
            return False

        original_count = len(results)
        limit = shaping.max_results
        include = shaping.include_types

        # Phase 1 + 2 - filter by include_types, then cap at max_results (single pass:
        # only kept items count toward the cap).
        shaped: List[Any] = []
        for item in results:
            if include is not None and (
                not isinstance(item, dict) or item.get(shaping.type_field) not in include
            ):
                continue
            shaped.append(item)
            if limit is not None and len(shaped) >= limit:
                break

        # Compute filter/cap change BEFORE pruning: prune mutates items in place, and
        # ``shaped`` holds the same dict references as ``results`` for kept items, so a
        # post-prune equality check would miss a prune-only change.
        list_changed = shaped != results

        # Phase 3 - prune: recursively strip drop_fields from each kept result, tallying
        # the names actually removed (not just the configured denylist).
        pruned: "Counter[str]" = Counter()
        if shaping.drop_fields:
            drop_names = set(shaping.drop_fields)
            for item in shaped:
                pruned.update(MCPToolProvider._prune_in_place(item, drop_names))

        # Phase 4 - commit: bail if nothing changed, else log and write the shaped list back.
        if not list_changed and not pruned:
            return False

        # Pruning and capping are silent and lossy (dropped data never reaches the LLM,
        # citations, or the trace), so record what shaping removed to make "why is this
        # field/result missing?" answerable without reconstructing it. INFO, not DEBUG:
        # this fires only on tools that opt into shaping AND only when something was
        # actually removed, so it is meaningful signal, not per-request noise.
        logger.info(
            "result_shaping on '%s' (%s): kept %d/%d result(s), pruned %d field(s)%s",
            tool_name or "<unknown tool>",
            shaping.results_path,
            len(shaped),
            original_count,
            sum(pruned.values()),
            f" {dict(sorted(pruned.items()))}" if pruned else "",
        )
        container[shaping.results_path] = shaped
        return True

    @staticmethod
    def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Parse ``text`` as a JSON object, returning the dict or ``None``.

        Only JSON *objects* (dicts) are returned. JSON arrays/scalars and non-JSON
        text return ``None`` so they remain available via the raw ``text`` field and
        are never hoisted onto the response.
        """
        stripped = text.strip()
        if not stripped or stripped[0] != "{":
            return None
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call an MCP tool on the server.

        Args:
            tool_name: The original MCP tool name (without provider prefix)
            args: Arguments to pass to the tool

        Returns:
            The tool's result from the MCP server
        """
        if not self._session:
            return {
                "status": "error",
                "error": f"MCP server '{self._server_name}' is not connected",
            }

        args = self._apply_arg_defaults(tool_name, args)

        try:
            logger.info(f"Calling MCP tool '{tool_name}' with args: {args}")
            result = await self._session.call_tool(tool_name, arguments=args)

            # Check for errors
            if result.isError:
                error_text = ""
                for content in result.content:
                    if isinstance(content, mcp_types.TextContent):
                        error_text += content.text
                return {
                    "status": "error",
                    "error": error_text or "Tool execution failed",
                }

            # Extract the result content
            response: Dict[str, Any] = {"status": "success"}

            # Optional client-side trim/filter for tools that opt in via config (e.g.
            # search tools whose server exposes no limit/type/score lever). See
            # MCPResultShaping / _apply_result_shaping. None for unconfigured tools.
            tool_config = self._get_tool_config(tool_name)
            result_shaping = tool_config.result_shaping if tool_config else None

            # If there's structured content, use it
            if result.structuredContent:
                response["data"] = result.structuredContent

            # Also extract text content for compatibility
            text_parts = []
            for content in result.content:
                if isinstance(content, mcp_types.TextContent):
                    text_parts.append(content.text)
                elif isinstance(content, mcp_types.ImageContent):
                    response.setdefault("images", []).append(
                        {
                            "mimeType": content.mimeType,
                            "data": content.data,
                        }
                    )
                elif isinstance(content, mcp_types.EmbeddedResource):
                    resource = content.resource
                    if isinstance(resource, mcp_types.TextResourceContents):
                        response.setdefault("resources", []).append(
                            {
                                "uri": str(resource.uri),
                                "text": resource.text,
                            }
                        )

            if text_parts:
                joined_text = "\n".join(text_parts)
                response["text"] = joined_text

                # Some MCP servers return their payload as a stringified JSON object in a
                # text block and set no structuredContent, leaving the model (and our
                # downstream consumers) an opaque blob. When that happens, parse the text
                # and merge its top-level keys onto the response so consumers see real
                # structure instead of a string: citation extraction reads
                # response["results"] and evidence expansion runs JMESPath over the
                # response. Only JSON objects are merged; envelope-owned keys are never
                # overwritten and the full raw payload always remains in response["text"].
                if not result.structuredContent:
                    parsed = self._try_parse_json_object(joined_text)
                    if parsed is not None:
                        # Apply configured result shaping before hoisting. When it changes
                        # the payload, rewrite the raw text blob too so the LLM-facing
                        # payload and the hoisted keys reflect the same shaped set (the
                        # trimmed-out results are deliberately dropped, not preserved).
                        if result_shaping and self._apply_result_shaping(
                            parsed, result_shaping, tool_name
                        ):
                            response["text"] = json.dumps(parsed)
                        for key, value in parsed.items():
                            if key not in RESERVED_RESPONSE_KEYS:
                                response.setdefault(key, value)

                        # The hoisted top-level keys now fully duplicate the raw text blob,
                        # so text is redundant and needlessly doubles the payload in the
                        # envelope (citations, expand_evidence, decision trace). Drop it -
                        # but only when no top-level key collided with a reserved envelope
                        # key: a colliding key (e.g. a payload-level "data") is skipped by
                        # the hoist above and survives ONLY in text, so keep text in that
                        # case. Non-hoisting tools (non-JSON text, single-page bodies) never
                        # reach this branch, so their text is left untouched.
                        if not (set(parsed) & RESERVED_RESPONSE_KEYS):
                            response.pop("text", None)

            return response

        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            return {
                "status": "error",
                "error": str(e),
            }
