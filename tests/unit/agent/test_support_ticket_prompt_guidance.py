"""Tests for support-ticket guidance in startup and agent prompts."""

from redis_sre_agent.agent.chat_agent import CHAT_SYSTEM_PROMPT
from redis_sre_agent.agent.knowledge_agent import KNOWLEDGE_SYSTEM_PROMPT
from redis_sre_agent.agent.knowledge_context import _tool_instruction_lines_for_categories
from redis_sre_agent.agent.prompts import REDIS_COMMAND_SEMANTICS_GUARDRAILS, SRE_SYSTEM_PROMPT
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition


def test_startup_context_includes_support_ticket_tool_instructions():
    lines = _tool_instruction_lines_for_categories(
        [
            ToolDefinition(
                name="knowledge_test_skills_check",
                description="Skills lookup",
                capability=ToolCapability.KNOWLEDGE,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolDefinition(
                name="knowledge_test_search_support_tickets",
                description="Support ticket search",
                capability=ToolCapability.TICKETS,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
        ]
    )
    joined = "\n".join(lines)
    assert "available tool categories: knowledge, tickets." in joined.lower()
    assert "tickets tools for historical incidents" in joined.lower()
    assert "general knowledge search does not include support tickets" in joined.lower()
    assert "cluster name or cluster host" in joined
    assert "search_support_tickets" not in joined
    assert "get_support_ticket" not in joined


def _knowledge_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Knowledge lookup",
        capability=ToolCapability.KNOWLEDGE,
        parameters={"type": "object", "properties": {}, "required": []},
    )


def test_knowledge_guidance_lists_sources_when_multiple_providers():
    lines = _tool_instruction_lines_for_categories(
        [
            _knowledge_tool("knowledge_12d8f6_search"),
            _knowledge_tool("mcp_atlassian_rovo_mcp_12e3dc_searchAtlassian"),
        ]
    )
    joined = "\n".join(lines)
    assert "2 distinct sources" in joined
    assert "query at least one knowledge tool from a different source" in joined
    # Grouped under their provider prefixes, not a flat list.
    assert "knowledge_12d8f6: knowledge_12d8f6_search" in joined
    assert (
        "mcp_atlassian_rovo_mcp_12e3dc: mcp_atlassian_rovo_mcp_12e3dc_searchAtlassian"
        in joined
    )


def test_knowledge_guidance_omitted_for_single_source_multiple_ops():
    # A single provider registering several knowledge ops must NOT trip the
    # multi-source nudge (they share one provider prefix / one source).
    lines = _tool_instruction_lines_for_categories(
        [
            _knowledge_tool("knowledge_12d8f6_search"),
            _knowledge_tool("knowledge_12d8f6_ingest"),
            _knowledge_tool("knowledge_12d8f6_get_skill"),
        ]
    )
    joined = "\n".join(lines)
    assert "distinct sources" not in joined
    assert "different source" not in joined


def test_knowledge_guidance_groups_mixed_ops_without_op_filtering():
    # Mixed ops in one provider stay in one group; a second provider makes it
    # multi-source. Ops like ingest/get_skill are grouped, not filtered out.
    lines = _tool_instruction_lines_for_categories(
        [
            _knowledge_tool("knowledge_12d8f6_search"),
            _knowledge_tool("knowledge_12d8f6_ingest"),
            _knowledge_tool("knowledge_12d8f6_get_skill"),
            _knowledge_tool("mcp_atlassian_rovo_mcp_12e3dc_searchConfluenceUsingCql"),
        ]
    )
    joined = "\n".join(lines)
    assert "2 distinct sources" in joined
    assert (
        "knowledge_12d8f6: knowledge_12d8f6_search, knowledge_12d8f6_ingest, "
        "knowledge_12d8f6_get_skill" in joined
    )


def test_startup_context_omits_ticket_workflow_when_tickets_category_unavailable():
    lines = _tool_instruction_lines_for_categories(
        [
            ToolDefinition(
                name="knowledge_test_skills_check",
                description="Skills lookup",
                capability=ToolCapability.KNOWLEDGE,
                parameters={"type": "object", "properties": {}, "required": []},
            ),
        ]
    )
    joined = "\n".join(lines)
    assert "support-ticket workflow" not in joined.lower()


def test_chat_prompt_mentions_support_ticket_usage():
    prompt = CHAT_SYSTEM_PROMPT.lower()
    assert "tools are available" in prompt
    assert "tickets" in prompt
    assert "support tickets" in prompt
    assert "general knowledge search excludes support tickets" in prompt


def test_chat_prompt_requires_explicit_skill_retrieval_and_scope_evidence():
    prompt = CHAT_SYSTEM_PROMPT.lower()
    assert "inventory only" in prompt
    assert "`get_skill`" in CHAT_SYSTEM_PROMPT
    assert "health check skill" in prompt
    assert "response as satisfying a skill" in prompt
    assert "captured package contents" in prompt
    assert "hostname or hostname fragment is not enough" in prompt
    assert "exact live match" in prompt


def test_knowledge_prompt_mentions_support_tickets():
    prompt = KNOWLEDGE_SYSTEM_PROMPT.lower()
    assert "support ticket" in prompt
    assert "general knowledge search excludes support tickets" in prompt


def test_sre_prompt_mentions_support_ticket_usage():
    prompt = SRE_SYSTEM_PROMPT.lower()
    assert "category tools in your batch" in prompt
    assert "support tickets" in prompt
    assert "general knowledge search excludes support tickets" in prompt


def test_sre_prompt_requires_explicit_skill_retrieval_and_scope_evidence():
    prompt = SRE_SYSTEM_PROMPT.lower()
    assert "inventory only" in prompt
    assert "`get_skill`" in SRE_SYSTEM_PROMPT
    assert "health-check skill" in prompt
    assert "response as satisfying a skill" in prompt
    assert "captured evidence, not current live state" in prompt
    assert "resolve the target before making live-state claims" in prompt


def test_chat_prompt_includes_command_semantics_guardrails():
    prompt = CHAT_SYSTEM_PROMPT.lower()
    assert "do not infer connection counts from `memory stats`".lower() in prompt
    assert "`info clients`" in prompt
    assert "`client list`" in prompt
    assert "clients.normal" in prompt


def test_sre_prompt_includes_command_semantics_guardrails():
    prompt = SRE_SYSTEM_PROMPT.lower()
    assert "do not infer connection counts from `memory stats`".lower() in prompt
    assert "`info clients`" in prompt
    assert "`client list`" in prompt
    assert "clients.normal" in prompt


def test_chat_and_sre_prompts_share_guardrails_constant():
    shared = REDIS_COMMAND_SEMANTICS_GUARDRAILS.strip()
    assert shared in CHAT_SYSTEM_PROMPT
    assert shared in SRE_SYSTEM_PROMPT
