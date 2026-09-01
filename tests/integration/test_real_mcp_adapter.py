"""Tests for the configured official MCP client adapter."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from mcp import types

from vehicle_risk_agent.adapters.mcp import (
    McpAdapterError,
    StreamableHttpVehicleMcpAdapter,
)
from vehicle_risk_agent.evidence.models import SafeErrorCategory


@pytest.mark.asyncio
async def test_streamable_http_adapter_validates_structured_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real adapter validates structured MCP output into local mirror models."""
    adapter = StreamableHttpVehicleMcpAdapter(
        server_url="http://mcp:8000/mcp",
        timeout_seconds=1,
        max_retries=0,
    )
    now = datetime.now(UTC).isoformat()
    result = types.CallToolResult(
        content=[],
        structured_content={
            "vin": "7AT0BK00X00000001",
            "revision_id": "rev-1",
            "revision_number": 1,
            "material_hash": "a" * 64,
            "canonical_fields": {"make": "HONDA"},
            "field_provenance": {},
            "conflicts": [],
            "confidence": {
                "score": 90,
                "band": "HIGH",
                "field_scores": {},
                "field_components": {},
                "rule_version": "v1",
                "explanation": "verified",
            },
            "as_of": now,
            "published_at": now,
        },
    )

    async def fake_call_once(_tool_name: str, _arguments: dict[str, Any]) -> types.CallToolResult:
        assert _tool_name == "lookup_vehicle"
        assert _arguments == {"vin": "7AT0BK00X00000001"}
        return result

    monkeypatch.setattr(adapter, "_call_once", fake_call_once)
    revision = await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert revision.revision_id == "rev-1"
    assert revision.canonical_fields["make"] == "HONDA"


@pytest.mark.asyncio
async def test_streamable_http_adapter_rejects_response_for_another_vin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid but request-mismatched MCP response is rejected."""
    adapter = StreamableHttpVehicleMcpAdapter(
        server_url="http://mcp:8000/mcp",
        timeout_seconds=1,
        max_retries=0,
    )
    now = datetime.now(UTC).isoformat()
    result = types.CallToolResult(
        content=[],
        structured_content={
            "vin": "1HGCM82633A004352",
            "revision_id": "rev-1",
            "revision_number": 1,
            "material_hash": "a" * 64,
            "canonical_fields": {},
            "confidence": {
                "score": 90,
                "band": "HIGH",
                "field_scores": {},
                "field_components": {},
                "rule_version": "v1",
                "explanation": "verified",
            },
            "as_of": now,
            "published_at": now,
        },
    )

    async def fake_call_once(_tool_name: str, _arguments: dict[str, Any]) -> types.CallToolResult:
        return result

    monkeypatch.setattr(adapter, "_call_once", fake_call_once)
    with pytest.raises(McpAdapterError) as exc_info:
        await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert exc_info.value.category == SafeErrorCategory.PIPELINE_CONTRACT_ERROR


@pytest.mark.asyncio
async def test_streamable_http_adapter_enforces_call_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hanging MCP call is bounded and projected as a retryable timeout."""
    adapter = StreamableHttpVehicleMcpAdapter(
        server_url="http://mcp:8000/mcp",
        timeout_seconds=0.01,
        max_retries=0,
    )

    async def slow_call_once(_tool_name: str, _arguments: dict[str, Any]) -> Any:
        await asyncio.sleep(1)
        return None

    monkeypatch.setattr(adapter, "_call_once", slow_call_once)
    with pytest.raises(McpAdapterError) as exc_info:
        await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert exc_info.value.category == SafeErrorCategory.PIPELINE_TIMEOUT
    assert exc_info.value.retryable


@pytest.mark.asyncio
async def test_streamable_http_adapter_projects_mcp_tool_errors_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP error content becomes a stable safe adapter error."""
    adapter = StreamableHttpVehicleMcpAdapter(
        server_url="http://mcp:8000/mcp",
        timeout_seconds=1,
        max_retries=0,
    )
    result = types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=(
                    '{"category":"VEHICLE_NOT_FOUND",'
                    '"message":"VIN secret must not escape",'
                    '"retryable":false,"remediation":"do not expose this"}'
                ),
            )
        ],
        is_error=True,
    )

    async def fake_call_once(_tool_name: str, _arguments: dict[str, Any]) -> types.CallToolResult:
        return result

    monkeypatch.setattr(adapter, "_call_once", fake_call_once)
    with pytest.raises(McpAdapterError) as exc_info:
        await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert exc_info.value.category == SafeErrorCategory.VEHICLE_NOT_FOUND
    assert "secret" not in exc_info.value.message
