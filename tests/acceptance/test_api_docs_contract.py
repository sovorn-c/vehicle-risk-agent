"""Acceptance contract for the branded public API documentation surface."""

from fastapi.testclient import TestClient

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings


def _client() -> TestClient:
    """Build an app with the MCP boundary explicitly unconfigured."""
    return TestClient(create_app(settings=Settings(mcp_server_url=None)))


def test_docs_is_a_branded_workflow_entrypoint() -> None:
    """The public docs page explains the auditable workflow and its boundaries."""
    with _client() as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for phrase in (
        "Vehicle Risk Assessment Agent",
        "Evidence → Policy → Risk → Review",
        "MCP-only evidence",
        "Policy citations",
        "Deterministic risk",
        "Human approval",
        "Synthetic demonstration data",
        "/openapi.json",
        "/reference",
    ):
        assert phrase in response.text
    assert '<main id="top">' in response.text
    assert 'aria-label="Documentation sections"' in response.text
    assert "prefers-reduced-motion" in response.text


def test_docs_explorer_is_openapi_backed_and_read_only_on_landing_page() -> None:
    """The explorer discovers the full contract but only executes GET operations."""
    with _client() as client:
        docs = client.get("/docs")
        schema = client.get("/openapi.json")

    assert schema.status_code == 200
    assert schema.json()["paths"]
    assert "Object.entries(schema.paths)" in docs.text
    assert "operation.method.toLowerCase() === 'get'" in docs.text
    assert "Authorization" not in docs.text
    assert "Try request" in docs.text
    assert "Copy cURL" in docs.text
    assert "escapeHtml" in docs.text


def test_reference_keeps_complete_swagger_fallback() -> None:
    """The complete built-in reference remains available for authenticated calls."""
    with _client() as client:
        reference = client.get("/reference")
        old_redoc = client.get("/redoc")

    assert reference.status_code == 200
    assert "swagger-ui" in reference.text.lower()
    assert "/openapi.json" in reference.text
    assert old_redoc.status_code == 404


def test_docs_contract_does_not_add_docs_routes_to_openapi() -> None:
    """Presentation-only routes stay out of the machine-readable API contract."""
    with _client() as client:
        schema = client.get("/openapi.json").json()

    assert "/docs" not in schema["paths"]
    assert "/reference" not in schema["paths"]
    assert "/openapi.json" not in schema["paths"]
    assert "/api/v1/assessments" in schema["paths"]
    assert "/api/v1/assessments/{assessment_id}/review/approve" in schema["paths"]
