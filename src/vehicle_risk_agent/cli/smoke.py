"""Live local-stack smoke verification through the public Agent API."""

# story: e07s02 e07s03

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase

_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_REQUESTER_TOKEN = "dev-requester-token"
_DEFAULT_REVIEWER_TOKEN = "dev-reviewer-token"
_CLEAN_VIN = "1HGCR2F85HA000000"
_RISKY_VIN = "1FA6P8CF8H5000000"
_UNKNOWN_VIN = "JM0BL10F000000000"
_TERMINAL_PHASES = {
    AssessmentRunPhase.COMPLETED.value,
    AssessmentRunPhase.INCOMPLETE.value,
    AssessmentRunPhase.FAILED.value,
}


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    expected_status: int,
    **kwargs: Any,
) -> dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    if response.status_code != expected_status:
        raise RuntimeError(f"{method} {path} returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} returned an invalid JSON object")
    return payload


async def _wait_until_terminal(
    client: httpx.AsyncClient,
    assessment_id: str,
    headers: dict[str, str],
    run_number: int = 1,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        payload = await _request(
            client,
            "GET",
            f"/api/v1/assessments/{assessment_id}",
            200,
            headers=headers,
        )
        runs = payload.get("runs", [])
        current_run = next(
            (run for run in runs if run.get("run_number") == run_number),
            None,
        )
        if isinstance(current_run, dict) and current_run.get("phase") in _TERMINAL_PHASES:
            return payload
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Assessment {assessment_id} did not reach a terminal phase")


async def _create_and_wait(
    client: httpx.AsyncClient,
    vin: str,
    sale_type: str,
    idempotency_key: str,
    requester_headers: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    payload = {"vin": vin, "context": {"sale_type": sale_type}}
    headers = {**requester_headers, "Idempotency-Key": idempotency_key}
    first = await _request(
        client,
        "POST",
        "/api/v1/assessments",
        201,
        headers=headers,
        json=payload,
    )
    second = await _request(
        client,
        "POST",
        "/api/v1/assessments",
        201,
        headers=headers,
        json=payload,
    )
    assessment_id = first.get("id")
    if not isinstance(assessment_id, str) or second.get("id") != assessment_id:
        raise RuntimeError("Assessment idempotency replay returned a different assessment")
    completed = await _wait_until_terminal(client, assessment_id, requester_headers)
    return assessment_id, completed


async def _assert_progress_stream(
    client: httpx.AsyncClient,
    assessment_id: str,
    headers: dict[str, str],
) -> None:
    async with client.stream(
        "GET",
        f"/api/v1/assessments/{assessment_id}/events?run_number=1",
        headers=headers,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"GET assessment events returned HTTP {response.status_code}")
        body = await response.aread()
    if b"event: progress" not in body or b"phase" not in body:
        raise RuntimeError("Assessment event stream did not contain persisted progress")


async def run_smoke(
    base_url: str = _DEFAULT_BASE_URL,
    requester_token: str = _DEFAULT_REQUESTER_TOKEN,
    reviewer_token: str = _DEFAULT_REVIEWER_TOKEN,
) -> dict[str, Any]:
    """Exercise seeded pipeline evidence, the Agent API, review, and SSE boundaries."""
    base_url = base_url.rstrip("/")
    requester_headers = {"Authorization": f"Bearer {requester_token}"}
    reviewer_headers = {"Authorization": f"Bearer {reviewer_token}"}

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        await _request(client, "GET", "/health", 200)
        await _request(client, "GET", "/ready", 200)

        clean_id, clean = await _create_and_wait(
            client, _CLEAN_VIN, "DEALER", "live-smoke-clean", requester_headers
        )
        risky_id, risky = await _create_and_wait(
            client, _RISKY_VIN, "PRIVATE", "live-smoke-risky", requester_headers
        )
        unknown_id, unknown = await _create_and_wait(
            client, _UNKNOWN_VIN, "AUCTION", "live-smoke-unknown", requester_headers
        )
        reinvest_id, _reinvest_run_one = await _create_and_wait(
            client, _CLEAN_VIN, "DEALER", "live-smoke-reinvest", requester_headers
        )

        clean_phase = clean["runs"][0]["phase"]
        risky_phase = risky["runs"][0]["phase"]
        unknown_phase = unknown["runs"][0]["phase"]
        if clean_phase != AssessmentRunPhase.COMPLETED.value:
            raise RuntimeError("Clean seeded vehicle did not complete")
        if risky_phase != AssessmentRunPhase.COMPLETED.value:
            raise RuntimeError("Risky seeded vehicle did not complete")
        if unknown_phase != AssessmentRunPhase.INCOMPLETE.value:
            raise RuntimeError("Unknown seeded vehicle did not withhold scoring")

        await _assert_progress_stream(client, clean_id, requester_headers)

        reinvestigation = await _request(
            client,
            "POST",
            f"/api/v1/assessments/{reinvest_id}/review/reinvestigate",
            200,
            headers={**reviewer_headers, "Idempotency-Key": "live-smoke-reinvestigate"},
            json={
                "run_number": 1,
                "rationale": "Verify the seeded vehicle again before release.",
                "questions": ["Confirm current register evidence."],
                "evidence_targets": ["ppsr_result"],
            },
        )
        if reinvestigation.get("next_run_number") != 2:
            raise RuntimeError("Reinvestigation did not allocate run 2")
        reinvested = await _wait_until_terminal(
            client, reinvest_id, requester_headers, run_number=2
        )
        if reinvested["runs"][1]["phase"] != AssessmentRunPhase.COMPLETED.value:
            raise RuntimeError("Reinvestigated seeded vehicle did not complete")
        await _request(
            client,
            "POST",
            f"/api/v1/assessments/{reinvest_id}/review/approve",
            200,
            headers={**reviewer_headers, "Idempotency-Key": "live-smoke-approve-reinvest"},
            json={"run_number": 2, "draft_outcome": "SCORED", "notes": "Smoke re-review"},
        )

        await _request(
            client,
            "POST",
            f"/api/v1/assessments/{clean_id}/review/approve",
            200,
            headers={**reviewer_headers, "Idempotency-Key": "live-smoke-approve"},
            json={"run_number": 1, "draft_outcome": "SCORED", "notes": "Smoke approval"},
        )
        await _request(
            client,
            "GET",
            f"/api/v1/assessments/{clean_id}/report",
            200,
            headers=requester_headers,
        )
        await _request(
            client,
            "POST",
            f"/api/v1/assessments/{risky_id}/review/reject",
            200,
            headers={**reviewer_headers, "Idempotency-Key": "live-smoke-reject"},
            json={"run_number": 1, "rationale": "Synthetic risk fixture is recorded as risky."},
        )
        await _request(
            client,
            "POST",
            f"/api/v1/assessments/{unknown_id}/review/approve",
            200,
            headers={**reviewer_headers, "Idempotency-Key": "live-smoke-approve-incomplete"},
            json={
                "run_number": 1,
                "draft_outcome": "INCOMPLETE",
                "acknowledge_missing_evidence": True,
                "rationale": "Synthetic fixture intentionally lacks required evidence.",
            },
        )

    return {
        "status": "success",
        "scenarios": ["clean", "risky", "unknown", "review", "sse", "idempotency"],
    }


async def _wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=3.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("Agent API did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live local-stack smoke verification")
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--requester-token", default=_DEFAULT_REQUESTER_TOKEN)
    parser.add_argument("--reviewer-token", default=_DEFAULT_REVIEWER_TOKEN)
    args = parser.parse_args()

    asyncio.run(_wait_for_health(args.base_url))
    result = asyncio.run(
        run_smoke(
            base_url=args.base_url,
            requester_token=args.requester_token,
            reviewer_token=args.reviewer_token,
        )
    )
    print(f"Live smoke verification result: {result['status']}")


if __name__ == "__main__":
    main()
