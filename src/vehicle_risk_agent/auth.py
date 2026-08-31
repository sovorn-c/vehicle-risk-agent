"""Authentication and Principal authorization contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.config import Settings


class Role(StrEnum):
    """Mutually exclusive Principal roles."""

    REQUESTER = "REQUESTER"
    REVIEWER = "REVIEWER"
    TECHNICAL_OPERATOR = "TECHNICAL_OPERATOR"
    POLICY_CORPUS_MAINTAINER = "POLICY_CORPUS_MAINTAINER"


class Principal(BaseModel):
    """Authenticated caller principal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str
    role: Role


def authenticate_bearer_token(token: str, settings: Settings) -> Principal | None:
    """Map a bearer token to a Principal with a single assigned role, or None if invalid."""
    if not token or not token.strip():
        return None

    clean_token = token.strip()
    if clean_token == settings.requester_token:
        return Principal(principal_id="principal-requester-1", role=Role.REQUESTER)
    if clean_token == settings.reviewer_token:
        return Principal(principal_id="principal-reviewer-1", role=Role.REVIEWER)
    if clean_token == settings.operator_token:
        return Principal(principal_id="principal-operator-1", role=Role.TECHNICAL_OPERATOR)
    if clean_token == settings.maintainer_token:
        return Principal(principal_id="principal-maintainer-1", role=Role.POLICY_CORPUS_MAINTAINER)

    return None
