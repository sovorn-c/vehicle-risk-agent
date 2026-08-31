"""Domain errors and exceptions."""


class DomainError(Exception):
    """Base class for all domain errors."""


class IdempotencyConflictError(DomainError):
    """Raised when an idempotency key is reused with a different request payload."""

    def __init__(
        self, key: str, message: str = "Idempotency key reused with different payload"
    ) -> None:
        super().__init__(message)
        self.key = key
