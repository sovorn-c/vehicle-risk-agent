"""Domain errors for review actions and report decisions."""


class ReviewError(Exception):
    """Base error for all review operations."""


class AssessmentNotFoundError(ReviewError):
    """Raised when target Assessment does not exist."""


class AssessmentNotReviewableError(ReviewError):
    """Raised when Assessment is not in AWAITING_REVIEW state."""


class DraftNotFoundError(ReviewError):
    """Raised when target ReportDraft does not exist for the specified run."""


class ReviewActionConflictError(ReviewError):
    """Raised when a ReviewAction already exists for the draft and cannot be overwritten."""


class ReinvestigationLimitReachedError(ReviewError):
    """Raised when reinvestigation is requested beyond the three-run limit."""
