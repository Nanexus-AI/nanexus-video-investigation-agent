"""Persistence integrity errors."""


class IdempotencyConflictError(ValueError):
    """Same operation identity with conflicting payload content."""


class PersistenceError(RuntimeError):
    """Generic Investigation Persistence failure."""


class CaseLifecycleError(PersistenceError):
    """Illegal or incomplete Case admission / lifecycle transition."""


class DomainFinalizationError(PersistenceError):
    """Terminal Case/report finalization rejected or rolled back."""
