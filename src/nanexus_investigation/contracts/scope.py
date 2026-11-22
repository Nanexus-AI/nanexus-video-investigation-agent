"""Investigation scope and Evidence Grant. G3-A §5.1 and §6.2."""

from pydantic import Field, field_validator

from nanexus_investigation.contracts.identity import ContractModel, Interval


class InvestigationScope(ContractModel):
    """Camera, source, and time bounds for an investigation. G3-A §5.1."""

    vms_instance_id: str = Field(min_length=1)
    camera_ids: tuple[str, ...] = Field(min_length=1)
    time_bounds: Interval

    @field_validator("vms_instance_id")
    @classmethod
    def _reject_blank_instance(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("vms_instance_id must be non-empty and trimmed")
        return value

    @field_validator("camera_ids")
    @classmethod
    def _validate_cameras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("camera_ids must contain at least one camera")
        cleaned: list[str] = []
        seen: set[str] = set()
        for camera_id in value:
            if camera_id.strip() != camera_id or not camera_id.strip():
                raise ValueError("camera identifiers must be non-empty and trimmed")
            if camera_id in seen:
                raise ValueError(f"duplicate camera_id: {camera_id}")
            seen.add(camera_id)
            cleaned.append(camera_id)
        return tuple(cleaned)

    def contains_camera(self, camera_id: str) -> bool:
        return camera_id in self.camera_ids

    def contains_interval(self, interval: Interval) -> bool:
        """True when the interval lies within this scope's time bounds."""
        bounds = self.time_bounds
        if bounds.end_utc is None:
            return False
        if interval.start_utc < bounds.start_utc:
            return False
        if interval.end_utc is None:
            return False
        return interval.end_utc <= bounds.end_utc


class EvidenceGrant(ContractModel):
    """Recorded evidence-ID set for one Case. G3-A §6.2.

    Field names are an implementation choice. In the full system the grant is
    derived from the effective scope and validated tool results. Constructing
    this model, or placing an ID in ``evidence_ids``, is not itself
    authorization, VMS permission, or proof that the referenced evidence was
    validated. External permission checks remain with the Harness and Adapter.
    """

    case_id: str = Field(min_length=1)
    effective_scope: InvestigationScope
    evidence_ids: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("case_id")
    @classmethod
    def _reject_blank_case(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("case_id must be non-empty and trimmed")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def _reject_blank_evidence_ids(cls, value: frozenset[str]) -> frozenset[str]:
        for evidence_id in value:
            if evidence_id.strip() != evidence_id or not evidence_id.strip():
                raise ValueError("evidence_ids must be non-empty and trimmed")
        return value

    def admits(self, evidence_id: str) -> bool:
        """Return whether ``evidence_id`` is present in the recorded set.

        This is membership only. It is not an authorization decision and does
        not replace Evidence Engine or Output Gate checks.
        """
        return evidence_id in self.evidence_ids


def apply_requested_scope(
    *,
    authorized_scope: InvestigationScope,
    requested_scope: InvestigationScope,
) -> InvestigationScope:
    """Intersect a request with a previously authorized scope record.

    This pure helper rejects camera, source, or time expansion beyond
    ``authorized_scope``. It does not authenticate a caller or consult a
    live policy store. G3-A §5.1.
    """
    if requested_scope.vms_instance_id != authorized_scope.vms_instance_id:
        raise ValueError("requested scope cannot change the authorized VMS instance")

    unauthorized = [
        camera_id
        for camera_id in requested_scope.camera_ids
        if camera_id not in authorized_scope.camera_ids
    ]
    if unauthorized:
        raise ValueError(
            "requested scope expands cameras beyond authorization: "
            + ", ".join(unauthorized)
        )

    auth_bounds = authorized_scope.time_bounds
    req_bounds = requested_scope.time_bounds
    if auth_bounds.end_utc is None or req_bounds.end_utc is None:
        raise ValueError("scope intersection requires completed time bounds")
    if req_bounds.start_utc < auth_bounds.start_utc or req_bounds.end_utc > auth_bounds.end_utc:
        raise ValueError("requested scope expands time bounds beyond authorization")

    return InvestigationScope(
        vms_instance_id=authorized_scope.vms_instance_id,
        camera_ids=requested_scope.camera_ids,
        time_bounds=req_bounds,
    )
