"""VMS capability snapshot. G3-C2 §7.2, G3-A §5.1."""

from pydantic import Field, field_validator

from nanexus_investigation.contracts.enums import CapabilityAvailability
from nanexus_investigation.contracts.identity import ContractModel


class OperationCapability(ContractModel):
    """Availability and permission for one adapter operation."""

    operation: str = Field(min_length=1)
    availability: CapabilityAvailability
    authorized_camera_ids: frozenset[str] | None = None

    @field_validator("operation")
    @classmethod
    def _reject_blank_operation(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("operation must be non-empty and trimmed")
        return value

    @field_validator("authorized_camera_ids")
    @classmethod
    def _reject_blank_cameras(
        cls, value: frozenset[str] | None
    ) -> frozenset[str] | None:
        if value is None:
            return value
        for camera_id in value:
            if camera_id.strip() != camera_id or not camera_id.strip():
                raise ValueError("camera identifiers must be non-empty and trimmed")
        return value


class CapabilitySnapshot(ContractModel):
    """Source-neutral capability description. G3-C2 §7.2.

    Availability and permission remain distinct. A globally supported operation
    may still be unauthorized for a particular camera.
    """

    vms_instance_id: str = Field(min_length=1)
    operations: tuple[OperationCapability, ...] = Field(min_length=1)

    @field_validator("vms_instance_id")
    @classmethod
    def _reject_blank_instance(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("vms_instance_id must be non-empty and trimmed")
        return value

    @field_validator("operations")
    @classmethod
    def _unique_operations(
        cls, value: tuple[OperationCapability, ...]
    ) -> tuple[OperationCapability, ...]:
        seen: set[str] = set()
        for item in value:
            if item.operation in seen:
                raise ValueError(f"duplicate operation: {item.operation}")
            seen.add(item.operation)
        return value

    def for_operation(self, operation: str) -> OperationCapability | None:
        for item in self.operations:
            if item.operation == operation:
                return item
        return None
