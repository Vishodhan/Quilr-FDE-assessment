"""Strict input/output schemas for the customer-support MCP tools.

Every tool argument is validated here before any business logic runs, so a
malformed call fails at the protocol boundary instead of half-way through a
refund. The JSON Schemas advertised by ``tools/list`` are generated from these
same models, so the contract and the enforcement can never drift apart.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

# "CUST-XXXXX": the literal prefix plus exactly five uppercase alphanumerics.
CUSTOMER_ID_PATTERN = r"^CUST-[A-Z0-9]{5}$"

CustomerId = Annotated[str, StringConstraints(pattern=CUSTOMER_ID_PATTERN)]

# strict=True stops Pydantic from quietly turning "100" into 100.0 or 1 into True;
# extra="forbid" makes an unknown argument an error rather than something ignored.
_STRICT_INPUT = ConfigDict(extra="forbid", strict=True, frozen=True)


class GetCustomerRecordInput(BaseModel):
    """Arguments accepted by ``get_customer_record``."""

    model_config = _STRICT_INPUT

    customer_id: CustomerId = Field(description="Customer identifier in CUST-XXXXX form, e.g. CUST-10042.")


class TriggerRefundInput(BaseModel):
    """Arguments accepted by ``trigger_refund``."""

    model_config = _STRICT_INPUT

    customer_id: CustomerId = Field(description="Customer identifier in CUST-XXXXX form, e.g. CUST-10042.")
    amount: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Refund amount in USD. Must be a finite number greater than zero.",
    )
    reason: Annotated[str, StringConstraints(min_length=10, max_length=500)] = Field(
        description="Why the refund is being issued. At least 10 characters.",
    )

    @field_validator("reason")
    @classmethod
    def _reject_blank_reason(cls, value: str) -> str:
        """Ten spaces satisfies min_length but is not a reason."""
        if not value.strip():
            raise ValueError("reason must contain non-whitespace characters")
        return value


class CustomerRecord(BaseModel):
    """Shape returned by ``get_customer_record``."""

    customer_id: CustomerId
    name: str
    email: str
    plan: str
    status: str
    lifetime_value_usd: float
    refundable_balance_usd: float


class RefundReceipt(BaseModel):
    """Shape returned by ``trigger_refund``."""

    refund_id: str
    customer_id: CustomerId
    amount: float
    reason: str
    status: str
    remaining_refundable_usd: float


def format_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Flatten a ValidationError into JSON-safe entries for a JSON-RPC error `data` field.

    The offending input value is deliberately dropped: it may hold customer PII
    and error payloads tend to end up in logs.
    """
    return [
        {
            "field": ".".join(str(part) for part in error["loc"]) or "<root>",
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors(include_url=False)
    ]
