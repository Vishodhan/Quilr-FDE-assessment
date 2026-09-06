"""Customer-support MCP server speaking JSON-RPC over stdio.

Exposes two tools - ``get_customer_record`` and ``trigger_refund`` - behind
strict Pydantic validation.

Error model:
  * malformed arguments     -> JSON-RPC ``-32602 Invalid params``
  * unknown tool name       -> JSON-RPC ``-32602 Invalid params``
  * business failure        -> ``CallToolResult(isError=True)``
  * unexpected server fault -> JSON-RPC ``-32603 Internal error`` (sanitised)

That split is what the MCP spec asks for: protocol-level mistakes are JSON-RPC
errors, while a failure the model could reasonably react to (no such customer,
refund over the balance) comes back as a tool result the model gets to read.

Run it with ``python server.py``. stdout carries JSON-RPC frames and nothing else.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Final, TextIO
from uuid import uuid4

import anyio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from models import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
    format_validation_errors,
)
from pydantic import BaseModel, ValidationError

SERVER_NAME: Final = "customer-support-mcp"
SERVER_VERSION: Final = "1.0.0"

logger = logging.getLogger(SERVER_NAME)


# --------------------------------------------------------------------------- #
# stdio isolation
# --------------------------------------------------------------------------- #
def reserve_stdout_for_protocol() -> TextIO:
    """Hand the real stdout to the JSON-RPC transport and repoint fd 1 at stderr.

    Once this has run, anything in the process that writes to stdout - our code,
    a stray print(), a chatty dependency - lands on stderr instead of corrupting
    the protocol stream. The returned handle is the only remaining route to the
    original stdout, and only the transport is given it.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except OSError:  # stderr unavailable (detached process): keep fd 1 as it was
        logger.warning("Could not repoint fd 1 at stderr; relying on the Python-level guard alone")
    sys.stdout = sys.stderr  # also catches writes made through the Python object
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1, newline="\n")


def configure_logging() -> None:
    """Send every log record to stderr, at the level named by MCP_LOG_LEVEL."""
    level = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )


# --------------------------------------------------------------------------- #
# in-memory backing store
# --------------------------------------------------------------------------- #
_SEED_CUSTOMERS: Final[dict[str, dict[str, Any]]] = {
    "CUST-10042": {
        "customer_id": "CUST-10042",
        "name": "Ada Lovelace",
        "email": "ada.lovelace@example.com",
        "plan": "enterprise",
        "status": "active",
        "lifetime_value_usd": 18450.00,
        "refundable_balance_usd": 250.00,
    },
    "CUST-20099": {
        "customer_id": "CUST-20099",
        "name": "Grace Hopper",
        "email": "grace.hopper@example.com",
        "plan": "pro",
        "status": "active",
        "lifetime_value_usd": 3120.50,
        "refundable_balance_usd": 75.25,
    },
    "CUST-AB123": {
        "customer_id": "CUST-AB123",
        "name": "Alan Turing",
        "email": "alan.turing@example.com",
        "plan": "starter",
        "status": "churned",
        "lifetime_value_usd": 240.00,
        "refundable_balance_usd": 0.00,
    },
}

_customers: dict[str, dict[str, Any]] = copy.deepcopy(_SEED_CUSTOMERS)
_refund_ledger: list[dict[str, Any]] = []
_store_lock = asyncio.Lock()  # tools/call requests can be in flight concurrently


def reset_store() -> None:
    """Restore the seed data. Used by the test suite between cases."""
    global _customers
    _customers = copy.deepcopy(_SEED_CUSTOMERS)
    _refund_ledger.clear()


class ToolExecutionError(Exception):
    """A business rule said no. Surfaced as an errored tool result, not a JSON-RPC error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_payload(self) -> dict[str, Any]:
        """Machine-readable body the calling model can branch on."""
        return {"error": {"code": self.code, "message": self.message, **self.details}}


# --------------------------------------------------------------------------- #
# tool implementations
# --------------------------------------------------------------------------- #
async def get_customer_record(args: GetCustomerRecordInput) -> CustomerRecord:
    """Look up a single customer by ID."""
    async with _store_lock:
        record = _customers.get(args.customer_id)
    if record is None:
        raise ToolExecutionError(
            "customer_not_found",
            "No customer exists with id " + args.customer_id + ".",
            customer_id=args.customer_id,
        )
    return CustomerRecord(**record)


async def trigger_refund(args: TriggerRefundInput) -> RefundReceipt:
    """Issue a refund against the customer's refundable balance."""
    async with _store_lock:
        record = _customers.get(args.customer_id)
        if record is None:
            raise ToolExecutionError(
                "customer_not_found",
                "No customer exists with id " + args.customer_id + ".",
                customer_id=args.customer_id,
            )

        balance = record["refundable_balance_usd"]
        # Compare rounded: binary floats put 75.25 - 75.25 a hair either side of zero.
        if round(args.amount, 2) > round(balance, 2):
            raise ToolExecutionError(
                "refund_exceeds_balance",
                f"Refund of {args.amount:.2f} USD exceeds the refundable balance of {balance:.2f} USD.",
                customer_id=args.customer_id,
                requested_usd=round(args.amount, 2),
                refundable_balance_usd=round(balance, 2),
            )

        remaining = round(balance - args.amount, 2)
        record["refundable_balance_usd"] = remaining
        receipt = RefundReceipt(
            refund_id="RFND-" + uuid4().hex[:12].upper(),
            customer_id=args.customer_id,
            amount=round(args.amount, 2),
            reason=args.reason,
            status="accepted",
            remaining_refundable_usd=remaining,
        )
        _refund_ledger.append(receipt.model_dump())

    logger.info("Refund %s accepted for %s (%.2f USD)", receipt.refund_id, args.customer_id, args.amount)
    return receipt


# name -> (input model, coroutine), so dispatch is a lookup rather than a chain of ifs
_TOOL_REGISTRY: Final[dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[BaseModel]]]]] = {
    "get_customer_record": (GetCustomerRecordInput, get_customer_record),
    "trigger_refund": (TriggerRefundInput, trigger_refund),
}

TOOLS: Final[list[types.Tool]] = [
    types.Tool(
        name="get_customer_record",
        title="Get customer record",
        description="Fetch the account record for a customer by their CUST-XXXXX identifier.",
        inputSchema=GetCustomerRecordInput.model_json_schema(),
        outputSchema=CustomerRecord.model_json_schema(),
        annotations=types.ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    ),
    types.Tool(
        name="trigger_refund",
        title="Trigger refund",
        description="Issue a refund against a refundable balance. Requires a written reason.",
        inputSchema=TriggerRefundInput.model_json_schema(),
        outputSchema=RefundReceipt.model_json_schema(),
        annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
    ),
]


# --------------------------------------------------------------------------- #
# JSON-RPC handlers
# --------------------------------------------------------------------------- #
def _invalid_params(message: str, **data: Any) -> McpError:
    """Build a -32602 error, attaching structured detail when there is any."""
    return McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message, data=data or None))


def _tool_result(model: BaseModel) -> types.ServerResult:
    """Wrap a successful tool payload as both structured and text content."""
    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(type="text", text=model.model_dump_json(indent=2))],
            structuredContent=model.model_dump(mode="json"),
            isError=False,
        )
    )


def _tool_error_result(error: ToolExecutionError) -> types.ServerResult:
    """Wrap a business failure as an errored tool result.

    No structuredContent here: the error body deliberately does not match the
    tool's declared outputSchema.
    """
    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(error.as_payload(), indent=2))],
            isError=True,
        )
    )


async def handle_list_tools(_request: types.ListToolsRequest) -> types.ServerResult:
    """Advertise the two tools with the JSON Schemas generated from their models."""
    logger.debug("tools/list -> %s", [tool.name for tool in TOOLS])
    return types.ServerResult(types.ListToolsResult(tools=TOOLS))


async def handle_call_tool(request: types.CallToolRequest) -> types.ServerResult:
    """Validate arguments, run the tool, and map every failure to the right code.

    Registered straight onto ``request_handlers`` rather than through
    ``@server.call_tool()``: that decorator turns raised exceptions into an
    ``isError`` result, which would hide the -32602 codes this server must emit.
    """
    name = request.params.name
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        logger.warning("tools/call for unknown tool %r", name)
        raise _invalid_params("Unknown tool: " + repr(name), available_tools=sorted(_TOOL_REGISTRY))

    model, handler = entry
    raw_arguments = request.params.arguments if request.params.arguments is not None else {}
    if not isinstance(raw_arguments, dict):
        raise _invalid_params("params.arguments must be a JSON object")

    try:
        arguments = model.model_validate(raw_arguments)
    except ValidationError as exc:
        logger.warning("Rejected %s: %d schema violation(s)", name, exc.error_count())
        raise _invalid_params(
            "Invalid arguments for tool " + repr(name),
            validation_errors=format_validation_errors(exc),
        ) from exc

    try:
        result = await handler(arguments)
    except ToolExecutionError as exc:
        logger.info("Tool %s refused the call: %s", name, exc.code)
        return _tool_error_result(exc)
    except Exception as exc:  # never let an internal traceback reach the wire
        logger.exception("Unhandled error while running tool %s", name)
        raise McpError(
            types.ErrorData(code=types.INTERNAL_ERROR, message="Internal error while executing " + repr(name))
        ) from exc

    return _tool_result(result)


def build_server() -> Server:
    """Wire the handlers onto a low-level MCP server instance."""
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)
    server.request_handlers[types.ListToolsRequest] = handle_list_tools
    server.request_handlers[types.CallToolRequest] = handle_call_tool
    return server


async def serve() -> None:
    """Run the server until the client closes stdin."""
    protocol_stdout = reserve_stdout_for_protocol()
    server = build_server()
    logger.info("%s v%s ready on stdio", SERVER_NAME, SERVER_VERSION)
    try:
        async with stdio_server(stdout=anyio.wrap_file(protocol_stdout)) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(NotificationOptions()),
            )
    finally:
        protocol_stdout.close()
        logger.info("%s shutting down", SERVER_NAME)


def main() -> int:
    """Console entrypoint. Returns a process exit code."""
    configure_logging()
    try:
        anyio.run(serve)
    except (KeyboardInterrupt, EOFError):
        return 0
    except Exception:
        logger.exception("Fatal error; server exiting")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
