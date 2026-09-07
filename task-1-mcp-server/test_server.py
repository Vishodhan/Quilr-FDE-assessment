"""Tests for the customer-support MCP server.

Three layers, matching the three things the task is scored on:

1. ``TestValidationDepth``   - the Pydantic schemas, field by field and edge by edge.
2. ``TestStdioIsolation``    - stdout carries JSON-RPC frames and nothing else.
3. ``TestProtocolCompliance``- correct JSON-RPC error codes and execution flow,
   checked both over the raw wire and through the official MCP client.

Run from this directory: ``pytest -v``
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from pydantic import ValidationError

SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

import server  # noqa: E402
from models import GetCustomerRecordInput, TriggerRefundInput  # noqa: E402

SUBPROCESS_TIMEOUT = 60


# --------------------------------------------------------------------------- #
# raw JSON-RPC helpers
# --------------------------------------------------------------------------- #
def _frame(method: str, params: dict[str, Any] | None = None, request_id: int | None = None) -> str:
    """Serialise one JSON-RPC frame as the single line stdio transport expects."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def _call(tool: str, arguments: dict[str, Any], request_id: int) -> str:
    """Serialise a tools/call frame."""
    return _frame("tools/call", {"name": tool, "arguments": arguments}, request_id)


def _parse_stdout_line(line: str) -> dict[str, Any] | None:
    """Parse one stdout line, failing the test if it is not JSON-RPC."""
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Non-JSON-RPC line on stdout: {line!r}") from exc


def _drive_server(frames: list[str]) -> tuple[list[dict[str, Any]], str]:
    """Run the server as a subprocess, feed it frames, and return (stdout frames, stderr).

    stdin is held open until every expected reply has arrived. The SDK cancels
    in-flight handlers the instant the transport closes, so a client that hangs
    up early throws away its own answers. stderr goes to a temp file rather than
    a pipe so a chatty server cannot deadlock against a full pipe buffer.
    """
    outgoing = [
        _frame(
            "initialize",
            {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest-harness", "version": "1.0.0"},
            },
            request_id=1,
        ),
        _frame("notifications/initialized"),
        *frames,
    ]
    expected_ids = {message["id"] for message in map(json.loads, outgoing) if "id" in message}

    received: list[dict[str, Any]] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errlog:
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errlog,
            cwd=str(SERVER_DIR),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        watchdog = threading.Timer(SUBPROCESS_TIMEOUT, process.kill)
        watchdog.start()
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write("\n".join(outgoing) + "\n")
            process.stdin.flush()

            seen: set[Any] = set()
            while not expected_ids.issubset(seen):
                line = process.stdout.readline()
                if not line:  # server exited before answering everything
                    break
                frame = _parse_stdout_line(line)
                if frame is None:
                    continue
                received.append(frame)
                if "id" in frame:
                    seen.add(frame["id"])

            process.stdin.close()  # answers are in, let the server shut down
            for line in process.stdout:
                frame = _parse_stdout_line(line)
                if frame is not None:
                    received.append(frame)
            process.wait(timeout=SUBPROCESS_TIMEOUT)
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for pipe in (process.stdin, process.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()

        errlog.seek(0)
        return received, errlog.read()


def _by_id(frames: list[dict[str, Any]], request_id: int) -> dict[str, Any]:
    """Pull the single response carrying the given request id."""
    matches = [frame for frame in frames if frame.get("id") == request_id]
    assert matches, f"no response for id {request_id}; got ids {[f.get('id') for f in frames]}"
    assert len(matches) == 1, f"duplicate responses for id {request_id}"
    return matches[0]


# --------------------------------------------------------------------------- #
# 1. validation depth
# --------------------------------------------------------------------------- #
class TestValidationDepth:
    """Field-level edge cases on the two input schemas."""

    @pytest.mark.parametrize("customer_id", ["CUST-10042", "CUST-00000", "CUST-AB123", "CUST-ZZZZZ"])
    def test_accepts_well_formed_ids(self, customer_id: str) -> None:
        assert GetCustomerRecordInput(customer_id=customer_id).customer_id == customer_id

    @pytest.mark.parametrize(
        "customer_id",
        [
            "cust-10042",  # lowercase prefix
            "CUST-1004",  # too short
            "CUST-100422",  # too long
            "CUST_10042",  # wrong separator
            "10042",  # no prefix
            " CUST-10042",  # leading whitespace
            "CUST-10042 ",  # trailing whitespace
            "CUST-1004!",  # non-alphanumeric body
            "CUST-abcde",  # lowercase body
            "",  # empty
            "CUST-10042\nCUST-20099",  # newline injection
        ],
    )
    def test_rejects_malformed_ids(self, customer_id: str) -> None:
        with pytest.raises(ValidationError):
            GetCustomerRecordInput(customer_id=customer_id)

    @pytest.mark.parametrize("value", [123, None, 12.5, True, ["CUST-10042"], {"id": "CUST-10042"}])
    def test_rejects_non_string_ids(self, value: Any) -> None:
        """strict=True means no silent coercion of ints or bools into strings."""
        with pytest.raises(ValidationError):
            GetCustomerRecordInput(customer_id=value)

    def test_rejects_unknown_arguments(self) -> None:
        with pytest.raises(ValidationError):
            GetCustomerRecordInput(customer_id="CUST-10042", debug=True)

    def test_rejects_missing_argument(self) -> None:
        with pytest.raises(ValidationError):
            GetCustomerRecordInput()

    @pytest.mark.parametrize("amount", [0.01, 1, 99.99, 1_000_000.0])
    def test_accepts_positive_finite_amounts(self, amount: float) -> None:
        """Ints are accepted for a float field - JSON has no separate int type."""
        model = TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="duplicate charge")
        assert model.amount == float(amount)

    @pytest.mark.parametrize(
        "amount",
        [0, -0.01, -50, float("nan"), float("inf"), float("-inf"), "50", "abc", None, True],
    )
    def test_rejects_invalid_amounts(self, amount: Any) -> None:
        with pytest.raises(ValidationError):
            TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="duplicate charge")

    @pytest.mark.parametrize("reason", ["0123456789", "customer was double charged in March"])
    def test_accepts_reasons_of_at_least_ten_characters(self, reason: str) -> None:
        assert TriggerRefundInput(customer_id="CUST-10042", amount=5, reason=reason).reason == reason

    @pytest.mark.parametrize(
        "reason",
        [
            "",  # empty
            "too short",  # 9 characters
            "         ",  # 9 spaces
            "          ",  # 10 spaces: long enough, but no content
            "\t\n\r\f\v \t\n\r\f",  # 10 whitespace characters
            "x" * 501,  # over the upper bound
            42,  # not a string
            None,
        ],
    )
    def test_rejects_invalid_reasons(self, reason: Any) -> None:
        with pytest.raises(ValidationError):
            TriggerRefundInput(customer_id="CUST-10042", amount=5, reason=reason)

    @pytest.mark.parametrize("amount", [10.005, 0.004, 250.001, 1.2345, 0.0001])
    def test_rejects_sub_cent_amounts(self, amount: float) -> None:
        """Money has two decimal places. Anything finer is a rounding error, not an amount."""
        with pytest.raises(ValidationError):
            TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="duplicate charge")

    @pytest.mark.parametrize("amount", [10, 10.0, 10.5, 10.55, 0.01, 1_000_000.99])
    def test_accepts_whole_cent_amounts(self, amount: float) -> None:
        model = TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="duplicate charge")
        assert model.amount == float(amount)

    def test_reason_boundary_is_exactly_ten(self) -> None:
        """Nine characters fails, ten passes - the boundary is where it is documented."""
        with pytest.raises(ValidationError):
            TriggerRefundInput(customer_id="CUST-10042", amount=5, reason="123456789")
        assert TriggerRefundInput(customer_id="CUST-10042", amount=5, reason="1234567890").reason == "1234567890"

    def test_inputs_are_frozen(self) -> None:
        """Validated arguments must not be mutable after the fact."""
        model = GetCustomerRecordInput(customer_id="CUST-10042")
        with pytest.raises(ValidationError):
            model.customer_id = "CUST-20099"

    def test_advertised_schema_matches_the_model(self) -> None:
        """tools/list must advertise the same constraints the server enforces."""
        schema = TriggerRefundInput.model_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["customer_id"]["pattern"] == r"^CUST-[A-Z0-9]{5}$"
        assert schema["properties"]["amount"]["exclusiveMinimum"] == 0
        assert schema["properties"]["reason"]["minLength"] == 10
        assert set(schema["required"]) == {"customer_id", "amount", "reason"}


@pytest.mark.asyncio
class TestRefundArithmetic:
    """The balance check and the debit must be the same number.

    Exercised in-process rather than over the wire, because the interesting cases
    are arithmetic rather than protocol.
    """

    @pytest.fixture(autouse=True)
    def fresh_store(self) -> None:
        """Every case starts from the seed balances."""
        server.reset_store()

    async def test_cannot_overdraw_a_zero_balance(self) -> None:
        """CUST-AB123 has 0.00 refundable. Nothing may be issued against it."""
        with pytest.raises(server.ToolExecutionError) as exc:
            await server.trigger_refund(
                TriggerRefundInput(customer_id="CUST-AB123", amount=0.01, reason="goodwill credit")
            )
        assert exc.value.code == "refund_exceeds_balance"
        assert server._customers["CUST-AB123"]["refundable_balance_usd"] == 0.0

    async def test_cannot_exceed_the_balance_by_one_cent(self) -> None:
        with pytest.raises(server.ToolExecutionError) as exc:
            await server.trigger_refund(
                TriggerRefundInput(customer_id="CUST-10042", amount=250.01, reason="please refund me")
            )
        assert exc.value.code == "refund_exceeds_balance"
        assert server._customers["CUST-10042"]["refundable_balance_usd"] == 250.00

    async def test_exact_balance_refund_leaves_a_clean_zero(self) -> None:
        """0.0, not -0.0: comparison and debit both happen in whole cents."""
        receipt = await server.trigger_refund(
            TriggerRefundInput(customer_id="CUST-20099", amount=75.25, reason="full refund issued")
        )
        assert receipt.remaining_refundable_usd == 0.0
        assert math.copysign(1.0, receipt.remaining_refundable_usd) == 1.0
        assert math.copysign(1.0, server._customers["CUST-20099"]["refundable_balance_usd"]) == 1.0

    async def test_receipt_ledger_and_balance_all_agree(self) -> None:
        receipt = await server.trigger_refund(
            TriggerRefundInput(customer_id="CUST-10042", amount=100.25, reason="shipping never arrived")
        )
        assert receipt.amount == 100.25
        assert receipt.remaining_refundable_usd == 149.75
        assert server._customers["CUST-10042"]["refundable_balance_usd"] == 149.75
        assert server._refund_ledger[-1]["amount"] == 100.25

    async def test_a_sub_cent_request_is_refused_even_if_validation_is_bypassed(self) -> None:
        """Defence in depth: the tool refuses it, not just the schema.

        model_construct skips validation, standing in for a caller that reached the
        business layer some other way.
        """
        sneaky = TriggerRefundInput.model_construct(
            customer_id="CUST-AB123", amount=0.004, reason="sub-cent probe"
        )
        with pytest.raises(server.ToolExecutionError) as exc:
            await server.trigger_refund(sneaky)
        assert exc.value.code == "amount_below_minimum"
        assert server._customers["CUST-AB123"]["refundable_balance_usd"] == 0.0

    async def test_repeated_refunds_cannot_drive_the_balance_negative(self) -> None:
        refunded = 0.0
        for _ in range(20):
            try:
                receipt = await server.trigger_refund(
                    TriggerRefundInput(customer_id="CUST-20099", amount=10.00, reason="incremental refund")
                )
            except server.ToolExecutionError:
                break
            refunded += receipt.amount

        balance = server._customers["CUST-20099"]["refundable_balance_usd"]
        assert balance >= 0.0
        assert refunded + balance == 75.25  # nothing created or destroyed


# --------------------------------------------------------------------------- #
# 2. stdio isolation
# --------------------------------------------------------------------------- #
class TestStdioIsolation:
    """stdout must carry JSON-RPC and nothing else."""

    def test_stdout_is_pure_json_rpc(self) -> None:
        """Every stdout line parses as a JSON-RPC 2.0 frame across a full session."""
        frames, _ = _drive_server(
            [
                _frame("tools/list", {}, 2),
                _call("get_customer_record", {"customer_id": "CUST-10042"}, 3),
                _call("get_customer_record", {"customer_id": "bad"}, 4),
            ]
        )
        assert frames, "server produced no output"
        for frame in frames:
            assert frame.get("jsonrpc") == "2.0", f"frame is not JSON-RPC 2.0: {frame}"
            assert ("result" in frame) or ("error" in frame) or ("method" in frame), f"malformed frame: {frame}"

    def test_logs_go_to_stderr(self) -> None:
        """The startup banner proves logging is wired to stderr, not stdout."""
        _, stderr = _drive_server([_frame("tools/list", {}, 2)])
        assert "ready on stdio" in stderr

    def test_stray_print_cannot_reach_stdout(self) -> None:
        """After the fd swap, print() and sys.stdout writes land on stderr."""
        probe = (
            "import sys; sys.path.insert(0, " + repr(str(SERVER_DIR)) + ");"
            "from server import reserve_stdout_for_protocol;"
            "protocol = reserve_stdout_for_protocol();"
            "print('POISON-VIA-PRINT');"
            "sys.stdout.write('POISON-VIA-WRITE\\n');"
            "sys.stdout.flush();"
            "protocol.write('{\"jsonrpc\": \"2.0\"}\\n'); protocol.flush()"
        )

        process = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        assert "POISON" not in process.stdout, f"stray writes reached stdout: {process.stdout!r}"
        assert "POISON-VIA-PRINT" in process.stderr
        assert "POISON-VIA-WRITE" in process.stderr
        assert process.stdout.strip() == '{"jsonrpc": "2.0"}'

    def test_writing_to_stdout_during_a_session_does_not_corrupt_the_stream(self) -> None:
        """A tool call that also logs still yields parseable frames only."""
        frames, stderr = _drive_server(
            [_call("trigger_refund", {"customer_id": "CUST-10042", "amount": 10, "reason": "goodwill credit"}, 2)]
        )
        assert "Refund RFND-" in stderr  # the tool logged, and it logged to stderr
        assert all(frame.get("jsonrpc") == "2.0" for frame in frames)


# --------------------------------------------------------------------------- #
# 3. protocol compliance
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def session() -> list[dict[str, Any]]:
    """One server run exercising every branch; each test below reads its own id."""
    frames, _ = _drive_server(
        [
            _frame("tools/list", {}, 10),
            _call("get_customer_record", {"customer_id": "CUST-10042"}, 11),
            _call("get_customer_record", {"customer_id": "cust-10042"}, 12),
            _call("trigger_refund", {"customer_id": "CUST-10042", "amount": "50", "reason": "double charged"}, 13),
            _call("no_such_tool", {}, 14),
            _call("trigger_refund", {"customer_id": "CUST-20099", "amount": 25.25, "reason": "duplicate charge"}, 15),
            _call("get_customer_record", {"customer_id": "CUST-99999"}, 16),
            _call("trigger_refund", {"customer_id": "CUST-AB123", "amount": 500, "reason": "please refund me"}, 17),
            _call("trigger_refund", {"customer_id": "CUST-10042", "amount": 5, "reason": "ok", "force": True}, 18),
            _call("trigger_refund", {"customer_id": "CUST-10042", "amount": 5, "reason": "          "}, 19),
            _call("get_customer_record", {}, 20),
        ]
    )
    return frames


class TestProtocolCompliance:
    """JSON-RPC error mapping and execution flow, checked on the wire."""

    def test_initialize_succeeds(self, session: list[dict[str, Any]]) -> None:
        result = _by_id(session, 1)["result"]
        assert result["serverInfo"]["name"] == "customer-support-mcp"
        assert "tools" in result["capabilities"]

    def test_tools_list_advertises_exactly_two_tools(self, session: list[dict[str, Any]]) -> None:
        tools = _by_id(session, 10)["result"]["tools"]
        assert {tool["name"] for tool in tools} == {"get_customer_record", "trigger_refund"}
        for tool in tools:
            assert tool["inputSchema"]["type"] == "object"
            assert tool["inputSchema"]["additionalProperties"] is False

    def test_valid_lookup_returns_structured_content(self, session: list[dict[str, Any]]) -> None:
        result = _by_id(session, 11)["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["name"] == "Ada Lovelace"

    def test_malformed_customer_id_is_invalid_params(self, session: list[dict[str, Any]]) -> None:
        error = _by_id(session, 12)["error"]
        assert error["code"] == types.INVALID_PARAMS  # -32602
        assert error["data"]["validation_errors"][0]["field"] == "customer_id"

    def test_wrong_type_for_amount_is_invalid_params(self, session: list[dict[str, Any]]) -> None:
        """A JSON string where a number belongs is rejected, not coerced."""
        error = _by_id(session, 13)["error"]
        assert error["code"] == types.INVALID_PARAMS
        assert any(item["field"] == "amount" for item in error["data"]["validation_errors"])

    def test_unknown_tool_is_invalid_params(self, session: list[dict[str, Any]]) -> None:
        error = _by_id(session, 14)["error"]
        assert error["code"] == types.INVALID_PARAMS
        assert error["data"]["available_tools"] == ["get_customer_record", "trigger_refund"]

    def test_valid_refund_succeeds_and_debits_the_balance(self, session: list[dict[str, Any]]) -> None:
        result = _by_id(session, 15)["result"]
        assert result["isError"] is False
        receipt = result["structuredContent"]
        assert receipt["status"] == "accepted"
        assert receipt["refund_id"].startswith("RFND-")
        assert receipt["remaining_refundable_usd"] == 50.0  # 75.25 - 25.25

    def test_unknown_customer_is_a_tool_result_not_a_protocol_error(self, session: list[dict[str, Any]]) -> None:
        """A well-formed request for a missing record is a business failure, not -32602."""
        frame = _by_id(session, 16)
        assert "error" not in frame
        assert frame["result"]["isError"] is True
        assert json.loads(frame["result"]["content"][0]["text"])["error"]["code"] == "customer_not_found"

    def test_refund_over_balance_is_a_tool_result(self, session: list[dict[str, Any]]) -> None:
        frame = _by_id(session, 17)
        assert "error" not in frame
        assert frame["result"]["isError"] is True
        assert json.loads(frame["result"]["content"][0]["text"])["error"]["code"] == "refund_exceeds_balance"

    def test_extra_argument_is_rejected(self, session: list[dict[str, Any]]) -> None:
        error = _by_id(session, 18)["error"]
        assert error["code"] == types.INVALID_PARAMS
        assert any(item["field"] == "force" for item in error["data"]["validation_errors"])

    def test_whitespace_only_reason_is_rejected(self, session: list[dict[str, Any]]) -> None:
        error = _by_id(session, 19)["error"]
        assert error["code"] == types.INVALID_PARAMS
        assert any(item["field"] == "reason" for item in error["data"]["validation_errors"])

    def test_missing_arguments_are_rejected(self, session: list[dict[str, Any]]) -> None:
        error = _by_id(session, 20)["error"]
        assert error["code"] == types.INVALID_PARAMS

    def test_errors_never_echo_the_offending_input(self, session: list[dict[str, Any]]) -> None:
        """Error payloads carry field, message and type - never the value, which may be PII."""
        for request_id in (12, 13, 18, 19):
            for item in _by_id(session, request_id)["error"]["data"]["validation_errors"]:
                assert set(item) == {"field", "message", "type"}


# --------------------------------------------------------------------------- #
# 4. interoperability with the official MCP client
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_official_client_can_drive_the_server() -> None:
    """The real SDK client sees the same results and the same error codes."""
    params = StdioServerParameters(command=sys.executable, args=["server.py"], cwd=str(SERVER_DIR))

    async with stdio_client(params) as (read_stream, write_stream), ClientSession(read_stream, write_stream) as session:
        init = await session.initialize()
        assert init.serverInfo.name == "customer-support-mcp"

        listing = await session.list_tools()
        assert {tool.name for tool in listing.tools} == {"get_customer_record", "trigger_refund"}

        ok = await session.call_tool("get_customer_record", {"customer_id": "CUST-10042"})
        assert ok.isError is False
        assert ok.structuredContent is not None
        assert ok.structuredContent["email"] == "ada.lovelace@example.com"

        with pytest.raises(McpError) as invalid_id:
            await session.call_tool("get_customer_record", {"customer_id": "not-an-id"})
        assert invalid_id.value.error.code == types.INVALID_PARAMS

        with pytest.raises(McpError) as short_reason:
            await session.call_tool(
                "trigger_refund", {"customer_id": "CUST-10042", "amount": 5.0, "reason": "short"}
            )
        assert short_reason.value.error.code == types.INVALID_PARAMS

        refund = await session.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-10042", "amount": 12.5, "reason": "shipping was never delivered"},
        )
        assert refund.isError is False
        assert refund.structuredContent is not None
        assert refund.structuredContent["remaining_refundable_usd"] == 237.5
