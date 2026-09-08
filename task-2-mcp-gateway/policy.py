"""JSON-RPC wire parsing, token/role resolution, and the tool-authorization policy.

Kept separate from the HTTP layer in ``mcp_gateway.py`` so the rules can be reasoned
about - and unit tested - without a server in the loop.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

JSONRPC_VERSION: Final = "2.0"

# Standard JSON-RPC codes, plus the MCP gateway's own -32001.
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603
UNAUTHORIZED_TOOL_CALL: Final = -32001

# Tools whose name carries this prefix are privileged.
ADMIN_TOOL_PREFIX: Final = "admin_"

METHOD_TOOLS_CALL: Final = "tools/call"


class Role(str, Enum):
    """Roles a bearer token can map to."""

    ADMIN = "admin"
    VIEWER = "viewer"


class JsonRpcProtocolError(Exception):
    """A request that is not a well-formed JSON-RPC/MCP frame."""

    def __init__(self, code: int, message: str, request_id: Any = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.data = data


@dataclass(frozen=True)
class Principal:
    """The caller behind a bearer token.

    ``token_id`` is a short hash of the secret, safe to put in logs; the secret
    itself never leaves the registry.
    """

    token_id: str
    role: Role


@dataclass(frozen=True)
class JsonRpcRequest:
    """A validated JSON-RPC request frame."""

    method: str
    request_id: str | int | None
    params: dict[str, Any] | None
    is_notification: bool


@dataclass(frozen=True)
class Decision:
    """The outcome of the authorization check for one request."""

    allowed: bool
    reason: str
    tool_name: str | None = None


class TokenRegistry:
    """Maps bearer tokens to principals.

    A real deployment would resolve these against an IdP or a JWT signature; the
    lookup is behind this one class so that swap touches nothing else.
    """

    _DEFAULT_TOKENS: Final[dict[str, str]] = {
        "admin-token-abc123": Role.ADMIN.value,
        "viewer-token-xyz789": Role.VIEWER.value,
    }

    def __init__(self, tokens: Mapping[str, Role]) -> None:
        if not tokens:
            raise ValueError("TokenRegistry needs at least one token")
        self._tokens: dict[str, Principal] = {
            secret: Principal(token_id=self.fingerprint(secret), role=role) for secret, role in tokens.items()
        }

    @staticmethod
    def fingerprint(secret: str) -> str:
        """Short, non-reversible label for a token, for use in logs."""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_env(cls, raw: str | None = None) -> TokenRegistry:
        """Build from ``GATEWAY_TOKENS``, formatted as ``token:role,token:role``.

        Falls back to the documented demo tokens when the variable is unset, so
        the gateway is runnable straight after clone.
        """
        raw = raw if raw is not None else os.getenv("GATEWAY_TOKENS", "")
        if not raw.strip():
            return cls({secret: Role(role) for secret, role in cls._DEFAULT_TOKENS.items()})

        tokens: dict[str, Role] = {}
        for index, entry in enumerate(raw.split(","), start=1):
            entry = entry.strip()
            if not entry:
                continue
            secret, separator, role_name = entry.rpartition(":")
            if not separator or not secret:
                raise ValueError(f"GATEWAY_TOKENS entry {index} is not in 'token:role' form")
            try:
                tokens[secret] = Role(role_name.strip().lower())
            except ValueError as exc:
                known = ", ".join(role.value for role in Role)
                raise ValueError(
                    f"GATEWAY_TOKENS entry {index} has unknown role {role_name!r}; expected one of {known}"
                ) from exc
        return cls(tokens)

    def resolve(self, token: str) -> Principal | None:
        """Return the principal for a token, or None.

        Every entry is compared with ``compare_digest`` and the loop does not
        break early, so a caller cannot time the answer to learn a prefix.
        """
        match: Principal | None = None
        for secret, principal in self._tokens.items():
            if secrets.compare_digest(secret, token):
                match = principal
        return match


def extract_bearer_token(header_value: str | None) -> str:
    """Pull the token out of an ``Authorization: Bearer <token>`` header.

    Raises JsonRpcProtocolError so the caller can answer with 401 and a
    JSON-RPC body rather than an ad-hoc string.
    """
    if not header_value:
        raise JsonRpcProtocolError(UNAUTHORIZED_TOOL_CALL, "Missing Authorization header")

    scheme, separator, token = header_value.partition(" ")
    if not separator or scheme.lower() != "bearer":
        raise JsonRpcProtocolError(UNAUTHORIZED_TOOL_CALL, "Authorization header must use the Bearer scheme")

    token = token.strip()
    if not token:
        raise JsonRpcProtocolError(UNAUTHORIZED_TOOL_CALL, "Bearer token is empty")
    return token


def parse_jsonrpc(payload: Any) -> JsonRpcRequest:
    """Validate a decoded JSON body as a single JSON-RPC 2.0 request frame."""
    if isinstance(payload, list):
        # MCP dropped JSON-RPC batching in the 2025-06-18 revision.
        raise JsonRpcProtocolError(INVALID_REQUEST, "Batched requests are not supported by MCP")
    if not isinstance(payload, dict):
        raise JsonRpcProtocolError(INVALID_REQUEST, "Request body must be a JSON object")

    # Recover the id first, so protocol errors can still be correlated by the client.
    raw_id = payload.get("id", ...)
    request_id: str | int | None = None
    is_notification = raw_id is ...
    if not is_notification:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            # bool is an int subclass, and MCP forbids a null id.
            raise JsonRpcProtocolError(INVALID_REQUEST, "Request id must be a string or an integer")
        request_id = raw_id

    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcProtocolError(INVALID_REQUEST, "Missing or unsupported 'jsonrpc' version, expected '2.0'", request_id)

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcProtocolError(INVALID_REQUEST, "Missing or invalid 'method'", request_id)

    params = payload.get("params")
    if params is not None and not isinstance(params, dict):
        raise JsonRpcProtocolError(INVALID_PARAMS, "'params' must be a JSON object when present", request_id)

    return JsonRpcRequest(method=method, request_id=request_id, params=params, is_notification=is_notification)


def authorize(request: JsonRpcRequest, principal: Principal) -> Decision:
    """Apply the gateway's method-level policy to one request.

    ``tools/list`` and every non-tool method pass through untouched. ``tools/call``
    is inspected: a tool named ``admin_*`` requires the admin role.

    Raises JsonRpcProtocolError when a ``tools/call`` frame is malformed, since
    a call without a usable tool name cannot be authorized either way.
    """
    if request.method != METHOD_TOOLS_CALL:
        return Decision(allowed=True, reason=f"{request.method} is not subject to tool policy")

    params = request.params or {}
    tool_name = params.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise JsonRpcProtocolError(INVALID_PARAMS, "tools/call requires a non-empty string 'params.name'", request.request_id)

    if not tool_name.startswith(ADMIN_TOOL_PREFIX):
        return Decision(allowed=True, reason="tool is not privileged", tool_name=tool_name)

    if principal.role is Role.ADMIN:
        return Decision(allowed=True, reason="admin role satisfies the admin_ prefix", tool_name=tool_name)

    return Decision(
        allowed=False,
        reason=f"tool {tool_name!r} requires the '{Role.ADMIN.value}' role, caller has '{principal.role.value}'",
        tool_name=tool_name,
    )


def error_body(code: int, message: str, request_id: Any = None, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC error response envelope."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}
