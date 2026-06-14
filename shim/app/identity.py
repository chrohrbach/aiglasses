"""Resolve a Plexus principal for a Rokid request and emit X-Plexus-* headers.

Plexus runs a Phase-2 per-user credential model: each MCP backend resolves the
*caller's own* downstream credential (their connected Gmail / Office / …) keyed
by the principal the gateway injects as `X-Plexus-Principal` / `X-Plexus-Email`.
The backend reads it via `shared.gateway_trust.extract_principal(...)` and looks
the credential up in the broker (`/<service>/users/<safe(email)>`).

The shim talks to mcp-hub directly on `mcp-network` (not through Caddy), and the
hub's middleware captures any inbound `X-Plexus-*` and forwards it to backends.
So the shim must inject these headers itself, mapping the Rokid `user_id` to a
Plexus principal.

Important: `extract_principal` returns ``None`` unless `X-Plexus-Principal` (the
``sub``) is present — sending only the email is not enough — so we always emit
the sub, defaulting it to the email.

Configuration (env):
    ROKID_PRINCIPAL_EMAIL      Default principal email to act as when a request
                               has no per-user mapping. For a single-user setup
                               this is all you need. Empty = no identity sent
                               (per-user MCP tools stay unauthenticated).
    ROKID_PRINCIPAL_SUB        Optional. Stable principal id (sub). Defaults to
                               the email — which is what the broker keys on.
    ROKID_PRINCIPAL_TYPE       Optional. Defaults to "user".
    ROKID_USER_PRINCIPAL_MAP   Optional JSON mapping a Rokid user_id to either a
                               bare email string or an object
                               {"email":..,"sub":..,"type":..}. Takes precedence
                               over the default for matching user_ids. Use this
                               when several Rokid users share one published agent
                               and must each reach their own mailbox.
"""

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Header names the gateway/hub trust model uses (see shared/gateway_trust.py).
H_PRINCIPAL = "X-Plexus-Principal"
H_TYPE = "X-Plexus-Principal-Type"
H_EMAIL = "X-Plexus-Email"


@dataclass(frozen=True)
class Principal:
    """A Plexus identity the shim acts as when calling MCP tools."""

    email: str = ""
    sub: str = ""
    type: str = "user"

    def headers(self) -> dict[str, str]:
        """X-Plexus-* headers the hub forwards to backends.

        Always emits the principal (sub) — backends drop the identity entirely
        when it is absent. Defaults sub to the email, which is the broker key.
        """
        sub = self.sub or self.email
        out: dict[str, str] = {}
        if sub:
            out[H_PRINCIPAL] = sub
        if self.type:
            out[H_TYPE] = self.type
        if self.email:
            out[H_EMAIL] = self.email
        return out


def _principal_from_entry(entry, *, default_type: str) -> Principal | None:
    """Build a Principal from a user-map entry (bare email str or object)."""
    if isinstance(entry, str):
        email = entry.strip()
        if not email:
            return None
        return Principal(email=email, sub=email, type=default_type)
    if isinstance(entry, dict):
        email = str(entry.get("email", "")).strip()
        sub = str(entry.get("sub", "")).strip() or email
        ptype = str(entry.get("type", "")).strip() or default_type
        if not (email or sub):
            return None
        return Principal(email=email, sub=sub, type=ptype)
    return None


def _load_user_map(raw: str, *, default_type: str) -> dict[str, Principal]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("ROKID_USER_PRINCIPAL_MAP is not valid JSON (%s) — ignoring", e)
        return {}
    if not isinstance(data, dict):
        logger.warning("ROKID_USER_PRINCIPAL_MAP must be a JSON object — ignoring")
        return {}
    out: dict[str, Principal] = {}
    for user_id, entry in data.items():
        p = _principal_from_entry(entry, default_type=default_type)
        if p is not None:
            out[str(user_id)] = p
        else:
            logger.warning("ROKID_USER_PRINCIPAL_MAP entry for %r is invalid — skipped", user_id)
    return out


class PrincipalResolver:
    """Maps a Rokid user_id to the Plexus principal to act as."""

    def __init__(
        self,
        *,
        default: Principal | None = None,
        user_map: dict[str, Principal] | None = None,
    ):
        self.default = default
        self.user_map = user_map or {}

    @property
    def configured(self) -> bool:
        return self.default is not None or bool(self.user_map)

    def for_user(self, user_id: str | None) -> Principal | None:
        if user_id and user_id in self.user_map:
            return self.user_map[user_id]
        return self.default


def resolver_from_env() -> PrincipalResolver:
    default_type = (os.environ.get("ROKID_PRINCIPAL_TYPE", "") or "user").strip() or "user"
    default_email = os.environ.get("ROKID_PRINCIPAL_EMAIL", "").strip()
    default_sub = os.environ.get("ROKID_PRINCIPAL_SUB", "").strip()
    default = None
    if default_email or default_sub:
        default = Principal(
            email=default_email,
            sub=default_sub or default_email,
            type=default_type,
        )
    user_map = _load_user_map(
        os.environ.get("ROKID_USER_PRINCIPAL_MAP", ""), default_type=default_type
    )
    resolver = PrincipalResolver(default=default, user_map=user_map)
    if resolver.configured:
        logger.info(
            "principal resolver active: default=%s mapped_users=%d",
            default.email if default else None,
            len(user_map),
        )
    else:
        logger.warning(
            "no Plexus principal configured (ROKID_PRINCIPAL_EMAIL / "
            "ROKID_USER_PRINCIPAL_MAP unset) — per-user MCP tools (Gmail, "
            "Office, …) will run without a caller identity and cannot resolve "
            "your account."
        )
    return resolver
