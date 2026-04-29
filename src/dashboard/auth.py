"""Cloudflare Access identity hook for the Dash app.

Cloudflare Access sits in front of the Cloudflare Tunnel and verifies the
user's Google/UMN login *before* traffic reaches our app. On every
authenticated request it injects two headers:

  * ``Cf-Access-Authenticated-User-Email``
  * ``Cf-Access-Jwt-Assertion``  (signed JWT)

We **must** verify the JWT signature against Cloudflare's public keys —
otherwise an attacker on the LAN could spoof the email header and bypass
SSO. Verified claims are stashed on ``flask.g.user`` so callbacks can
attribute writes (annotations, future review-form submissions) to the
real person.

A ``dev_bypass`` mode lets you run ``python main.py`` locally without
the tunnel during development; the configured ``dev_bypass_email`` is
used as the identity. **Never** enable ``dev_bypass`` on a host that is
publicly reachable.
"""

import logging
import threading
import time
from urllib.request import urlopen

from flask import abort, g, request

logger = logging.getLogger("qc_monitor.dashboard.auth")

# Cloudflare's public-key set has a 1h refresh window; we cache it.
_CERT_CACHE_TTL_SEC = 3600
_cert_cache: dict[str, dict] = {}
_cert_lock = threading.Lock()


def _get_cf_jwks(team_domain: str) -> dict:
    """Fetch and cache Cloudflare Access JWKS for *team_domain*."""
    assert isinstance(team_domain, str) and team_domain, "team_domain is required"

    now = time.time()
    with _cert_lock:
        cached = _cert_cache.get(team_domain)
        if cached and (now - cached["fetched_at"]) < _CERT_CACHE_TTL_SEC:
            return cached["jwks"]

    url = f"https://{team_domain}/cdn-cgi/access/certs"
    with urlopen(url, timeout=10) as resp:
        import json as _json
        jwks = _json.loads(resp.read())

    with _cert_lock:
        _cert_cache[team_domain] = {"jwks": jwks, "fetched_at": now}
    return jwks


def verify_cf_access_jwt(token: str, team_domain: str,
                         audience: str) -> dict | None:
    """Verify a Cloudflare Access JWT and return its claims, or None on failure.

    Imports PyJWT lazily so that ``auth.enabled: false`` deployments don't
    need the dependency installed.
    """
    if not token:
        return None
    try:
        import jwt  # type: ignore
    except ImportError:
        logger.error(
            "auth.enabled=true but PyJWT is not installed. "
            "Run: pip install 'pyjwt[crypto]>=2.8'"
        )
        return None
    try:
        jwks = _get_cf_jwks(team_domain)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        key_data = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
        if key_data is None:
            logger.warning("CF Access JWT kid %s not in JWKS", kid)
            return None
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=f"https://{team_domain}",
        )
        return claims
    except jwt.PyJWTError as e:
        logger.warning("CF Access JWT verification failed: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error verifying CF Access JWT: %s", e, exc_info=True)
        return None


def current_user_email() -> str | None:
    """Return the email of the currently authenticated user, if any."""
    user = getattr(g, "user", None)
    return user.get("email") if user else None


def register_auth(server, store, config: dict) -> None:
    """Install a Flask before_request hook on the Dash server.

    On every request:
      * If ``auth.enabled`` is False — pass through (legacy LAN-only mode).
      * Else if ``dev_bypass`` — set ``g.user`` to the bypass identity.
      * Else verify the CF Access JWT from the request header and set
        ``g.user``; on failure return 403.

    Static assets and the Dash callback dispatch path both go through this
    hook, so the entire app is gated.
    """
    auth_cfg = (config or {}).get("auth", {}) or {}
    enabled = bool(auth_cfg.get("enabled", False))
    dev_bypass = bool(auth_cfg.get("dev_bypass", False))
    dev_bypass_email = auth_cfg.get("dev_bypass_email") or "dev@localhost"
    team_domain = auth_cfg.get("cf_team_domain") or ""
    audience = auth_cfg.get("cf_audience") or ""

    if not enabled:
        logger.warning(
            "auth.enabled is false — dashboard is unauthenticated. "
            "This is acceptable only on a trusted LAN."
        )
        return

    if dev_bypass:
        logger.warning(
            "auth.dev_bypass=true — all requests will be authorized as %s. "
            "Disable this before exposing the dashboard.",
            dev_bypass_email,
        )

    if not dev_bypass and (not team_domain or not audience):
        raise RuntimeError(
            "auth.enabled=true requires cf_team_domain and cf_audience "
            "(or set dev_bypass: true for local development)."
        )

    @server.before_request
    def _cf_access_gate():  # pragma: no cover — exercised by integration only
        if dev_bypass:
            g.user = {"email": dev_bypass_email, "role": "mentor"}
            try:
                store.upsert_user(dev_bypass_email)
            except Exception as e:
                logger.debug("upsert_user failed in dev_bypass: %s", e)
            return None

        token = request.headers.get("Cf-Access-Jwt-Assertion", "")
        claims = verify_cf_access_jwt(token, team_domain, audience)
        if not claims:
            logger.info("Rejecting request to %s: no/invalid CF Access JWT",
                        request.path)
            abort(403)

        email = claims.get("email") or request.headers.get(
            "Cf-Access-Authenticated-User-Email", ""
        )
        if not email:
            logger.warning("CF Access JWT verified but no email claim")
            abort(403)

        try:
            store.upsert_user(email)
            role = store.get_user_role(email)
        except Exception as e:
            logger.error("upsert_user failed for %s: %s", email, e)
            role = "reviewer"

        g.user = {"email": email, "role": role, "claims": claims}
        return None
