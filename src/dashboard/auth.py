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

import json
import logging
import os
import secrets as _secrets
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import requests
from flask import abort, g, redirect, request, session

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


# ===================================================================== #
#  App-level Google OAuth (alternative to Cloudflare Access)
# ===================================================================== #
#  The app runs the Google sign-in flow itself -- no tunnel required.
#  Uses only `requests` + `pyjwt` (already deps); verifies Google's
#  signed ID token against Google's published certs, exactly like the
#  CF Access path above. Sign-in is restricted to one email domain.

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

_google_certs: dict = {}
_google_cert_lock = threading.Lock()


def _get_google_jwks() -> dict:
    """Fetch + cache Google's OAuth2 signing keys (1h TTL)."""
    now = time.time()
    with _google_cert_lock:
        cached = _google_certs.get("jwks")
        if cached and (now - _google_certs.get("at", 0)) < _CERT_CACHE_TTL_SEC:
            return cached
    with urlopen(_GOOGLE_CERTS_URL, timeout=10) as resp:
        jwks = json.loads(resp.read())
    with _google_cert_lock:
        _google_certs["jwks"] = jwks
        _google_certs["at"] = now
    return jwks


def verify_google_id_token(id_token: str, client_id: str) -> dict | None:
    """Verify a Google ID token (RS256) and return claims, or None."""
    if not id_token:
        return None
    try:
        import jwt  # type: ignore
    except ImportError:
        logger.error("Google OAuth needs PyJWT: pip install 'pyjwt[crypto]'")
        return None
    try:
        jwks = _get_google_jwks()
        kid = jwt.get_unverified_header(id_token).get("kid")
        key_data = next((k for k in jwks.get("keys", [])
                          if k.get("kid") == kid), None)
        if key_data is None:
            logger.warning("Google ID token kid %s not in JWKS", kid)
            return None
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
        claims = jwt.decode(id_token, public_key, algorithms=["RS256"],
                             audience=client_id)
        if claims.get("iss") not in _GOOGLE_ISSUERS:
            logger.warning("Google ID token bad issuer: %s",
                           claims.get("iss"))
            return None
        return claims
    except jwt.PyJWTError as e:
        logger.warning("Google ID token verification failed: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected Google ID token error: %s", e,
                     exc_info=True)
        return None


def _read_secret(value: str | None, env_var: str,
                 file_path: str | None) -> str | None:
    """A secret from (in order): explicit config value, env var, file."""
    if value:
        return str(value).strip()
    env = os.environ.get(env_var)
    if env:
        return env.strip()
    if file_path and os.path.isfile(file_path):
        with open(file_path, encoding="utf-8") as fh:
            return fh.read().strip()
    return None


def _load_session_secret(auth_cfg: dict) -> str:
    """Flask session signing key. Auto-generate + persist to
    secrets/session_secret on first run so logins survive restarts."""
    s = _read_secret(auth_cfg.get("session_secret"),
                     "QC_SESSION_SECRET", "secrets/session_secret")
    if s:
        return s
    os.makedirs("secrets", exist_ok=True)
    s = _secrets.token_urlsafe(48)
    with open("secrets/session_secret", "w", encoding="utf-8") as fh:
        fh.write(s)
    logger.info("Generated a new session secret at secrets/session_secret")
    return s


def register_google_oauth(server, store, config: dict) -> None:
    """Install the Google OAuth login flow + an auth gate on *server*.

    Requires ``auth.google_client_id`` and a client secret (via
    ``QC_GOOGLE_CLIENT_SECRET`` env or secrets/google_client_secret).
    Only ``auth.allowed_domain`` emails (default umn.edu) may sign in.
    """
    auth_cfg = (config or {}).get("auth", {}) or {}
    client_id = (auth_cfg.get("google_client_id") or "").strip()
    client_secret = _read_secret(auth_cfg.get("google_client_secret"),
                                 "QC_GOOGLE_CLIENT_SECRET",
                                 "secrets/google_client_secret")
    allowed_domain = (auth_cfg.get("allowed_domain")
                       or "umn.edu").lower().lstrip("@")
    redirect_base = (auth_cfg.get("oauth_redirect_base")
                      or "http://localhost:8050").rstrip("/")
    redirect_uri = f"{redirect_base}/auth/google/callback"

    if not client_id or not client_secret:
        raise RuntimeError(
            "auth.method=google_oauth needs google_client_id (config) "
            "and a client secret (QC_GOOGLE_CLIENT_SECRET env or "
            "secrets/google_client_secret). See the Google Cloud "
            "Console -> Credentials -> OAuth 2.0 Client ID.")

    server.secret_key = _load_session_secret(auth_cfg)
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=redirect_base.startswith("https"),
    )

    @server.route("/auth/google/login")
    def _google_login():  # pragma: no cover - exercised live
        state = _secrets.token_urlsafe(24)
        session["oauth_state"] = state
        session["oauth_next"] = request.args.get("next") or "/"
        params = urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
            "hd": allowed_domain,
        })
        return redirect(f"{_GOOGLE_AUTH_URL}?{params}")

    @server.route("/auth/google/callback")
    def _google_callback():  # pragma: no cover - exercised live
        if request.args.get("state") != session.pop("oauth_state", None):
            abort(400, "OAuth state mismatch")
        code = request.args.get("code")
        if not code:
            abort(400, "missing code")
        try:
            resp = requests.post(_GOOGLE_TOKEN_URL, timeout=10, data={
                "code": code, "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            })
        except requests.RequestException as e:
            logger.error("Google token exchange failed: %s", e)
            abort(502)
        if resp.status_code != 200:
            logger.warning("Google token endpoint %s: %s",
                           resp.status_code, resp.text[:200])
            abort(403)
        claims = verify_google_id_token(
            resp.json().get("id_token", ""), client_id)
        if not claims:
            abort(403)
        email = (claims.get("email") or "").lower()
        if (not claims.get("email_verified")
                or not email.endswith("@" + allowed_domain)):
            logger.info("Rejected Google login for %r (domain gate)", email)
            return (f"Access is limited to @{allowed_domain} accounts. "
                    f"You signed in as {email or 'an unknown account'}.",
                    403)
        session["email"] = email
        try:
            store.upsert_user(email)
        except Exception as e:
            logger.debug("upsert_user failed for %s: %s", email, e)
        return redirect(session.pop("oauth_next", "/") or "/")

    @server.route("/auth/logout")
    def _logout():  # pragma: no cover - exercised live
        session.clear()
        return redirect("/auth/google/login")

    @server.before_request
    def _google_gate():  # pragma: no cover - exercised live
        if request.path.startswith("/auth/"):
            return None
        email = session.get("email")
        if not email:
            # Full-page GETs -> bounce to Google. XHR (Dash callbacks)
            # -> 401 so the browser doesn't try to render a redirect.
            if request.method == "GET" and "text/html" in (
                    request.headers.get("Accept", "")):
                return redirect("/auth/google/login?next="
                                + quote(request.full_path or "/"))
            abort(401)
        try:
            role = store.get_user_role(email)
        except Exception:
            role = "reviewer"
        g.user = {"email": email, "role": role}
        return None


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
    method = (auth_cfg.get("method") or "cf_access").lower()

    if not enabled:
        logger.warning(
            "auth.enabled is false — dashboard is unauthenticated. "
            "This is acceptable only on a trusted LAN."
        )
        return

    # App-level Google sign-in. dev_bypass still short-circuits for
    # local development without any OAuth round-trip.
    if method == "google_oauth" and not dev_bypass:
        logger.info("Auth method: app-level Google OAuth")
        register_google_oauth(server, store, config)
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
