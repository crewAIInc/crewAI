"""Centralized OpenAI credential resolution with Codex OAuth support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import httpx
import jwt


LOGGER = logging.getLogger(__name__)

OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_KEYRING_SERVICE = "Codex Auth"
DEFAULT_CODEX_HOME = "~/.codex"

VALID_AUTH_MODES = {"api_key", "oauth_codex", "oauth_token"}
TOKEN_EXPIRY_SKEW_SECONDS = 60
TOKEN_FALLBACK_TTL_SECONDS = 3600


class OpenAIAuthError(ValueError):
    """Raised when OpenAI credentials cannot be resolved safely."""


@dataclass
class CodexCredentials:
    """Loaded Codex authentication payload and storage metadata."""

    source: Literal["auth_json", "keyring"]
    payload: dict[str, Any]
    auth_json_path: Path
    keyring_key: str | None = None


@dataclass(frozen=True)
class ResolvedOpenAIAuth:
    """Resolved OpenAI bearer token and metadata."""

    token: str
    source: Literal[
        "explicit_api_key",
        "env_openai_api_key",
        "env_oauth_access_token",
        "codex_auth_json_api_key",
        "codex_auth_json_oauth",
        "codex_keyring_api_key",
        "codex_keyring_oauth",
    ]
    account_id: str | None = None
    is_oauth: bool = False


def mask_token(token: str | None) -> str:
    """Mask a token for safe logging."""
    if not token:
        return "<empty>"
    token = token.strip()
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


def resolve_openai_bearer_token(
    explicit_api_key: str | None = None,
    *,
    auth_mode: str | None = None,
    codex_home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    http_client: httpx.Client | None = None,
) -> ResolvedOpenAIAuth:
    """Resolve OpenAI credentials with API key compatibility and OAuth fallback.

    Priority:
        1) explicit_api_key
        2) OPENAI_API_KEY
        3) OPENAI_OAUTH_ACCESS_TOKEN
        4) Codex local credentials (auth.json/keyring)
    """
    env_map: Mapping[str, str]
    if env is None:
        env_map = os.environ
    else:
        env_map = env

    explicit = _normalize_secret(explicit_api_key)
    if explicit:
        return ResolvedOpenAIAuth(
            token=explicit, source="explicit_api_key", is_oauth=False
        )

    env_api_key = _normalize_secret(env_map.get("OPENAI_API_KEY"))
    if env_api_key:
        return ResolvedOpenAIAuth(
            token=env_api_key, source="env_openai_api_key", is_oauth=False
        )

    mode_raw = auth_mode if auth_mode is not None else env_map.get("CREWAI_OPENAI_AUTH_MODE")
    mode = _normalize_auth_mode(mode_raw)

    env_oauth_access_token = _normalize_secret(env_map.get("OPENAI_OAUTH_ACCESS_TOKEN"))
    if env_oauth_access_token:
        return ResolvedOpenAIAuth(
            token=env_oauth_access_token,
            source="env_oauth_access_token",
            is_oauth=True,
        )

    if mode == "oauth_token":
        raise OpenAIAuthError(
            "OPENAI_OAUTH_ACCESS_TOKEN is required when CREWAI_OPENAI_AUTH_MODE=oauth_token."
        )

    should_try_codex = mode == "oauth_codex" or mode_raw is None
    if should_try_codex:
        credentials = load_codex_credentials(codex_home=codex_home, env=env_map)
        if credentials:
            resolved = _resolve_from_codex_credentials(
                credentials, http_client=http_client
            )
            if mode_raw is None:
                LOGGER.info(
                    "OPENAI_API_KEY is unset; using OpenAI credentials from local Codex auth (%s).",
                    resolved.source,
                )
            return resolved

    if mode == "oauth_codex":
        raise OpenAIAuthError(
            "CREWAI_OPENAI_AUTH_MODE=oauth_codex is enabled but no usable Codex credentials "
            "were found. Run `codex login` or set OPENAI_OAUTH_ACCESS_TOKEN."
        )

    raise OpenAIAuthError(
        "OpenAI credentials not found. Set OPENAI_API_KEY, set OPENAI_OAUTH_ACCESS_TOKEN, "
        "or configure CREWAI_OPENAI_AUTH_MODE=oauth_codex after running `codex login`."
    )


def load_codex_credentials(
    *,
    codex_home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CodexCredentials | None:
    """Load Codex credentials from auth.json first, then keyring."""
    codex_home_path = _resolve_codex_home(codex_home=codex_home, env=env)
    auth_json_path = codex_home_path / "auth.json"

    if auth_json_path.exists():
        try:
            payload = json.loads(auth_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenAIAuthError(
                f"Failed to parse Codex auth file at {auth_json_path}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise OpenAIAuthError(
                f"Codex auth file at {auth_json_path} must contain a JSON object."
            )
        return CodexCredentials(
            source="auth_json", payload=payload, auth_json_path=auth_json_path
        )

    keyring_payload, keyring_key = _load_codex_credentials_from_keyring(codex_home_path)
    if keyring_payload is None:
        return None

    return CodexCredentials(
        source="keyring",
        payload=keyring_payload,
        auth_json_path=auth_json_path,
        keyring_key=keyring_key,
    )


def refresh_codex_access_token(
    refresh_token: str,
    *,
    http_client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Refresh Codex OAuth access token using the refresh token."""
    refresh_token = _normalize_secret(refresh_token)
    if not refresh_token:
        raise OpenAIAuthError("refresh_token is required to refresh Codex OAuth tokens.")

    payload = {
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": "openid profile email",
    }

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    try:
        response = client.post(OPENAI_OAUTH_TOKEN_URL, json=payload)
    except httpx.HTTPError as exc:
        raise OpenAIAuthError(
            "Failed to refresh Codex OAuth access token due to a network error."
        ) from exc
    finally:
        if owns_client:
            client.close()

    if response.status_code == 401:
        raise OpenAIAuthError(
            "Codex refresh_token is invalid or expired. Please run `codex login` again."
        )

    if not response.is_success:
        detail = (response.text or "").strip()
        detail = detail[:240] if detail else "no response body"
        raise OpenAIAuthError(
            "Failed to refresh Codex OAuth access token "
            f"(status={response.status_code}, detail={detail}). Existing token was kept unchanged."
        )

    try:
        refreshed = response.json()
    except ValueError as exc:
        raise OpenAIAuthError("Codex OAuth refresh response was not valid JSON.") from exc

    if not isinstance(refreshed, dict):
        raise OpenAIAuthError("Codex OAuth refresh response must be a JSON object.")

    if not _normalize_secret(_get_ci(refreshed, "access_token")):
        raise OpenAIAuthError(
            "Codex OAuth refresh succeeded but did not return an access_token."
        )

    return refreshed


def token_expiry_check(
    access_token: str | None,
    *,
    last_refresh: str | None = None,
    now: datetime | None = None,
    expiry_skew_seconds: int = TOKEN_EXPIRY_SKEW_SECONDS,
    fallback_ttl_seconds: int = TOKEN_FALLBACK_TTL_SECONDS,
) -> bool:
    """Return True when token is expired (or about to expire)."""
    token = _normalize_secret(access_token)
    if not token:
        return True

    current_time = now or datetime.now(UTC)
    jwt_exp = _extract_jwt_exp(token)
    if jwt_exp is not None:
        expiry_time = datetime.fromtimestamp(jwt_exp, tz=UTC)
        return current_time >= (expiry_time - timedelta(seconds=expiry_skew_seconds))

    refreshed_at = _parse_iso8601(last_refresh)
    if refreshed_at is None:
        return False

    assumed_expiry = refreshed_at + timedelta(seconds=fallback_ttl_seconds)
    return current_time >= (assumed_expiry - timedelta(seconds=expiry_skew_seconds))


def persist_updated_tokens(
    credentials: CodexCredentials,
    refreshed_tokens: Mapping[str, Any],
    *,
    last_refresh: datetime | None = None,
) -> None:
    """Persist refreshed token values back to the original storage location."""
    payload = credentials.payload
    tokens = _ensure_tokens_object(payload)

    for token_key in ("access_token", "refresh_token", "id_token", "account_id"):
        token_value = _normalize_secret(_get_ci(refreshed_tokens, token_key))
        if token_value:
            _set_ci(tokens, token_key, token_value)

    refresh_timestamp = (last_refresh or datetime.now(UTC)).isoformat()
    _set_ci(payload, "last_refresh", refresh_timestamp)

    if credentials.source == "auth_json":
        _write_auth_json_securely(credentials.auth_json_path, payload)
        return

    _persist_codex_credentials_to_keyring(credentials, payload)


def _resolve_from_codex_credentials(
    credentials: CodexCredentials,
    *,
    http_client: httpx.Client | None = None,
) -> ResolvedOpenAIAuth:
    codex_api_key = _normalize_secret(_get_ci(credentials.payload, "OPENAI_API_KEY"))
    if codex_api_key:
        return ResolvedOpenAIAuth(
            token=codex_api_key,
            source=(
                "codex_auth_json_api_key"
                if credentials.source == "auth_json"
                else "codex_keyring_api_key"
            ),
            is_oauth=False,
        )

    tokens = _ensure_tokens_object(credentials.payload)
    access_token = _normalize_secret(_get_ci(tokens, "access_token"))
    refresh_token = _normalize_secret(_get_ci(tokens, "refresh_token"))
    account_id = _normalize_secret(_get_ci(tokens, "account_id"))
    last_refresh = _normalize_secret(_get_ci(credentials.payload, "last_refresh"))

    should_refresh = token_expiry_check(access_token, last_refresh=last_refresh)
    if should_refresh and refresh_token:
        refreshed_tokens = refresh_codex_access_token(
            refresh_token, http_client=http_client
        )
        if account_id and not _normalize_secret(_get_ci(refreshed_tokens, "account_id")):
            refreshed_tokens = dict(refreshed_tokens)
            refreshed_tokens["account_id"] = account_id

        persist_updated_tokens(credentials, refreshed_tokens)

        access_token = _normalize_secret(_get_ci(refreshed_tokens, "access_token"))
        account_id = _normalize_secret(_get_ci(refreshed_tokens, "account_id")) or account_id

    if not access_token:
        if refresh_token:
            refreshed_tokens = refresh_codex_access_token(
                refresh_token, http_client=http_client
            )
            persist_updated_tokens(credentials, refreshed_tokens)
            access_token = _normalize_secret(_get_ci(refreshed_tokens, "access_token"))
            account_id = (
                _normalize_secret(_get_ci(refreshed_tokens, "account_id")) or account_id
            )

    if not access_token:
        raise OpenAIAuthError(
            "Codex credentials were found, but no usable access_token or OPENAI_API_KEY "
            "is available. Please run `codex login` again."
        )

    if token_expiry_check(access_token, last_refresh=_normalize_secret(_get_ci(credentials.payload, "last_refresh"))):
        raise OpenAIAuthError(
            "Codex access_token is expired and could not be refreshed. Please run `codex login` again."
        )

    return ResolvedOpenAIAuth(
        token=access_token,
        source=(
            "codex_auth_json_oauth"
            if credentials.source == "auth_json"
            else "codex_keyring_oauth"
        ),
        account_id=account_id,
        is_oauth=True,
    )


def _normalize_auth_mode(auth_mode: str | None) -> str:
    if auth_mode is None:
        return "api_key"
    normalized = auth_mode.strip().lower()
    if normalized not in VALID_AUTH_MODES:
        allowed = ", ".join(sorted(VALID_AUTH_MODES))
        raise OpenAIAuthError(
            f"Invalid CREWAI_OPENAI_AUTH_MODE={auth_mode!r}. Expected one of: {allowed}."
        )
    return normalized


def _resolve_codex_home(
    *,
    codex_home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    if codex_home is not None:
        value = str(codex_home)
    else:
        env_map = env if env is not None else os.environ
        value = env_map.get("CODEX_HOME", DEFAULT_CODEX_HOME)
    return Path(value).expanduser().resolve()


def _load_codex_credentials_from_keyring(
    codex_home: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    keyring = _import_keyring()
    if keyring is None:
        return None, None

    key = _codex_keyring_key(codex_home)
    try:
        serialized = keyring.get_password(CODEX_KEYRING_SERVICE, key)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to read Codex keyring entry: %s", exc)
        return None, key

    if not serialized:
        return None, key

    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise OpenAIAuthError(
            f"Codex keyring entry {key!r} is not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise OpenAIAuthError(f"Codex keyring entry {key!r} must be a JSON object.")

    return payload, key


def _persist_codex_credentials_to_keyring(
    credentials: CodexCredentials, payload: Mapping[str, Any]
) -> None:
    if credentials.keyring_key is None:
        return
    keyring = _import_keyring()
    if keyring is None:
        return
    try:
        keyring.set_password(
            CODEX_KEYRING_SERVICE,
            credentials.keyring_key,
            json.dumps(payload),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to persist refreshed Codex token to keyring: %s", exc)


def _codex_keyring_key(codex_home: Path) -> str:
    canonical_home = str(codex_home.expanduser().resolve())
    digest = hashlib.sha256(canonical_home.encode("utf-8")).hexdigest()[:16]
    return f"cli|{digest}"


def _write_auth_json_securely(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if os.name == "nt":
        path.write_text(serialized, encoding="utf-8")
        LOGGER.warning(
            "Updated %s with refreshed tokens. On Windows, ensure this file is only accessible to your user.",
            path,
        )
        return

    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
        file_obj.write(serialized)
    os.chmod(path, 0o600)


def _extract_jwt_exp(token: str) -> int | None:
    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_aud": False,
                "verify_iss": False,
            },
            algorithms=["HS256", "RS256", "ES256"],
        )
    except Exception:  # noqa: BLE001
        return None

    exp = payload.get("exp")
    if isinstance(exp, int):
        return exp
    if isinstance(exp, float):
        return int(exp)
    return None


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _import_keyring() -> Any | None:
    try:
        import keyring  # type: ignore[import-not-found]

        return keyring
    except Exception:  # noqa: BLE001
        return None


def _get_ci(mapping: Mapping[str, Any], key: str) -> Any:
    for current_key, value in mapping.items():
        if isinstance(current_key, str) and current_key.lower() == key.lower():
            return value
    return None


def _set_ci(mapping: dict[str, Any], key: str, value: Any) -> None:
    for current_key in list(mapping.keys()):
        if isinstance(current_key, str) and current_key.lower() == key.lower():
            mapping[current_key] = value
            return
    mapping[key] = value


def _ensure_tokens_object(payload: dict[str, Any]) -> dict[str, Any]:
    tokens = _get_ci(payload, "tokens")
    if isinstance(tokens, dict):
        return tokens
    tokens = {}
    _set_ci(payload, "tokens", tokens)
    return tokens


def _normalize_secret(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized
