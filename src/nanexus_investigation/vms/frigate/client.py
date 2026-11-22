"""Minimal authenticated read-only Frigate HTTP client.

Allowlisted GET operations only. No model-proposed paths, no retries, no
write endpoints. Transport is separate from Nanexus evidence contracts.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

ALLOWED_GET_PATHS = frozenset(
    {
        "/api/version",
        "/api/events",
    }
)

# Event-bound binary media resources verified against Frigate 0.17.1-416a9b7.
ALLOWED_EVENT_MEDIA_SUFFIXES = frozenset(
    {
        "snapshot.jpg",
        "clip.mp4",
    }
)

DEFAULT_MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_CLIP_BYTES = 25 * 1024 * 1024


class FrigateHttpError(RuntimeError):
    """Base Frigate transport error. Messages must not include credentials."""


class FrigateAuthError(FrigateHttpError):
    pass


class FrigateNotFoundError(FrigateHttpError):
    pass


class FrigateUnavailableError(FrigateHttpError):
    pass


class FrigateProtocolError(FrigateHttpError):
    pass


class FrigateScopeError(FrigateHttpError):
    """Raised before a request when camera/time/label constraints are violated."""


class FrigateMediaError(FrigateHttpError):
    """Media acquisition failed without implying visual interpretation."""


class FrigateMediaTooLargeError(FrigateMediaError):
    pass


class FrigateMediaTypeError(FrigateMediaError):
    pass


@dataclass(frozen=True, slots=True)
class FrigateClientConfig:
    """Connection settings for the authenticated Frigate entry (typically :8971)."""

    base_url: str
    vms_instance_id: str
    username: str | None = None
    password: str | None = None
    bearer_token: str | None = None
    # Optional pre-established Frigate JWT session cookie ("name=value" or raw JWT).
    session_cookie: str | None = None
    ca_bundle: str | None = None
    # When True, verify TLS against ca_bundle/system trust but skip hostname/SAN
    # matching for local leaf certificates that lack a matching SAN.
    disable_hostname_check: bool = False
    timeout_s: float = 30.0
    authorized_cameras: tuple[str, ...] = ()
    max_limit: int = 50

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an absolute http(s) origin")
        # Refuse the unauthenticated Frigate internal port as a configured origin.
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.port == 5000:
            raise ValueError(
                "unauthenticated Frigate port 5000 is not permitted; "
                "use authenticated entry (typically 8971)"
            )
        methods = [
            bool(self.bearer_token),
            bool(self.username and self.password),
            bool(self.session_cookie),
        ]
        if sum(1 for used in methods if used) > 1:
            raise ValueError(
                "configure only one of bearer_token, username/password, or session_cookie"
            )
        # Hostname/SAN bypass is only allowed with an explicit pinned trust anchor so
        # certificate mismatch still fails closed under CERT_REQUIRED verification.
        if self.disable_hostname_check and not self.ca_bundle:
            raise ValueError(
                "disable_hostname_check requires ca_bundle (pinned certificate trust)"
            )


@dataclass
class FrigateHttpResponse:
    """Raw transport response retained for recording/replay."""

    status_code: int
    path: str
    query: dict[str, str]
    body: bytes
    headers: dict[str, str]
    retrieved_at: datetime
    frigate_version: str | None = None


@dataclass
class FrigateReadOnlyClient:
    """Allowlisted GET client with optional login-cookie or bearer auth."""

    config: FrigateClientConfig
    _cookie: str | None = field(default=None, init=False, repr=False)
    _version: str | None = field(default=None, init=False)

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.config.base_url.startswith("https://"):
            return None
        if self.config.ca_bundle:
            # Pin the operator-supplied trust anchor; CERT_REQUIRED remains on.
            ctx = ssl.create_default_context(cafile=self.config.ca_bundle)
        else:
            ctx = ssl.create_default_context()
        if self.config.disable_hostname_check:
            # Scoped exception for Frigate default leaf (CN=*) when ca_bundle pins
            # that exact certificate. Wrong/mismatched certs still fail verification.
            ctx.check_hostname = False
        return ctx

    @staticmethod
    def _session_cookie_from_headers(headers: dict[str, str]) -> str | None:
        """Extract Frigate session cookie from Set-Cookie header(s).

        Frigate 0.17 authenticated UI/API uses a login JWT cookie (not HTTP Basic).
        """
        # urllib may expose a single Set-Cookie or fold multiples.
        candidates: list[str] = []
        for key, value in headers.items():
            if key.lower() == "set-cookie":
                candidates.append(value)
        for raw in candidates:
            # May contain multiple cookies separated by commas in folded form;
            # take name=value before the first ';' of the first cookie pair.
            first = raw.split(",", 1)[0]
            token = first.split(";", 1)[0].strip()
            if "=" in token and not token.lower().startswith(("path=", "expires=", "max-age=")):
                return token
        return None

    def _ensure_auth(self) -> None:
        if self.config.bearer_token:
            return
        if self._cookie:
            return
        if self.config.session_cookie:
            raw = self.config.session_cookie.strip()
            # Accept "frigate=<jwt>" or raw JWT value.
            self._cookie = raw if "=" in raw else f"frigate={raw}"
            return
        if not (self.config.username and self.config.password):
            raise FrigateAuthError(
                "authenticated Frigate access requires bearer_token, "
                "username/password, or session_cookie"
            )
        # Frigate 0.17 /api/login expects JSON {"user","password"} and returns
        # a Set-Cookie session JWT for subsequent GETs on :8971.
        status, headers, _body = self._raw_request(
            "POST",
            "/api/login",
            json_body={
                "user": self.config.username,
                "password": self.config.password,
            },
            allow_unlisted=True,
        )
        if status in {401, 403}:
            raise FrigateAuthError("Frigate login rejected")
        if status >= 500:
            raise FrigateUnavailableError(f"Frigate login unavailable ({status})")
        if status >= 400:
            raise FrigateProtocolError(f"Frigate login failed ({status})")
        self._cookie = self._session_cookie_from_headers(headers)
        if not self._cookie:
            raise FrigateAuthError("Frigate login did not return a session cookie")

    def _authorize_scope(
        self,
        *,
        cameras: tuple[str, ...],
        after: float | None,
        before: float | None,
        limit: int,
    ) -> None:
        if limit < 1 or limit > self.config.max_limit:
            raise FrigateScopeError(
                f"limit must be between 1 and {self.config.max_limit}"
            )
        if after is not None and before is not None and before <= after:
            raise FrigateScopeError("time range must be a positive half-open interval")
        authorized = set(self.config.authorized_cameras)
        if authorized:
            for camera in cameras:
                if camera not in authorized:
                    raise FrigateScopeError(f"camera not authorized: {camera}")
        elif not cameras:
            raise FrigateScopeError("at least one camera must be specified")

    @staticmethod
    def _is_allowlisted_get_path(base_path: str) -> bool:
        if base_path in ALLOWED_GET_PATHS:
            return True
        if not base_path.startswith("/api/events/"):
            return False
        parts = [part for part in base_path.split("/") if part]
        # events detail: api/events/{id}
        if len(parts) == 3:
            return True
        # event media: api/events/{id}/snapshot.jpg|clip.mp4
        if len(parts) == 4 and parts[3] in ALLOWED_EVENT_MEDIA_SUFFIXES:
            return True
        return False

    def _auth_headers(self, *, accept: str) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": accept}
        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self._cookie:
            headers["Cookie"] = self._cookie
        return headers

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_unlisted: bool = False,
        accept: str = "application/json",
        max_bytes: int | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        if method == "GET":
            base_path = path.split("?", 1)[0]
            if not self._is_allowlisted_get_path(base_path):
                raise FrigateScopeError(f"path not allowlisted: {base_path}")
        elif not allow_unlisted:
            raise FrigateScopeError(f"method not allowlisted: {method} {path}")

        origin = urllib.parse.urlparse(self.config.base_url)
        url = self.config.base_url.rstrip("/") + path
        if query:
            url = url + "?" + urllib.parse.urlencode(query)
        headers = self._auth_headers(accept=accept)
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        ctx = self._ssl_context()

        class _RejectForeignRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
                parsed = urllib.parse.urlparse(newurl)
                if parsed.scheme != origin.scheme or parsed.netloc != origin.netloc:
                    raise FrigateScopeError(
                        "refusing media redirect to an unapproved host"
                    )
                return urllib.request.HTTPRedirectHandler.redirect_request(
                    self, req, fp, code, msg, headers, newurl
                )

        handlers: list[Any] = [_RejectForeignRedirect()]
        if ctx is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=self.config.timeout_s) as response:
                final = urllib.parse.urlparse(response.geturl())
                if final.scheme != origin.scheme or final.netloc != origin.netloc:
                    raise FrigateScopeError(
                        "refusing media response from an unapproved host"
                    )
                headers_out = dict(response.headers.items())
                if max_bytes is None:
                    body = response.read()
                else:
                    body = self._read_bounded(response, max_bytes=max_bytes)
                return response.status, headers_out, body
        except FrigateMediaTooLargeError:
            raise
        except FrigateScopeError:
            raise
        except urllib.error.HTTPError as exc:
            try:
                err_body = (
                    exc.read()
                    if max_bytes is None
                    else self._read_bounded(exc, max_bytes=max_bytes)
                )
            except FrigateMediaTooLargeError:
                raise
            return exc.code, dict(exc.headers.items()), err_body
        except urllib.error.URLError as exc:
            raise FrigateUnavailableError(
                f"Frigate endpoint unavailable: {exc.reason}"
            ) from exc

    @staticmethod
    def _read_bounded(stream: Any, *, max_bytes: int) -> bytes:
        """Read at most max_bytes; refuse unbounded buffering of oversized bodies."""
        if max_bytes < 1:
            raise FrigateMediaTooLargeError("max_bytes must be >= 1")
        chunks: list[bytes] = []
        total = 0
        while True:
            # Read slightly past the limit so oversized bodies are detected.
            to_read = min(64 * 1024, max_bytes - total + 1)
            chunk = stream.read(to_read)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise FrigateMediaTooLargeError(
                    f"media response exceeds max_bytes={max_bytes}"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _get(self, path: str, query: dict[str, str] | None = None) -> FrigateHttpResponse:
        self._ensure_auth()
        status, headers, body = self._raw_request("GET", path, query=query)
        if status in {401, 403}:
            raise FrigateAuthError(f"Frigate authorization denied ({status})")
        if status == 404:
            raise FrigateNotFoundError(f"Frigate resource not found ({path})")
        if status >= 500:
            raise FrigateUnavailableError(f"Frigate unavailable ({status})")
        if status >= 400:
            raise FrigateProtocolError(f"Frigate request rejected ({status})")
        return FrigateHttpResponse(
            status_code=status,
            path=path,
            query=dict(query or {}),
            body=body,
            headers={k: v for k, v in headers.items()},
            retrieved_at=datetime.now(tz=timezone.utc),
            frigate_version=self._version,
        )

    def get_version(self) -> str:
        response = self._get("/api/version")
        text = response.body.decode("utf-8").strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = text
        if isinstance(decoded, str) and decoded:
            self._version = decoded
            return decoded
        if isinstance(decoded, dict) and isinstance(decoded.get("version"), str):
            self._version = decoded["version"]
            return decoded["version"]
        raise FrigateProtocolError("invalid Frigate version response")

    def list_events(
        self,
        *,
        cameras: tuple[str, ...],
        after: float | None = None,
        before: float | None = None,
        label: str | None = None,
        limit: int = 25,
    ) -> FrigateHttpResponse:
        self._authorize_scope(cameras=cameras, after=after, before=before, limit=limit)
        query: dict[str, str] = {
            "cameras": ",".join(cameras),
            "limit": str(limit),
            "include_thumbnails": "0",
        }
        if after is not None:
            query["after"] = str(after)
        if before is not None:
            query["before"] = str(before)
        if label:
            query["labels"] = label
        return self._get("/api/events", query=query)

    def get_event(self, event_id: str) -> FrigateHttpResponse:
        if not event_id or event_id.strip() != event_id:
            raise FrigateScopeError("event_id must be non-empty and trimmed")
        if "/" in event_id or ".." in event_id:
            raise FrigateScopeError("event_id contains forbidden characters")
        path = f"/api/events/{quote(event_id, safe='')}"
        return self._get(path)

    def _validate_event_id(self, event_id: str) -> str:
        if not event_id or event_id.strip() != event_id:
            raise FrigateScopeError("event_id must be non-empty and trimmed")
        if "/" in event_id or ".." in event_id or "?" in event_id:
            raise FrigateScopeError("event_id contains forbidden characters")
        return event_id

    def _get_event_media(
        self,
        event_id: str,
        *,
        suffix: str,
        expected_types: frozenset[str],
        max_bytes: int,
    ) -> FrigateHttpResponse:
        """Bounded authenticated GET for an allowlisted event media resource."""
        if suffix not in ALLOWED_EVENT_MEDIA_SUFFIXES:
            raise FrigateScopeError(f"media suffix not allowlisted: {suffix}")
        event_id = self._validate_event_id(event_id)
        self._ensure_auth()
        path = f"/api/events/{quote(event_id, safe='')}/{suffix}"
        status, headers, body = self._raw_request(
            "GET",
            path,
            accept="*/*",
            max_bytes=max_bytes,
        )
        if status in {401, 403}:
            raise FrigateAuthError(f"Frigate authorization denied ({status})")
        if status == 404:
            raise FrigateNotFoundError(f"Frigate media not found ({suffix})")
        if status >= 500:
            raise FrigateUnavailableError(f"Frigate unavailable ({status})")
        if status >= 400:
            raise FrigateProtocolError(f"Frigate media request rejected ({status})")
        content_type = (headers.get("Content-Type") or headers.get("content-type") or "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in expected_types:
            raise FrigateMediaTypeError(
                f"unexpected media content type for {suffix}: {media_type or 'missing'}"
            )
        if not body:
            raise FrigateMediaError(f"empty media body for {suffix}")
        return FrigateHttpResponse(
            status_code=status,
            path=path,
            query={},
            body=body,
            headers={k: v for k, v in headers.items()},
            retrieved_at=datetime.now(tz=timezone.utc),
            frigate_version=self._version,
        )

    def get_event_snapshot(
        self,
        event_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    ) -> FrigateHttpResponse:
        return self._get_event_media(
            event_id,
            suffix="snapshot.jpg",
            expected_types=frozenset({"image/jpeg", "image/jpg"}),
            max_bytes=max_bytes,
        )

    def get_event_clip(
        self,
        event_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_CLIP_BYTES,
    ) -> FrigateHttpResponse:
        return self._get_event_media(
            event_id,
            suffix="clip.mp4",
            expected_types=frozenset({"video/mp4", "application/mp4"}),
            max_bytes=max_bytes,
        )
