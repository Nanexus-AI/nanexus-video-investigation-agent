"""Local Web console configuration. Defaults keep the process on localhost."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Server-side settings. Secrets never leave the process."""

    host: str = "127.0.0.1"
    port: int = 8787
    domain_db_path: Path = Path(".local/web/domain.sqlite")
    media_store_root: Path = Path(".local/web/media_store")
    frigate_env_path: Path = Path(".local/frigate.env")
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:14b"
    visual_model: str = "moondream:1.8b"
    enable_observe_media: bool = True
    max_plan_steps: int = 8
    max_model_calls: int = 10
    max_tool_attempts: int = 10
    timezone_label: str = "America/Toronto"
    # live | stub — stub is for deterministic tests only.
    tool_backend: str = "live"
    stub_cameras: tuple[str, ...] = ("front_yard",)
    # Empty = no client IP filter (localhost default). Set for temporary LAN review.
    allowed_clients: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "WebConfig":
        backend = (os.environ.get("NANEXUS_WEB_BACKEND") or "live").strip().lower()
        stub_cameras = tuple(
            c.strip()
            for c in (os.environ.get("NANEXUS_WEB_STUB_CAMERAS") or "front_yard").split(
                ","
            )
            if c.strip()
        )
        allowed_clients = tuple(
            c.strip()
            for c in (os.environ.get("NANEXUS_WEB_ALLOWED_CLIENTS") or "").split(",")
            if c.strip()
        )
        return cls(
            host=os.environ.get("NANEXUS_WEB_HOST", "127.0.0.1"),
            port=int(os.environ.get("NANEXUS_WEB_PORT", "8787")),
            domain_db_path=Path(
                os.environ.get("NANEXUS_WEB_DOMAIN_DB", ".local/web/domain.sqlite")
            ),
            media_store_root=Path(
                os.environ.get("NANEXUS_WEB_MEDIA_ROOT", ".local/web/media_store")
            ),
            frigate_env_path=Path(
                os.environ.get("NANEXUS_FRIGATE_ENV", ".local/frigate.env")
            ),
            ollama_base_url=os.environ.get(
                "NANEXUS_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            chat_model=os.environ.get("NANEXUS_CHAT_MODEL", "qwen3:14b"),
            visual_model=os.environ.get("NANEXUS_VISUAL_MODEL", "moondream:1.8b"),
            enable_observe_media=(
                os.environ.get("NANEXUS_ENABLE_OBSERVE", "1").lower()
                not in {"0", "false", "no"}
            ),
            timezone_label=os.environ.get("NANEXUS_TZ", "America/Toronto"),
            tool_backend=backend if backend in {"live", "stub"} else "live",
            stub_cameras=stub_cameras or ("front_yard",),
            allowed_clients=allowed_clients,
        )
