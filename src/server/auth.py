"""
Authentication and API key management.

Token system like Telegram Bot API:
- Users get an API key to connect their voice widget
- Keys can be scoped (rate limits, features)
- Hosted version charges per minute or monthly
"""

import asyncio
import json
import os
import secrets
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class APIKey:
    """API key with metadata and limits."""

    key_id: str
    key_hash: str  # Store hash, not plaintext
    name: str
    created_at: datetime

    # Limits
    rate_limit_per_minute: int = 60  # requests per minute
    monthly_minutes: Optional[int] = None  # None = unlimited

    # Usage tracking
    minutes_used: float = 0.0
    last_request_at: Optional[datetime] = None
    request_count_this_minute: int = 0

    # Features
    features: Dict[str, bool] = field(
        default_factory=lambda: {
            "continuous_mode": True,
            "voice_cloning": False,
            "priority_queue": False,
        }
    )

    # Status
    active: bool = True
    tier: str = "free"  # free, pro, enterprise


DB_DIR = Path(__file__).resolve().parent.parent / "data"


class TokenManager:
    """
    Manage API tokens for voice connections.

    Uses in-memory dicts for speed, backed by SQLite for persistence.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._keys: Dict[str, APIKey] = {}
        self._key_to_id: Dict[str, str] = {}
        self._db_path = db_path or os.getenv("OPENCLAW_AUTH_DB") or str(DB_DIR / "auth.db")
        self._rl_lock = asyncio.Lock()

    def _ensure_db(self):
        """Create the directory and database table if missing."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                rate_limit_per_minute INTEGER NOT NULL DEFAULT 60,
                monthly_minutes INTEGER,
                minutes_used REAL NOT NULL DEFAULT 0.0,
                active INTEGER NOT NULL DEFAULT 1,
                tier TEXT NOT NULL DEFAULT 'free',
                features TEXT NOT NULL DEFAULT '{}',
                _window_start TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save(self):
        """Persist all keys to SQLite."""
        self._ensure_db()
        conn = sqlite3.connect(self._db_path)
        for key_id, key in self._keys.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO api_keys
                (key_id, key_hash, name, created_at, rate_limit_per_minute,
                 monthly_minutes, minutes_used, active, tier, features, _window_start)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key.key_id,
                    key.key_hash,
                    key.name,
                    key.created_at.isoformat(),
                    key.rate_limit_per_minute,
                    key.monthly_minutes,
                    key.minutes_used,
                    1 if key.active else 0,
                    key.tier,
                    json.dumps(key.features),
                    key._window_start.isoformat() if getattr(key, "_window_start", None) else None,
                ),
            )
        conn.commit()
        conn.close()

    def load(self):
        """Load keys from SQLite into memory."""
        self._ensure_db()
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT * FROM api_keys")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            data = dict(zip(columns, row))
            key = APIKey(
                key_id=data["key_id"],
                key_hash=data["key_hash"],
                name=data["name"],
                created_at=datetime.fromisoformat(data["created_at"]),
                rate_limit_per_minute=data["rate_limit_per_minute"],
                monthly_minutes=data["monthly_minutes"],
                minutes_used=data["minutes_used"],
                active=bool(data["active"]),
                tier=data["tier"],
                features=json.loads(data["features"]),
            )
            if data.get("_window_start"):
                key._window_start = datetime.fromisoformat(data["_window_start"])
            self._keys[key.key_id] = key
            self._key_to_id[key.key_hash] = key.key_id

        if rows:
            logger.info(f"Loaded {len(rows)} API keys from {self._db_path}")

    def generate_key(
        self,
        name: str,
        tier: str = "free",
        rate_limit: int = 60,
        monthly_minutes: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.

        Returns:
            (plaintext_key, APIKey object)

        Note: Plaintext key is only returned once!
        """
        # Generate secure random key
        key_id = secrets.token_hex(8)
        plaintext_key = f"ocv_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(plaintext_key)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(tz=None),
            rate_limit_per_minute=rate_limit,
            monthly_minutes=monthly_minutes,
            tier=tier,
        )

        self._keys[key_id] = api_key
        self._key_to_id[key_hash] = key_id
        self.save()

        logger.info(f"Generated API key: {key_id} ({name}, tier={tier})")

        return plaintext_key, api_key

    def validate_key(self, plaintext_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its metadata.

        Returns None if invalid.
        """
        if not plaintext_key or not plaintext_key.startswith("ocv_"):
            return None

        key_hash = self._hash_key(plaintext_key)
        key_id = self._key_to_id.get(key_hash)

        if not key_id:
            return None

        api_key = self._keys.get(key_id)

        if not api_key or not api_key.active:
            return None

        return api_key

    async def check_rate_limit(self, api_key: APIKey) -> bool:
        """
        Check if request is within rate limits using a fixed window.

        Returns True if allowed, False if rate limited.
        """
        async with self._rl_lock:
            now = datetime.now(tz=None)
            window_start = getattr(api_key, "_window_start", None)

            if window_start is None or (now - window_start).total_seconds() >= 60:
                api_key._window_start = now
                api_key.request_count_this_minute = 0

            if api_key.request_count_this_minute >= api_key.rate_limit_per_minute:
                return False

            api_key.request_count_this_minute += 1
            return True

    def check_monthly_quota(self, api_key: APIKey, minutes: float = 0) -> bool:
        """
        Check if within monthly minute quota.

        Returns True if allowed, False if quota exceeded.
        """
        if api_key.monthly_minutes is None:
            return True  # Unlimited

        return (api_key.minutes_used + minutes) <= api_key.monthly_minutes

    def record_usage(self, api_key: APIKey, minutes: float):
        """Record minutes used for billing."""
        api_key.minutes_used += minutes
        self.save()
        logger.debug(
            f"Key {api_key.key_id}: used {minutes:.2f} min, total {api_key.minutes_used:.2f}"
        )

    def get_usage(self, api_key: APIKey) -> Dict[str, Any]:
        """Get usage stats for an API key."""
        return {
            "key_id": api_key.key_id,
            "name": api_key.name,
            "tier": api_key.tier,
            "minutes_used": round(api_key.minutes_used, 2),
            "monthly_limit": api_key.monthly_minutes,
            "rate_limit": api_key.rate_limit_per_minute,
            "features": api_key.features,
        }

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].active = False
            self.save()
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False

    def _hash_key(self, plaintext_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(plaintext_key.encode()).hexdigest()


# Global token manager instance
token_manager = TokenManager()


# Helper to load keys from environment and persisted database
def load_keys_from_env():
    """
    Load API keys from SQLite database, then overlay env var keys.

    Format: OPENCLAW_API_KEY_<name>=<plaintext_key>
    """

    # Restore keys from database first
    token_manager.load()

    # Check for master key (allows all access)
    master_key = os.getenv("OPENCLAW_MASTER_KEY")
    if master_key:
        # Register master key
        key_hash = token_manager._hash_key(master_key)
        api_key = APIKey(
            key_id="master",
            key_hash=key_hash,
            name="Master Key",
            created_at=datetime.now(tz=None),
            rate_limit_per_minute=1000,
            monthly_minutes=None,
            tier="enterprise",
        )
        api_key.features = {
            "continuous_mode": True,
            "voice_cloning": True,
            "priority_queue": True,
        }
        token_manager._keys["master"] = api_key
        token_manager._key_to_id[key_hash] = "master"
        token_manager.save()
        logger.info("Loaded master API key from environment")


# Pricing tiers for hosted version
PRICING_TIERS = {
    "free": {
        "monthly_minutes": 60,
        "rate_limit": 30,
        "price": 0,
        "features": ["continuous_mode"],
    },
    "pro": {
        "monthly_minutes": 500,
        "rate_limit": 120,
        "price": 29,  # $/month
        "features": ["continuous_mode", "voice_cloning"],
    },
    "enterprise": {
        "monthly_minutes": None,  # Unlimited
        "rate_limit": 500,
        "price": 99,  # $/month
        "features": ["continuous_mode", "voice_cloning", "priority_queue"],
    },
}
