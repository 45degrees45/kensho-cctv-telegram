# =============================================================================
# CCTV Alert System - SQLite Database Layer
# v1.0.0
# =============================================================================
# Manages a lightweight SQLite database that logs every motion event.
#
# Table schema — `events`:
#   id          INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp   TEXT    ISO-8601 datetime when motion was detected
#   screenshot  TEXT    Absolute path to the saved JPEG file
#   notified    INTEGER 1 if Telegram alert was sent, 0 if it failed
#
# SQLite is ideal here: it's serverless, zero-config, ships with Python,
# and handles the modest write throughput of a single-camera system easily.
# =============================================================================

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

from app.config import DATABASE_PATH, DATA_DIR

# Module-level logger.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection
# ---------------------------------------------------------------------------
# We keep a single connection open for the lifetime of the process.
# SQLite in WAL mode supports concurrent reads and serialised writes,
# which is perfectly fine for a single-threaded alert loop.
# ---------------------------------------------------------------------------
_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    """Return the module-level SQLite connection, creating it on first call.

    The database file and its parent directory are created automatically.
    WAL (Write-Ahead Logging) mode is enabled for better concurrency
    and crash resilience.

    Returns:
        An open sqlite3.Connection object.
    """
    global _connection

    if _connection is not None:
        return _connection

    # Ensure the data/ directory exists before SQLite tries to create the file.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Opening database: %s", DATABASE_PATH)
    _connection = sqlite3.connect(str(DATABASE_PATH))

    # Enable WAL mode. This allows readers to proceed without blocking
    # behind a writer, and vice-versa. It also reduces the chance of
    # database corruption if the process is killed mid-write.
    _connection.execute("PRAGMA journal_mode=WAL;")

    # Return rows as sqlite3.Row objects so columns can be accessed by name.
    _connection.row_factory = sqlite3.Row

    return _connection


def init_db() -> None:
    """Create the events table if it doesn't already exist.

    Called once at application startup. IF NOT EXISTS makes the call
    idempotent so it's safe to run on every launch. Also migrates
    existing databases by adding the detected_objects column.
    """
    conn = _get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            screenshot       TEXT    NOT NULL,
            notified         INTEGER NOT NULL DEFAULT 0,
            detected_objects TEXT    NOT NULL DEFAULT ''
        );
        """
    )
    conn.commit()

    # Migration for existing databases that lack the detected_objects column.
    try:
        conn.execute("ALTER TABLE events ADD COLUMN detected_objects TEXT NOT NULL DEFAULT '';")
        conn.commit()
        logger.info("Migrated events table — added detected_objects column.")
    except sqlite3.OperationalError:
        # Column already exists — nothing to do.
        pass

    logger.info("Database initialised — events table ready.")


def insert_event(
    screenshot_path: Path,
    notified: bool,
    detected_objects: str = "",
) -> int:
    """Insert a new detection event record.

    Args:
        screenshot_path:  Absolute path to the JPEG file saved for this event.
        notified:         Whether the Telegram alert was sent successfully.
        detected_objects: Comma-separated list of detected object labels
                          (e.g. "2x person, 1x car"). Defaults to empty string.

    Returns:
        The auto-generated row ID of the new record.
    """
    conn = _get_connection()

    # ISO-8601 timestamp for human-readable querying.
    now = datetime.now().isoformat()

    cursor = conn.execute(
        "INSERT INTO events (timestamp, screenshot, notified, detected_objects) VALUES (?, ?, ?, ?);",
        (now, str(screenshot_path), int(notified), detected_objects),
    )
    conn.commit()

    row_id = cursor.lastrowid
    logger.info(
        "Event #%d recorded — screenshot=%s, notified=%s, objects=%s.",
        row_id,
        screenshot_path.name,
        notified,
        detected_objects or "(none)",
    )
    return row_id


def close_db() -> None:
    """Close the database connection cleanly.

    Called during application shutdown. Safe to call even if the
    connection was never opened (e.g. early exit due to config error).
    """
    global _connection

    if _connection is not None:
        _connection.close()
        _connection = None
        logger.info("Database connection closed.")
