# =============================================================================
# CCTV Alert System - Configuration Loader
# v1.0.0
# =============================================================================
# Centralizes all configuration by reading environment variables from a .env
# file at the project root. Every other module imports settings from here,
# ensuring a single source of truth and clean separation of secrets from code.
# =============================================================================

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
# BASE_DIR points to the project root (one level above this file's parent).
# All relative paths (database, screenshots) are anchored here so the app
# works regardless of the current working directory at launch time.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load .env file
# ---------------------------------------------------------------------------
# load_dotenv() reads key=value pairs from the .env file and injects them
# into os.environ. If the file is missing the app will still work as long
# as the variables are exported in the shell environment.
# ---------------------------------------------------------------------------
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Print to stderr because logging is not configured yet at import time.
    print(
        f"WARNING: .env file not found at {_env_path}. "
        "Falling back to shell environment variables.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# RTSP stream
# ---------------------------------------------------------------------------
# The full RTSP URL including credentials and path. Required.
RTSP_URL: str = os.getenv("RTSP_URL", "")

# ---------------------------------------------------------------------------
# Telegram bot credentials
# ---------------------------------------------------------------------------
# TELEGRAM_TOKEN: obtained from @BotFather.
# TELEGRAM_CHAT_ID: the numeric chat/group ID to send alerts to.
# Both are required for the alert subsystem to function.
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Motion detection tuning
# ---------------------------------------------------------------------------
# MOTION_THRESHOLD (int): minimum contour area in pixels that qualifies
#     as real motion. Contours smaller than this are ignored as noise.
#     For a 1080p stream (~2M pixels), 5000 catches a small moving object.
#
# COOLDOWN_SECONDS (int): minimum seconds between consecutive alerts.
#     Prevents flooding Telegram when motion is continuous.
# ---------------------------------------------------------------------------
MOTION_THRESHOLD: int = int(os.getenv("MOTION_THRESHOLD", "5000"))
COOLDOWN_SECONDS: int = int(os.getenv("COOLDOWN_SECONDS", "20"))

# ---------------------------------------------------------------------------
# Stream resilience
# ---------------------------------------------------------------------------
# RECONNECT_DELAY_SECONDS (int): pause before each reconnection attempt
#     after the RTSP stream drops. Gives the camera time to recover.
#
# MAX_RECONNECT_ATTEMPTS (int): hard cap on consecutive reconnection
#     tries. After this many failures the application exits with an error.
# ---------------------------------------------------------------------------
RECONNECT_DELAY_SECONDS: int = int(os.getenv("RECONNECT_DELAY_SECONDS", "5"))
MAX_RECONNECT_ATTEMPTS: int = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "50"))

# ---------------------------------------------------------------------------
# YOLO object detection
# ---------------------------------------------------------------------------
# YOLO_MODEL (str): model file name, downloaded automatically on first run.
#     yolov8n.pt = nano (~6 MB, fastest), yolov8s.pt = small (~22 MB).
#
# YOLO_CONFIDENCE (float): minimum confidence score (0.0-1.0) for a
#     detection to be accepted. Higher = fewer false positives.
#
# YOLO_INTERVAL (float): minimum seconds between YOLO inferences.
#     0.33 ≈ 3 FPS, which balances responsiveness with CPU usage.
#
# YOLO_CLASSES (str): comma-separated list of COCO class names to detect.
# ---------------------------------------------------------------------------
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
YOLO_INTERVAL: float = float(os.getenv("YOLO_INTERVAL", "0.33"))
YOLO_CLASSES: str = os.getenv(
    "YOLO_CLASSES",
    "person,car,truck,motorcycle,bicycle,bus,cat,dog,bird,horse,cow,sheep,bear",
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
# DATA_DIR:        root directory for all persistent data.
# SCREENSHOTS_DIR: timestamped motion-capture images.
# DATABASE_PATH:   SQLite database file storing event records.
# Directories are created automatically on first use by their modules.
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
SCREENSHOTS_DIR: Path = DATA_DIR / "screenshots"
DATABASE_PATH: Path = DATA_DIR / "cctv.db"

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# A shared logging setup so every module writes consistent, timestamped
# output. Level is INFO for production; change to DEBUG for development.
# The format includes module name and line number for fast triage.
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger with a consistent format and level.

    Called once at application startup in main.py. All modules that use
    logging.getLogger(__name__) automatically inherit this configuration.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )


def validate_config() -> None:
    """Check that all required settings are present and raise early if not.

    This runs at startup so the operator gets a clear error message
    instead of a cryptic failure minutes later when a module first
    tries to use a missing value.
    """
    missing: list[str] = []

    if not RTSP_URL:
        missing.append("RTSP_URL")
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Create a .env file at {_env_path} — see .env.example for reference."
        )
