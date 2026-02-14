# =============================================================================
# CCTV Alert System - Screenshot Storage
# v1.0.0
# =============================================================================
# Handles saving motion-capture frames to disk as timestamped JPEG files.
# Files are written to data/screenshots/ with names like:
#   2025-01-15_14-30-45.jpg
#
# The directory is created automatically on first use if it doesn't exist.
# =============================================================================

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import SCREENSHOTS_DIR

# Module-level logger.
logger = logging.getLogger(__name__)


def ensure_directory() -> None:
    """Create the screenshots directory if it doesn't already exist.

    Uses parents=True so any missing intermediate directories are also
    created (e.g. if data/ itself is absent). exist_ok=True makes the
    call idempotent — safe to call on every save without checking first.
    """
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_screenshot(frame: np.ndarray) -> Path:
    """Write a BGR frame to disk as a timestamped JPEG.

    Args:
        frame: The BGR numpy array captured from the camera.

    Returns:
        The Path object pointing to the saved file. This is passed to
        the Telegram alert module and stored in the database record.

    Raises:
        IOError: If cv2.imwrite fails (e.g. disk full, bad permissions).
    """
    # Make sure the target directory exists before writing.
    ensure_directory()

    # Build a filename from the current local time.
    # Format: YYYY-MM-DD_HH-MM-SS.jpg
    # Colons are avoided because they are illegal in filenames on macOS/Windows.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.jpg"
    filepath = SCREENSHOTS_DIR / filename

    # cv2.imwrite returns True on success.
    # JPEG quality defaults to 95 which is a good size/quality trade-off.
    success = cv2.imwrite(str(filepath), frame)

    if not success:
        raise IOError(f"Failed to write screenshot to {filepath}")

    logger.info("Screenshot saved: %s", filepath)
    return filepath
