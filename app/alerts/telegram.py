# =============================================================================
# CCTV Alert System - Telegram Alert Sender
# v1.0.0
# =============================================================================
# Sends motion-detection screenshots to a Telegram chat via the Bot API.
#
# Uses the `python-telegram-bot` library (v20+), which provides an async
# interface. Because the main detection loop is synchronous (simpler and
# perfectly adequate for a single-camera system), we bridge the gap by
# running the async send call inside asyncio.run().
#
# If sending fails (network error, invalid token, etc.) the failure is
# logged and the function returns False so the caller can record it in
# the database — alerts should never crash the monitoring loop.
# =============================================================================

import asyncio
import logging
from pathlib import Path

import telegram

from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Module-level logger.
logger = logging.getLogger(__name__)


async def _send_photo_async(photo_path: Path, caption: str) -> bool:
    """Internal async implementation of the Telegram photo send.

    This is a coroutine that creates a Bot instance, opens the image
    file, and sends it with a caption to the configured chat.

    Args:
        photo_path: Absolute path to the JPEG screenshot.
        caption:    Text to attach below the photo in the message.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    # Create a Bot instance. The token authenticates every API request.
    bot = telegram.Bot(token=TELEGRAM_TOKEN)

    try:
        # Open the screenshot in binary mode and send as a photo message.
        # Telegram compresses photos automatically; for full-resolution
        # delivery use send_document() instead (larger file, slower).
        with open(photo_path, "rb") as photo_file:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo_file,
                caption=caption,
            )
        logger.info("Telegram alert sent: %s", photo_path.name)
        return True

    except telegram.error.TelegramError as exc:
        # Catches authentication errors, rate limits, network failures, etc.
        logger.error("Telegram send failed: %s", exc)
        return False

    except OSError as exc:
        # File not found, permission denied, etc.
        logger.error("Could not open screenshot for Telegram: %s", exc)
        return False


def send_alert(photo_path: Path, caption: str = "") -> bool:
    """Send a motion-detection screenshot to Telegram.

    This is the synchronous entry point called from the main loop.
    It wraps the async send in asyncio.run() so callers don't need
    to manage an event loop.

    Args:
        photo_path: Path to the saved JPEG screenshot.
        caption:    Optional message text. Defaults to a generic alert
                    string that includes the filename (which contains
                    the timestamp).

    Returns:
        True if the alert was delivered, False on any failure.
    """
    # Build a sensible default caption if none was provided.
    if not caption:
        caption = f"Motion detected — {photo_path.name}"

    logger.info("Sending Telegram alert for: %s", photo_path.name)

    try:
        # asyncio.run() creates a new event loop, runs the coroutine to
        # completion, and then closes the loop. This is the recommended
        # way to call async code from synchronous contexts in Python 3.10+.
        return asyncio.run(_send_photo_async(photo_path, caption))

    except RuntimeError as exc:
        # If an event loop is already running (e.g. inside Jupyter or an
        # async framework), asyncio.run() raises RuntimeError. In that
        # edge case, fall back to creating a task on the existing loop.
        logger.error("asyncio.run() failed: %s — trying fallback.", exc)
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_send_photo_async(photo_path, caption))
        except Exception as inner_exc:
            logger.error("Fallback also failed: %s", inner_exc)
            return False
