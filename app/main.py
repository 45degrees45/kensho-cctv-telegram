# =============================================================================
# CCTV Alert System - Application Entry Point
# v2.0.0
# =============================================================================
# Orchestrates the full monitoring pipeline:
#
#   1. Load and validate configuration from .env.
#   2. Initialise the SQLite database.
#   3. Connect to the RTSP camera stream.
#   4. Load YOLO object detection model.
#   5. Enter the main loop (throttled to ~3 FPS):
#      a. Read a frame from the stream.
#      b. Run YOLO detection (people, vehicles, animals).
#      c. If objects detected and cooldown clear ->
#         annotate frame with bounding boxes -> save screenshot ->
#         send Telegram alert with caption -> log to database.
#      d. On stream failure -> attempt automatic reconnection.
#   6. On shutdown (Ctrl-C) -> release resources cleanly.
#
# Run with:  python app/main.py   or   python run.py
# =============================================================================

import sys
import os
import time
import signal
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path fix so "python app/main.py" works from the project root.
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import setup_logging, validate_config, YOLO_INTERVAL, COOLDOWN_SECONDS
from app.camera.stream import CameraStream
from app.camera.detector import YOLODetector
from app.storage.database import init_db, insert_event, close_db
from app.storage.screenshots import save_screenshot
from app.alerts.telegram import send_alert

logger = logging.getLogger(__name__)

_shutdown_requested: bool = False


def _handle_signal(signum: int, _frame) -> None:
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — shutting down gracefully …", sig_name)
    _shutdown_requested = True


def main() -> None:
    """Application entry point — sets up resources and runs the detection loop."""

    # ------------------------------------------------------------------
    # Step 1: Logging
    # ------------------------------------------------------------------
    setup_logging()
    logger.info("=" * 60)
    logger.info("CCTV Alert System v2.0.0 starting up")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 2: Configuration validation
    # ------------------------------------------------------------------
    try:
        validate_config()
    except EnvironmentError as exc:
        logger.critical(str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Register signal handlers
    # ------------------------------------------------------------------
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ------------------------------------------------------------------
    # Step 4: Initialise subsystems
    # ------------------------------------------------------------------
    init_db()
    stream = CameraStream()
    yolo = YOLODetector()

    # ------------------------------------------------------------------
    # Step 5: Connect to the camera
    # ------------------------------------------------------------------
    try:
        stream.connect()
    except ConnectionError as exc:
        logger.critical("Initial connection failed: %s", exc)
        close_db()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Main detection loop
    # ------------------------------------------------------------------
    logger.info("Entering YOLO detection loop (~%.0f FPS). Press Ctrl-C to stop.", 1.0 / YOLO_INTERVAL)

    last_inference_time: float = 0.0
    last_alert_time: float = 0.0

    try:
        while not _shutdown_requested:
            # --- 6a: Read a frame from the stream -----------------------
            frame = stream.read()

            if frame is None:
                logger.warning("Stream dropped — starting reconnection.")
                try:
                    stream.reconnect()
                    continue
                except ConnectionError as exc:
                    logger.critical("Reconnection failed: %s", exc)
                    break

            # --- 6b: Throttle to configured FPS -------------------------
            now = time.time()
            if now - last_inference_time < YOLO_INTERVAL:
                time.sleep(0.01)
                continue
            last_inference_time = now

            # --- 6c: YOLO detection -------------------------------------
            detections = yolo.detect(frame)

            if not detections:
                continue

            # --- 6d: Cooldown check -------------------------------------
            if now - last_alert_time < COOLDOWN_SECONDS:
                continue

            last_alert_time = now

            # --- 6e: Detection confirmed — execute alert pipeline -------
            caption = yolo.build_caption(detections)
            logger.info(">>> ALERT: %s", caption)

            annotated = yolo.annotate(frame, detections)

            try:
                screenshot_path = save_screenshot(annotated)
            except IOError as exc:
                logger.error("Screenshot save failed: %s", exc)
                continue

            notified = send_alert(screenshot_path, caption=caption)
            insert_event(screenshot_path, notified, detected_objects=caption)

    except Exception as exc:
        logger.exception("Unexpected error in main loop: %s", exc)

    # ------------------------------------------------------------------
    # Step 7: Cleanup
    # ------------------------------------------------------------------
    logger.info("Shutting down …")
    stream.release()
    close_db()
    logger.info("CCTV Alert System stopped. Goodbye.")


# ---------------------------------------------------------------------------
# Allow running directly with: python app/main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
