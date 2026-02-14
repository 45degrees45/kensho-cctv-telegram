# =============================================================================
# CCTV Alert System - RTSP Stream Handler
# v1.0.0
# =============================================================================
# Manages the OpenCV VideoCapture connection to an RTSP camera.
# Encapsulates connect / read / reconnect logic so the rest of the app
# never touches cv2.VideoCapture directly. Reconnect behaviour is
# configurable via environment variables (see config.py).
# =============================================================================

from __future__ import annotations

import time
import logging

import cv2
import numpy as np

from app.config import RTSP_URL, RECONNECT_DELAY_SECONDS, MAX_RECONNECT_ATTEMPTS

# Module-level logger. Inherits the root configuration set in main.py.
logger = logging.getLogger(__name__)


class CameraStream:
    """Persistent wrapper around an RTSP video capture.

    Usage::

        stream = CameraStream()
        stream.connect()          # blocks until the stream is live

        frame = stream.read()     # returns a numpy frame or None
        if frame is None:
            stream.reconnect()    # tries up to MAX_RECONNECT_ATTEMPTS

        stream.release()          # clean shutdown
    """

    def __init__(self, rtsp_url: str = RTSP_URL) -> None:
        """Store the RTSP URL and prepare an empty capture handle.

        Args:
            rtsp_url: Full RTSP address including credentials and path.
                      Defaults to the value loaded from the environment.
        """
        self._url: str = rtsp_url

        # _cap will hold the cv2.VideoCapture object once connect() is called.
        self._cap: cv2.VideoCapture | None = None

        # Mask the password in log output so credentials don't leak.
        self._safe_url = self._mask_url(rtsp_url)

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def connect(self) -> None:
        """Open the RTSP stream and verify that frames can be read.

        Tries multiple OpenCV backends in order of preference:
          1. FFMPEG  — best RTSP support, but may not be compiled into
                       pip-installed OpenCV on macOS.
          2. GSTREAMER — good alternative if installed via homebrew.
          3. ANY     — let OpenCV auto-detect the best available backend.

        The environment variable OPENCV_FFMPEG_CAPTURE_OPTIONS is set
        to prefer TCP transport, which is more reliable than UDP for
        most cameras and prevents packet-loss artefacts.

        Raises:
            ConnectionError: If no backend can open the stream.
        """
        import os

        logger.info("Connecting to RTSP stream: %s", self._safe_url)

        # Hint for the FFMPEG backend: use RTSP-over-TCP. TCP is slower
        # to start but far more reliable than UDP on home/office networks.
        # This env var is read by OpenCV's FFMPEG wrapper if that backend
        # is selected. Harmless if another backend is used instead.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Try backends in order of preference. On macOS with pip-installed
        # opencv-python, CAP_FFMPEG often fails because the bundled FFmpeg
        # lacks full RTSP support. CAP_ANY lets OpenCV pick whatever works.
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_GSTREAMER, "GSTREAMER"),
            (cv2.CAP_ANY, "ANY"),
        ]

        for backend_id, backend_name in backends:
            logger.info("Trying backend: %s", backend_name)
            cap = cv2.VideoCapture(self._url, backend_id)

            if cap.isOpened():
                # Verify we can actually read a frame, not just "open".
                ret, _ = cap.read()
                if ret:
                    self._cap = cap
                    logger.info(
                        "Stream connected successfully via %s backend.",
                        backend_name,
                    )
                    return
                else:
                    logger.warning(
                        "%s backend opened but couldn't read a frame.", backend_name
                    )
                    cap.release()
            else:
                logger.warning("%s backend failed to open stream.", backend_name)

        raise ConnectionError(
            f"Failed to open RTSP stream with any backend: {self._safe_url}"
        )

    def read(self) -> np.ndarray | None:
        """Grab and return one frame from the stream.

        Returns:
            A BGR numpy array on success, or None if the read failed
            (indicating the stream may have dropped).
        """
        if self._cap is None:
            logger.warning("read() called before connect(). Returning None.")
            return None

        # ret is a boolean indicating whether the frame was captured.
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Frame read failed — stream may have dropped.")
            return None
        return frame

    def reconnect(self) -> None:
        """Release the current capture and try to re-establish the stream.

        Retries up to MAX_RECONNECT_ATTEMPTS times with a configurable
        delay between each attempt. If all attempts fail, raises
        ConnectionError to let the caller decide how to handle it
        (e.g. exit or fall back to a secondary camera).

        Raises:
            ConnectionError: After exhausting all reconnection attempts.
        """
        logger.warning(
            "Starting reconnection sequence (max %d attempts, %ds delay).",
            MAX_RECONNECT_ATTEMPTS,
            RECONNECT_DELAY_SECONDS,
        )

        # Always release the old capture first to free OS resources.
        self.release()

        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            logger.info(
                "Reconnect attempt %d / %d …", attempt, MAX_RECONNECT_ATTEMPTS
            )
            time.sleep(RECONNECT_DELAY_SECONDS)

            try:
                self.connect()
                # Verify the stream actually delivers a frame.
                test_frame = self.read()
                if test_frame is not None:
                    logger.info("Reconnected successfully on attempt %d.", attempt)
                    return
                else:
                    logger.warning("Connected but no frame received. Retrying.")
                    self.release()
            except ConnectionError:
                logger.warning("Attempt %d failed.", attempt)

        # If we land here, every attempt was unsuccessful.
        raise ConnectionError(
            f"Unable to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts."
        )

    def release(self) -> None:
        """Release the underlying VideoCapture and free resources.

        Safe to call multiple times or before connect().
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera stream released.")

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _mask_url(url: str) -> str:
        """Replace the password portion of an RTSP URL with asterisks.

        Example::
            rtsp://admin:secret@192.168.1.1/stream
            -> rtsp://admin:****@192.168.1.1/stream

        If the URL does not contain credentials the original is returned.
        """
        # Split on '://' to isolate the scheme.
        if "://" not in url:
            return url

        scheme, rest = url.split("://", 1)

        # Credentials sit before the '@' sign.
        if "@" not in rest:
            return url

        creds, host = rest.split("@", 1)
        if ":" in creds:
            user, _ = creds.split(":", 1)
            return f"{scheme}://{user}:****@{host}"

        return url
