# =============================================================================
# CCTV Alert System - Motion Detector
# v1.0.0
# =============================================================================
# Implements motion detection using frame differencing with OpenCV.
#
# Algorithm — simple and lightweight, ideal for CCTV:
#   1. Capture frame1 and frame2 (consecutive frames).
#   2. Convert both to grayscale (removes colour noise).
#   3. Apply Gaussian blur to suppress sensor noise.
#   4. difference = abs(frame1 - frame2).
#   5. Threshold the difference into a binary mask.
#   6. Dilate to fill gaps between changed pixels.
#   7. Find contours in the binary mask.
#   8. If any contour area > MOTION_THRESHOLD → motion detected.
#
# The detector is stateful: it keeps the previous frame and the timestamp
# of the last alert so it can enforce a cooldown period.
# =============================================================================

from __future__ import annotations

import time
import logging

import cv2
import numpy as np

from app.config import MOTION_THRESHOLD, COOLDOWN_SECONDS

# Module-level logger.
logger = logging.getLogger(__name__)


class MotionDetector:
    """Detect motion between consecutive video frames using contour area.

    Typical usage inside the main loop::

        detector = MotionDetector()
        for frame in frames:
            if detector.detect(frame):
                # handle alert
    """

    def __init__(
        self,
        threshold: int = MOTION_THRESHOLD,
        cooldown: int = COOLDOWN_SECONDS,
    ) -> None:
        """Initialise the detector with tuning parameters.

        Args:
            threshold: Minimum contour area (in pixels) to count as real
                       motion. Contours smaller than this are treated as
                       noise and ignored. Default 5000.
            cooldown:  Minimum seconds between successive True returns.
                       Prevents alert flooding during sustained motion.
        """
        # The grayscale + blurred version of the previous frame, used for
        # differencing. Starts as None; the first frame is always a
        # "learning" frame and will never trigger motion.
        self._prev_gray: np.ndarray | None = None

        # Tunables.
        self._threshold: int = threshold
        self._cooldown: int = cooldown

        # Timestamp (epoch seconds) of the last positive detection.
        # Initialised to 0 so the very first motion event is never
        # suppressed by cooldown.
        self._last_alert_time: float = 0.0

        logger.info(
            "MotionDetector initialised — threshold=%d px area, cooldown=%ds.",
            self._threshold,
            self._cooldown,
        )

    def detect(self, frame: np.ndarray) -> bool:
        """Analyse a single BGR frame for motion.

        Args:
            frame: A BGR image (numpy array) from the camera stream.

        Returns:
            True if a contour exceeding the threshold area is found AND
            the cooldown period has elapsed since the last alert. False
            otherwise.
        """
        # --- Step 1: Convert to grayscale -----------------------------------
        # Grayscale reduces the data from 3 channels to 1, making the
        # difference computation faster and less sensitive to colour shifts
        # caused by auto white-balance adjustments.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Gaussian blur ------------------------------------------
        # A 21x21 kernel smooths out camera sensor noise and tiny
        # irrelevant pixel fluctuations (e.g. compression artefacts).
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # --- Step 3: Store first frame as reference -------------------------
        # On the very first call we have no previous frame to compare, so
        # we simply store this one and return False (no motion).
        if self._prev_gray is None:
            self._prev_gray = gray
            logger.debug("First frame stored as reference — no detection yet.")
            return False

        # --- Step 4: Absolute difference ------------------------------------
        # difference = abs(frame1 - frame2)
        # Each pixel value becomes |current - previous|. Unchanged areas
        # become 0 (black); changed areas become bright.
        delta = cv2.absdiff(self._prev_gray, gray)

        # --- Step 5: Binary threshold ---------------------------------------
        # Pixels with a difference > 25 (out of 255) are set to 255 (white).
        # The value 25 filters out minor luminance drift while still
        # catching real movement.
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

        # --- Step 6: Dilation -----------------------------------------------
        # Dilate the white regions to merge nearby blobs into contiguous
        # areas, reducing false fragmentation. Two iterations is a good
        # balance between filling gaps and not over-inflating.
        thresh = cv2.dilate(thresh, None, iterations=2)

        # --- Step 7: Find contours ------------------------------------------
        # findContours returns a list of contour outlines. Each contour is
        # an array of (x,y) points forming the boundary of a white region.
        # RETR_EXTERNAL: only outermost contours (ignore nested holes).
        # CHAIN_APPROX_SIMPLE: compress straight segments to save memory.
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # --- Step 8: Update the reference frame -----------------------------
        # Always update so the detector compares consecutive frames, not
        # the current frame against a stale baseline. This makes it
        # responsive to gradual lighting changes (day/night transition).
        self._prev_gray = gray

        # --- Step 9: Check contour areas ------------------------------------
        # If any single contour area > threshold → motion detected.
        # This filters out small noise blobs (insects, pixel flicker) while
        # catching real objects (people, cars, animals).
        motion_detected = False
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
            if area > self._threshold:
                motion_detected = True
                break  # One big contour is enough — no need to check the rest.

        if not motion_detected:
            return False

        # --- Step 10: Cooldown check ----------------------------------------
        # Motion is real, but we only fire an alert if enough time has
        # passed since the last one to avoid spamming Telegram.
        now = time.time()
        elapsed = now - self._last_alert_time

        if elapsed < self._cooldown:
            logger.debug(
                "Motion (area=%d) suppressed by cooldown (%ds remaining).",
                max_area,
                int(self._cooldown - elapsed),
            )
            return False

        # --- Motion confirmed and cooldown clear ----------------------------
        logger.info(
            "Motion detected — contour area=%d px (threshold=%d px).",
            max_area,
            self._threshold,
        )
        self._last_alert_time = now
        return True
