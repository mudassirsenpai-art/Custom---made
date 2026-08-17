"""Detect a soft glow/halo around source text, as opposed to a hard-edged
outline or plain text with no decoration.

Used by both core.image.cleaning (speech bubbles) and
core.outside_text_processor (OSB titles/SFX) to power the "Copy Original
Text Style" glow-replication feature (RenderingConfig.detect_glow).

Approach: grow the text-ink mask outward ring by ring (1px dilation steps)
and, for each ring, measure what fraction of its pixels are still visibly
different from the surrounding background (LAB color distance). A hard
stroke outline transitions from "clearly not background" to "background"
within a ring or two. A soft blurred glow fades out gradually over several
pixels. Requiring a minimum contiguous extent before calling something a
"glow" is what tells the two apart - this module does not attempt to
distinguish glow from outline by width alone, but by how gradually the
ring colors approach the background color instead of stepping to it.
"""

from typing import Optional, Tuple, TypedDict

import cv2
import numpy as np


class GlowInfo(TypedDict):
    color: Tuple[int, int, int]  # RGB
    radius: float  # px, for use as a Skia blur "reach" (see drawing_engine)


# A ring counts as "still glowing" if at least this fraction of its pixels
# remain meaningfully different from the background color.
RING_NON_BG_RATIO_THRESHOLD = 0.5
# LAB distance below which a pixel is considered indistinguishable from the
# sampled background color.
BG_LAB_DISTANCE_THRESHOLD = 12.0
# A glow must extend at least this many 1px rings beyond the ink mask to be
# treated as a soft halo rather than a thin hard-stroke outline (which the
# existing fixed outline_width feature already covers) or plain text.
MIN_GLOW_RING_EXTENT = 3
MAX_GLOW_RING_EXTENT = 16
MIN_RING_SAMPLE_PIXELS = 6


def detect_glow_halo(
    rgb_crop: np.ndarray,
    ink_mask: np.ndarray,
    bg_rgb: Tuple[int, int, int],
) -> Optional[GlowInfo]:
    """Measure whether text in rgb_crop has a soft glow/halo around it.

    Args:
        rgb_crop: HxWx3 uint8 RGB image, a crop tightly around the text
            (same crop the ink_mask was measured from).
        ink_mask: HxW uint8 mask (255 = text ink pixels), e.g. the
            clean_mask already computed for text-color extraction.
        bg_rgb: (r, g, b) reference background color immediately around the
            text (already sampled by the caller for its own color-contrast
            purposes - reused here so this stays a single extra pass, not a
            second independent sampling step).

    Returns:
        GlowInfo with the glow's median color and an approximate radius
        (in source-crop px), or None if no soft glow was detected (plain
        text, or only a thin hard-edged outline).
    """
    if ink_mask is None or rgb_crop is None:
        return None
    if not np.any(ink_mask == 255):
        return None
    if rgb_crop.shape[:2] != ink_mask.shape[:2]:
        return None

    lab_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(
        np.uint8([[bg_rgb]]), cv2.COLOR_RGB2LAB
    )[0][0].astype(np.float32)

    kernel = np.ones((3, 3), np.uint8)
    prev_dilated = (ink_mask == 255).astype(np.uint8) * 255
    ring_ratios = []
    ring_colors = []

    for radius in range(1, MAX_GLOW_RING_EXTENT + 1):
        dilated = cv2.dilate(prev_dilated, kernel, iterations=1)
        # Cap growth at the crop edges so a text box flush against the crop
        # boundary doesn't silently starve the ring of pixels.
        ring = cv2.bitwise_and(dilated, cv2.bitwise_not(prev_dilated))
        ring_bool = ring == 255
        count = int(np.count_nonzero(ring_bool))
        if count < MIN_RING_SAMPLE_PIXELS:
            break

        ring_lab = lab_crop[ring_bool]
        dist_from_bg = np.linalg.norm(ring_lab - bg_lab, axis=1)
        non_bg_ratio = float(np.mean(dist_from_bg > BG_LAB_DISTANCE_THRESHOLD))
        ring_ratios.append(non_bg_ratio)
        ring_colors.append(np.median(rgb_crop[ring_bool], axis=0))

        prev_dilated = dilated

    if not ring_ratios:
        return None

    # Contiguous extent from the ink edge outward where rings still read as
    # "not background". A hard outline drops below threshold almost
    # immediately (extent 1-2); a soft glow fades out gradually over more
    # rings.
    extent = 0
    for ratio in ring_ratios:
        if ratio > RING_NON_BG_RATIO_THRESHOLD:
            extent += 1
        else:
            break

    if extent < MIN_GLOW_RING_EXTENT:
        return None

    glow_ring_colors = np.array(ring_colors[:extent])
    color = np.median(glow_ring_colors, axis=0).astype(int)
    return {
        "color": (int(color[0]), int(color[1]), int(color[2])),
        "radius": float(extent),
    }
