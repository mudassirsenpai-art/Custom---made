"""
Extraction of the *original* lettering style from source artwork.

The renderer normally picks its own look: text colour is derived from bubble
brightness (black on light, white on dark), the outline colour is the inverse of
the text colour, and the font size is whatever the largest size is that still
fits the bubble. That is robust but it throws away the letterer's intent - a
red SFX with a yellow keyline and a soft glow comes back as flat black text.

This module measures those attributes off the untouched page (before cleaning or
inpainting erases them) so the translated text can be drawn in the same style:

  * fill colour       - the colour of the glyph interior
  * outline colour    - the colour of the keyline/stroke around the glyph
  * outline width     - thickness of that keyline, as a Skia stroke width
  * glow colour       - the colour of a soft halo outside the keyline
  * glow radius       - how far that halo extends
  * font size         - estimated from measured cap height

Everything here is deliberately conservative: each attribute is returned only
when the measurement is unambiguous, and a per-style `confidence` lets the
caller reject weak reads. When a measurement is missing the caller keeps its own
default, so a failed extraction degrades to stock rendering rather than to a
wrong-looking page.

The returned style is a plain dict of built-in scalars so it survives the pickle
round-trip used by manual-mode checkpoints.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from utils.logging import log_message

# A region smaller than this in either axis carries too few pixels for the
# core/rim split below to mean anything.
MIN_REGION_SIDE = 8

# Fraction of the shorter side sampled as a border ring to estimate the local
# background (bubble fill, or the artwork behind outside-bubble text).
BORDER_RING_FRAC = 0.12

# Ink coverage sanity window. Below this the region is effectively empty; above
# it the "background" ring was probably itself ink, so the read is meaningless.
MIN_INK_RATIO = 0.004
MAX_INK_RATIO = 0.80

# Minimum CIELAB departure-from-background of the strongest ink before we accept
# that the region contains text at all.
MIN_INK_CONTRAST = 12.0

# Top slice of the distance map averaged by `_peak_contrast`: 0.2% of the region,
# but never fewer than 40 pixels so that small crops stay meaningful.
_PEAK_SAMPLE_FRAC = 0.002
_PEAK_SAMPLE_MIN = 40

# CIELAB distance above which two sampled colours count as deliberately
# different (roughly "clearly distinguishable" rather than JND-level).
COLOR_DISTINCT_DELTA = 22.0

# Fraction of the peak ink contrast a pixel must reach to count as fill rather
# than halo. Glow is a blend towards the page, so it always falls short of the
# lettering it surrounds; 0.85 keeps a solid fill (whose pixels sit at the peak)
# while excluding even a bright inner glow.
FILL_TIER_FRAC = 0.85

# A component spanning nearly the whole region is a glyph only if it actually
# fills the box it spans. Below this coverage it is a traced line - a bubble
# border, a panel edge - rather than lettering.
FRAME_FILL_RATIO = 0.30

# Cap height as a fraction of em size. Comic/manga display faces cluster near
# this value, so font_size ~= measured_cap_height / CAP_HEIGHT_RATIO.
CAP_HEIGHT_RATIO = 0.72

# Skia strokes straddle the glyph outline, so a keyline that reads as `t` pixels
# thick on the page needs a stroke width of 2*t.
SKIA_STROKE_DOUBLING = 2.0

OUTLINE_WIDTH_LIMITS = (0.5, 16.0)
GLOW_RADIUS_LIMITS = (1.0, 24.0)
FONT_SIZE_LIMITS = (4.0, 400.0)

_K3 = np.ones((3, 3), np.uint8)


def _border_ring_mask(height: int, width: int) -> np.ndarray:
    """Build a mask covering only the outer band of a region."""
    band = max(1, int(round(min(height, width) * BORDER_RING_FRAC)))
    ring = np.zeros((height, width), dtype=bool)
    ring[:band, :] = True
    ring[-band:, :] = True
    ring[:, :band] = True
    ring[:, -band:] = True
    return ring


def _median_color(region_bgr: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Median BGR colour of the pixels selected by a boolean/uint8 mask."""
    selected = region_bgr[mask > 0] if mask.dtype == np.uint8 else region_bgr[mask]
    if selected.size == 0:
        return None
    median = np.median(selected.reshape(-1, 3), axis=0)
    return tuple(int(round(float(c))) for c in median)


def _bgr_to_rgb(color: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    if color is None:
        return None
    return (color[2], color[1], color[0])


def _lab_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """CIELAB distance between two BGR colours."""
    lab_a = cv2.cvtColor(np.uint8([[list(a)]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
    lab_b = cv2.cvtColor(np.uint8([[list(b)]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
    return float(np.linalg.norm(lab_a - lab_b))


def _peak_contrast(dist_from_bg: np.ndarray) -> float:
    """
    How far the strongest ink in the region departs from the background.

    A plain percentile cannot be used here: lettering covers only a percent or
    two of a roomy bubble, so even the 95th percentile of the distance map is
    pure background. Averaging the top slice instead measures the ink itself
    while staying immune to the single brightest speck of scan noise.
    """
    flat = dist_from_bg.reshape(-1)
    if flat.size == 0:
        return 0.0
    sample = min(flat.size, max(_PEAK_SAMPLE_MIN, int(flat.size * _PEAK_SAMPLE_FRAC)))
    top = np.partition(flat, flat.size - sample)[flat.size - sample :]
    return float(top.mean())


def _clean_ink_mask(
    raw_ink: np.ndarray, height: int, width: int
) -> Tuple[np.ndarray, list]:
    """
    Turn a raw contrast mask into a glyph-only mask plus its component stats.

    Two kinds of non-glyph ink routinely survive contrast thresholding: speckle
    noise (killed by the area floor) and structural lines - bubble outlines,
    panel gutters, borders of the crop. A structural line gives itself away by
    spanning most of the region while filling almost none of the box it spans:
    a bubble border traced around a crop covers under a tenth of its own bounding
    box, where even a wide glyph covers a third or more of its own.
    """
    closed = cv2.morphologyEx(raw_ink, cv2.MORPH_CLOSE, _K3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    min_area = max(3.0, height * width * 1e-4)
    keep = np.zeros((height, width), dtype=np.uint8)
    kept_stats = []
    for label in range(1, count):
        x, y, w, h, area = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
            int(stats[label, cv2.CC_STAT_AREA]),
        )
        if area < min_area:
            continue
        touches_edge = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        spans_axis = w >= width * 0.8 or h >= height * 0.8
        if touches_edge and spans_axis:
            continue
        spans_region = w >= width * 0.8 and h >= height * 0.8
        if spans_region and area < w * h * FRAME_FILL_RATIO:
            continue
        keep[labels == label] = 255
        kept_stats.append({"x": x, "y": y, "w": w, "h": h, "area": area})

    return keep, kept_stats


def _estimate_font_size(component_stats: list) -> Optional[float]:
    """
    Estimate em size from the heights of the glyph components.

    Component heights are dominated by capitals/ascenders in the lettering this
    runs on (manga and comic display faces are largely caps), so the median
    height of the non-trivial components approximates cap height. Components
    shorter than a quarter of the tallest are punctuation, tittles or diacritics
    and would drag that median down, so they are excluded.
    """
    heights = [c["h"] for c in component_stats if c["h"] > 0]
    if not heights:
        return None

    tallest = max(heights)
    glyph_heights = [h for h in heights if h >= tallest * 0.25]
    if not glyph_heights:
        return None

    cap_height = float(np.median(glyph_heights))
    if cap_height <= 0:
        return None
    return cap_height / CAP_HEIGHT_RATIO


def _ring_profile(
    region_bgr: np.ndarray,
    ink: np.ndarray,
    dist_from_bg: np.ndarray,
    max_radius: int,
) -> list:
    """
    Sample one-pixel concentric rings just outside the ink.

    Each entry is (mean departure-from-background, median BGR colour) for the
    pixels exactly `radius` away from the nearest ink pixel. Sampling stops as
    soon as a ring is too small to average meaningfully.
    """
    dist_from_ink = cv2.distanceTransform(cv2.bitwise_not(ink), cv2.DIST_L2, 5)
    rings = []
    for radius in range(1, max_radius + 1):
        ring = (dist_from_ink > radius - 1) & (dist_from_ink <= radius)
        if np.count_nonzero(ring) < 8:
            break
        rings.append((float(np.mean(dist_from_bg[ring])), _median_color(region_bgr, ring)))
    return rings


def _mean_color(colors: list) -> Optional[Tuple[int, int, int]]:
    """Average a list of sampled BGR colours, ignoring the ones that failed."""
    usable = [c for c in colors if c is not None]
    if not usable:
        return None
    mean = np.mean(np.asarray(usable, dtype=np.float32), axis=0)
    return tuple(int(round(float(c))) for c in mean)


def _measure_halo(
    region_bgr: np.ndarray,
    ink: np.ndarray,
    dist_from_bg: np.ndarray,
    stroke_width: float,
    ink_contrast: float,
    fill_bgr: Tuple[int, int, int],
) -> Tuple[Optional[Tuple[int, int, int]], float, Optional[Tuple[int, int, int]], float]:
    """
    Read whatever surrounds the glyph ink: a hard keyline, a soft glow, or both.

    Both live in the same annulus, so they cannot be told apart by strength -
    a bright keyline and a bright glow look identical to a threshold. They are
    told apart by the shape of the ring profile instead. A keyline holds one
    colour at a near-constant strength and then stops dead: a plateau ending in
    a cliff, with at most the single blended pixel antialiasing leaves behind.
    A glow keeps fading, so its plateau trails off over several more rings.

    Returns (outline_bgr, outline_band_px, glow_bgr, glow_radius_px), with None
    and 0.0 for whichever of the two is not present.
    """
    none_found = (None, 0.0, None, 0.0)

    max_radius = int(min(max(4.0, stroke_width * 2.5), GLOW_RADIUS_LIMITS[1] + 4))
    rings = _ring_profile(region_bgr, ink, dist_from_bg, max_radius)
    if not rings:
        return none_found

    inner_contrast = rings[0][0]
    # Anything this close to the background is scan noise or JPEG ringing.
    floor = max(MIN_INK_CONTRAST * 0.5, ink_contrast * 0.12)
    if inner_contrast < floor:
        return none_found

    # The plateau: leading rings holding roughly the innermost strength.
    plateau = 0
    for contrast, _ in rings:
        if contrast < inner_contrast * 0.60 or contrast > inner_contrast * 1.15:
            break
        plateau += 1

    # The tail: how much further the annulus keeps fading before it dies out.
    tail = 0
    previous = rings[plateau - 1][0] if plateau else inner_contrast
    for contrast, _ in rings[plateau:]:
        if contrast < floor or contrast > previous * 1.10:
            break
        tail += 1
        previous = contrast

    # Ring 1 straddles the fill/keyline boundary, so its colour is a blend of
    # the two; skip it whenever there are enough rings left to sample without it.
    band_colors = [c for _, c in rings[1:plateau]] if plateau >= 3 else [c for _, c in rings[:plateau]]
    band_bgr = _mean_color(band_colors)
    tail_bgr = _mean_color([c for _, c in rings[plateau : plateau + tail]])

    if tail <= 1:
        # A cliff: hard keyline, no halo. One plateau pixel is indistinguishable
        # from antialiasing, so it is not reported as a keyline.
        if plateau >= 2 and band_bgr is not None:
            if _lab_distance(band_bgr, fill_bgr) > COLOR_DISTINCT_DELTA:
                return band_bgr, float(plateau), None, 0.0
        return none_found

    # A fading tail: soft halo. Its visible extent is measured from the ink edge,
    # which is what the renderer's blur radius has to reproduce.
    glow_bgr = band_bgr if tail_bgr is None else _mean_color([band_bgr, tail_bgr])
    glow_radius = float(plateau + tail)
    if glow_bgr is None or _lab_distance(glow_bgr, fill_bgr) <= COLOR_DISTINCT_DELTA:
        # A halo the colour of the text is just the text's own soft edge.
        return none_found

    outline_bgr: Optional[Tuple[int, int, int]] = None
    outline_band = 0.0
    if (
        plateau >= 2
        and band_bgr is not None
        and tail_bgr is not None
        and _lab_distance(band_bgr, tail_bgr) > COLOR_DISTINCT_DELTA
        and _lab_distance(band_bgr, fill_bgr) > COLOR_DISTINCT_DELTA
    ):
        # A flat band of its own colour inside the halo: keyline *and* glow.
        outline_bgr, outline_band = band_bgr, float(plateau)
        glow_bgr = tail_bgr

    return outline_bgr, outline_band, glow_bgr, glow_radius


def extract_text_style(
    region_bgr: np.ndarray,
    *,
    verbose: bool = False,
    label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Measure the lettering style of the text inside an untouched image region.

    Args:
        region_bgr: BGR crop taken from the page *before* cleaning/inpainting.
        verbose: Whether to log the measurement breakdown.
        label: Identifier used in log lines (bubble id, bbox, ...).

    Returns:
        A style dict, or None when the region holds no legible text. Keys:
            fill_rgb, outline_rgb, glow_rgb: (r, g, b) or None
            outline_width, glow_radius: float px (0.0 = absent)
            font_size_px: float or None
            stroke_width_px: measured total ink stroke thickness
            ink_bbox: (x1, y1, x2, y2) of the ink within the region, or None
            region_wh: (width, height) of the region measured
            line_count, component_count: ints
            confidence: 0.0-1.0
    """
    if region_bgr is None or region_bgr.ndim != 3 or region_bgr.shape[2] != 3:
        return None

    height, width = region_bgr.shape[:2]
    if min(height, width) < MIN_REGION_SIDE:
        return None

    try:
        lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    except cv2.error:
        return None

    ring = _border_ring_mask(height, width)
    background_lab = np.median(lab[ring].reshape(-1, 3), axis=0)
    dist_from_bg = np.linalg.norm(lab - background_lab, axis=2)

    ink_contrast = _peak_contrast(dist_from_bg)
    if ink_contrast < MIN_INK_CONTRAST:
        log_message(
            f"  - Style probe {label}: no usable contrast (peak={ink_contrast:.1f})",
            verbose=verbose,
        )
        return None

    threshold = max(18.0, ink_contrast * 0.55)
    raw_ink = (dist_from_bg > threshold).astype(np.uint8) * 255
    ink, component_stats = _clean_ink_mask(raw_ink, height, width)

    ink_count = int(np.count_nonzero(ink))
    ink_ratio = ink_count / float(height * width)
    if not component_stats or not (MIN_INK_RATIO <= ink_ratio <= MAX_INK_RATIO):
        log_message(
            f"  - Style probe {label}: implausible ink coverage ({ink_ratio:.3f})",
            verbose=verbose,
        )
        return None

    return _build_style(
        region_bgr,
        ink,
        component_stats,
        dist_from_bg,
        ink_contrast,
        ink_count,
        ink_ratio,
        (width, height),
        verbose=verbose,
        label=label,
    )


def _fill_sample_mask(
    core: np.ndarray,
    dist_from_bg: np.ndarray,
    ink_contrast: float,
) -> np.ndarray:
    """
    Pick the pixels to read the fill colour from, inside the eroded glyph core.

    A soft glow lifts its whole halo above the ink threshold, so the mask is far
    fatter than the glyph and the erosion - sized from that mask - can land in
    the halo instead of the lettering. That is how white text with a blue glow
    used to read back as light blue. The fill is always the strongest-contrast
    ink on the page, the glow a blend towards the background, so keeping only
    the top-contrast tier of the core pins the sample to the lettering. Falls
    back to the whole core when too little of it is top-tier, which is the
    normal case for plain text where the core is uniform anyway.
    """
    strong = ((core > 0) & (dist_from_bg >= ink_contrast * FILL_TIER_FRAC)).astype(np.uint8) * 255
    strong_count = int(np.count_nonzero(strong))
    core_count = int(np.count_nonzero(core))
    if strong_count >= 12 and strong_count >= core_count * 0.05:
        return strong
    return core


def _build_style(
    region_bgr: np.ndarray,
    ink: np.ndarray,
    component_stats: list,
    dist_from_bg: np.ndarray,
    ink_contrast: float,
    ink_count: int,
    ink_ratio: float,
    region_wh: Tuple[int, int],
    *,
    verbose: bool,
    label: str,
) -> Optional[Dict[str, Any]]:
    """Split the ink into interior/keyline/halo and read a colour off each."""
    # Stroke thickness from the ink's medial axis: the distance transform peaks
    # at the glyph centreline, so its high percentile is the stroke half-width.
    ink_dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    stroke_half = float(np.percentile(ink_dt[ink > 0], 85))
    stroke_width = max(1.0, stroke_half * 2.0)

    # Erode ~60% of the way into the stroke to isolate the glyph interior. On
    # hairline text this erases everything, in which case the whole ink is the
    # interior and there is no separable keyline to measure.
    erode_iters = max(1, int(round(stroke_half * 0.6)))
    core = cv2.erode(ink, _K3, iterations=erode_iters)
    core_is_fallback = int(np.count_nonzero(core)) < 10
    if core_is_fallback:
        core = ink.copy()

    fill_bgr = _median_color(region_bgr, _fill_sample_mask(core, dist_from_bg, ink_contrast))
    if fill_bgr is None:
        return None

    # A keyline can sit on either side of the ink threshold. When it contrasts
    # with the page as strongly as the fill does, it is inside the ink mask and
    # shows up as a rim around the eroded interior. When it does not - a yellow
    # keyline on white paper around a red fill - it falls outside the mask and
    # only the ring walk below can see it.
    outline_bgr: Optional[Tuple[int, int, int]] = None
    outline_band = 0.0
    if not core_is_fallback:
        # Sample the rim one pixel in from the ink edge: that outermost pixel is
        # the antialiased blend into the page, and reading it would turn every
        # plain glyph's soft edge into an imaginary keyline.
        solid = cv2.erode(ink, _K3, iterations=1)
        rim = cv2.bitwise_and(solid, cv2.bitwise_not(cv2.dilate(core, _K3, iterations=1)))
        rim_count = int(np.count_nonzero(rim))
        if rim_count >= max(12, int(ink_count * 0.12)):
            candidate = _median_color(region_bgr, rim)
            if (
                candidate is not None
                and _lab_distance(candidate, fill_bgr) > COLOR_DISTINCT_DELTA
            ):
                # The keyline band is what the stroke has that the interior does
                # not.
                core_half = float(
                    np.percentile(cv2.distanceTransform(core, cv2.DIST_L2, 5)[core > 0], 85)
                )
                band = max(0.0, stroke_half - core_half)
                if band > 0.0:
                    outline_bgr, outline_band = candidate, band

    halo_outline_bgr, halo_outline_band, glow_bgr, glow_radius = _measure_halo(
        region_bgr, ink, dist_from_bg, stroke_width, ink_contrast, fill_bgr
    )
    if outline_bgr is None and halo_outline_bgr is not None:
        outline_bgr, outline_band = halo_outline_bgr, halo_outline_band

    # Skia centres strokes on the glyph outline, so a band that reads as `t`
    # pixels wide on the page needs a stroke width of 2t.
    outline_width = 0.0
    if outline_bgr is not None:
        outline_width = min(
            max(outline_band * SKIA_STROKE_DOUBLING, OUTLINE_WIDTH_LIMITS[0]),
            OUTLINE_WIDTH_LIMITS[1],
        )

    if glow_bgr is not None:
        glow_radius = min(
            max(glow_radius, GLOW_RADIUS_LIMITS[0]), GLOW_RADIUS_LIMITS[1]
        )
        # A halo the same colour as the keyline is that keyline's antialiasing.
        if (
            outline_bgr is not None
            and _lab_distance(glow_bgr, outline_bgr) <= COLOR_DISTINCT_DELTA
        ):
            glow_bgr, glow_radius = None, 0.0
    else:
        glow_radius = 0.0

    font_size = _estimate_font_size(component_stats)
    region_w, region_h = region_wh
    if font_size is not None:
        # A glyph cannot be taller than the region that contains it; a reading
        # that says otherwise measured something else.
        if not (FONT_SIZE_LIMITS[0] <= font_size <= min(FONT_SIZE_LIMITS[1], region_h * 1.25)):
            font_size = None

    xs1 = min(c["x"] for c in component_stats)
    ys1 = min(c["y"] for c in component_stats)
    xs2 = max(c["x"] + c["w"] for c in component_stats)
    ys2 = max(c["y"] + c["h"] for c in component_stats)

    style = {
        "fill_rgb": _bgr_to_rgb(fill_bgr),
        "outline_rgb": _bgr_to_rgb(outline_bgr),
        "outline_width": round(float(outline_width), 2),
        "glow_rgb": _bgr_to_rgb(glow_bgr),
        "glow_radius": round(float(glow_radius), 2),
        "font_size_px": round(float(font_size), 2) if font_size else None,
        "stroke_width_px": round(stroke_width, 2),
        "ink_bbox": (int(xs1), int(ys1), int(xs2), int(ys2)),
        "region_wh": (int(region_w), int(region_h)),
        "component_count": len(component_stats),
        "line_count": _count_text_lines(component_stats),
        "confidence": _score_confidence(
            component_stats, ink_ratio, ink_contrast, core_is_fallback, region_wh
        ),
    }

    log_message(f"  - Style probe {label}: {describe_style(style)}", verbose=verbose)
    return style


def _count_text_lines(component_stats: list) -> int:
    """Group glyph components into text lines by vertical overlap."""
    if not component_stats:
        return 0

    ordered = sorted(component_stats, key=lambda c: c["y"] + c["h"] / 2.0)
    lines = 1
    line_top = ordered[0]["y"]
    line_bottom = ordered[0]["y"] + ordered[0]["h"]
    for comp in ordered[1:]:
        top, bottom = comp["y"], comp["y"] + comp["h"]
        overlap = min(bottom, line_bottom) - max(top, line_top)
        if overlap >= 0.4 * min(comp["h"], line_bottom - line_top):
            line_top = min(line_top, top)
            line_bottom = max(line_bottom, bottom)
        else:
            lines += 1
            line_top, line_bottom = top, bottom
    return lines


def _score_confidence(
    component_stats: list,
    ink_ratio: float,
    ink_contrast: float,
    core_is_fallback: bool,
    region_wh: Tuple[int, int],
) -> float:
    """
    Rate how much the caller should trust this measurement.

    Each factor is a multiplier rather than an additive term so that any single
    bad signal is enough to push the style below a caller's threshold.
    """
    score = 1.0

    # One or two components is as likely to be a stray mark as it is lettering.
    if len(component_stats) < 3:
        score *= 0.55
    elif len(component_stats) < 6:
        score *= 0.85

    # Very sparse or very dense ink both mean the background estimate is shaky.
    if ink_ratio < 0.01 or ink_ratio > 0.55:
        score *= 0.7

    # Weak separation from the background makes every sampled colour noisy.
    if ink_contrast < 25.0:
        score *= 0.6
    elif ink_contrast < 40.0:
        score *= 0.85

    # Hairline text: no interior survived erosion, so fill/outline could not be
    # told apart and the fill colour is an average of both.
    if core_is_fallback:
        score *= 0.7

    # Small crops give the morphology too little to work with.
    if min(region_wh) < 24:
        score *= 0.75

    return round(min(max(score, 0.0), 1.0), 3)


def describe_style(style: Optional[Dict[str, Any]]) -> str:
    """One-line human-readable summary of a style dict, for logs."""
    if not style:
        return "none"
    parts = [f"fill={style.get('fill_rgb')}"]
    if style.get("outline_rgb"):
        parts.append(f"outline={style['outline_rgb']}@{style.get('outline_width')}px")
    if style.get("glow_rgb"):
        parts.append(f"glow={style['glow_rgb']}r{style.get('glow_radius')}")
    if style.get("font_size_px"):
        parts.append(f"font~{style['font_size_px']}px")
    parts.append(f"lines={style.get('line_count')}")
    parts.append(f"conf={style.get('confidence')}")
    return ", ".join(parts)


class StyleOverrides:
    """
    Rendering parameters derived from a measured original style.

    Every field is "keep the caller's value" when None/unset, so a partial
    measurement only overrides the parts it actually established.
    """

    __slots__ = (
        "text_color_rgb",
        "outline_color_rgb",
        "outline_width",
        "glow_color_rgb",
        "glow_radius",
        "max_font_size",
        "applied",
    )

    def __init__(self):
        self.text_color_rgb: Optional[Tuple[int, int, int]] = None
        self.outline_color_rgb: Optional[Tuple[int, int, int]] = None
        self.outline_width: Optional[float] = None
        self.glow_color_rgb: Optional[Tuple[int, int, int]] = None
        self.glow_radius: Optional[float] = None
        self.max_font_size: Optional[int] = None
        self.applied: list = []

    def __bool__(self) -> bool:
        return bool(self.applied)

    def summary(self) -> str:
        return "+".join(self.applied) if self.applied else "none"


def resolve_style_overrides(
    style: Optional[Dict[str, Any]],
    *,
    base_min_font: int,
    base_max_font: int,
    tolerance: float = 0.25,
    min_confidence: float = 0.35,
    hard_max_font: int = 384,
) -> StyleOverrides:
    """
    Convert a measured style into concrete render overrides.

    Font size is applied as a *ceiling*, not a fixed size: the layout engine
    already binary-searches downward from the ceiling, so asking for the
    original size means "render this big if the translation fits, otherwise
    shrink as usual". Translations are routinely longer than the source text, so
    pinning the size exactly would clip; letting it shrink is the graceful path.

    Args:
        style: Style dict from extract_text_style, or None.
        base_min_font: Caller's current minimum font size (never raised above
            the derived ceiling, so shrink-to-fit keeps working).
        base_max_font: Caller's current maximum font size.
        tolerance: Headroom above the measured size, as a fraction.
        min_confidence: Styles scoring below this are ignored entirely.
        hard_max_font: Absolute ceiling for the derived font size.

    Returns:
        StyleOverrides; falsy when nothing could be applied.
    """
    overrides = StyleOverrides()
    if not style:
        return overrides
    if float(style.get("confidence") or 0.0) < min_confidence:
        return overrides

    fill = style.get("fill_rgb")
    if fill:
        overrides.text_color_rgb = tuple(int(c) for c in fill)
        overrides.applied.append("fill")

    outline = style.get("outline_rgb")
    outline_width = float(style.get("outline_width") or 0.0)
    if outline and outline_width > 0.0:
        overrides.outline_color_rgb = tuple(int(c) for c in outline)
        overrides.outline_width = outline_width
        overrides.applied.append("outline")

    glow = style.get("glow_rgb")
    glow_radius = float(style.get("glow_radius") or 0.0)
    if glow and glow_radius > 0.0:
        overrides.glow_color_rgb = tuple(int(c) for c in glow)
        overrides.glow_radius = glow_radius
        overrides.applied.append("glow")

    font_size = style.get("font_size_px")
    if font_size:
        ceiling = int(round(float(font_size) * (1.0 + max(0.0, tolerance))))
        ceiling = max(base_min_font, min(ceiling, hard_max_font))
        if ceiling != base_max_font:
            overrides.max_font_size = ceiling
            overrides.applied.append("size")

    return overrides
