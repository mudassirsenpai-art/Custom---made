"""Tests for original-lettering-style extraction (core.text.style_extraction)."""

import importlib.util
import pathlib
import sys

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "core" / "text" / "style_extraction.py"
)
# Loaded by path on purpose: importing `core.text` pulls in the detection stack
# (torch), which this module does not need and which is not installed for tests.
_spec = importlib.util.spec_from_file_location("style_extraction_under_test", _MODULE_PATH)
style_extraction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(style_extraction)

extract_text_style = style_extraction.extract_text_style
resolve_style_overrides = style_extraction.resolve_style_overrides
describe_style = style_extraction.describe_style

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SAMPLE_TEXT = "HELLO"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _to_bgr(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _render(
    *,
    size=(320, 120),
    background=(255, 255, 255),
    fill=(0, 0, 0),
    font_size=48,
    stroke_width=0,
    stroke_fill=None,
    glow=None,
    glow_radius=0,
    text=SAMPLE_TEXT,
):
    """Draw a text sample the way a letterer would, for the extractor to read back."""
    canvas = Image.new("RGB", size, background)

    if glow is not None and glow_radius:
        halo = Image.new("RGB", size, background)
        ImageDraw.Draw(halo).text(
            (size[0] // 2, size[1] // 2),
            text,
            font=_font(font_size),
            fill=glow,
            anchor="mm",
            stroke_width=glow_radius,
            stroke_fill=glow,
        )
        canvas = halo.filter(ImageFilter.GaussianBlur(glow_radius * 0.9))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (size[0] // 2, size[1] // 2),
        text,
        font=_font(font_size),
        fill=fill,
        anchor="mm",
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return _to_bgr(canvas)


def _close(actual, expected, tolerance=40):
    """Colours survive antialiasing and median sampling only approximately."""
    assert actual is not None, "expected a colour, got None"
    for a, e in zip(actual, expected):
        assert abs(int(a) - int(e)) <= tolerance, f"{actual} not within {tolerance} of {expected}"


class TestFillColor:
    def test_black_on_white_bubble(self):
        style = extract_text_style(_render())
        assert style is not None
        _close(style["fill_rgb"], (0, 0, 0))
        assert style["confidence"] > 0.5

    def test_white_on_black_bubble(self):
        style = extract_text_style(
            _render(background=(0, 0, 0), fill=(255, 255, 255))
        )
        assert style is not None
        _close(style["fill_rgb"], (255, 255, 255))

    def test_saturated_fill_is_preserved_not_snapped_to_mono(self):
        """The whole point of the feature: a red SFX must not come back black."""
        style = extract_text_style(_render(fill=(200, 30, 30)))
        assert style is not None
        r, g, b = style["fill_rgb"]
        assert r > 140 and g < 90 and b < 90


class TestOutline:
    def test_contrasting_keyline_is_detected(self):
        style = extract_text_style(
            _render(fill=(200, 30, 30), stroke_width=4, stroke_fill=(250, 230, 40))
        )
        assert style is not None
        _close(style["fill_rgb"], (200, 30, 30), tolerance=60)
        _close(style["outline_rgb"], (250, 230, 40), tolerance=60)
        assert style["outline_width"] > 0.0

    def test_thicker_keyline_measures_wider(self):
        # Mid-grey ground: white fill and black keyline both stand off the page,
        # which is the situation the measurement is defined for.
        thin = extract_text_style(
            _render(background=(128, 128, 128), fill=(255, 255, 255),
                    stroke_width=2, stroke_fill=(0, 0, 0))
        )
        thick = extract_text_style(
            _render(background=(128, 128, 128), fill=(255, 255, 255),
                    stroke_width=6, stroke_fill=(0, 0, 0))
        )
        assert thin is not None and thick is not None
        assert thick["outline_width"] > thin["outline_width"]

    def test_no_keyline_reports_none(self):
        style = extract_text_style(_render())
        assert style is not None
        assert style["outline_rgb"] is None
        assert style["outline_width"] == 0.0


class TestGlow:
    def test_soft_halo_is_detected(self):
        style = extract_text_style(
            _render(
                background=(20, 20, 30),
                fill=(255, 255, 255),
                glow=(40, 120, 255),
                glow_radius=5,
            )
        )
        assert style is not None
        assert style["glow_rgb"] is not None, "expected a glow to be reported"
        assert style["glow_radius"] >= 2.0

    def test_plain_text_reports_no_glow(self):
        style = extract_text_style(_render())
        assert style is not None
        assert style["glow_rgb"] is None
        assert style["glow_radius"] == 0.0

    def test_glow_does_not_contaminate_the_fill(self):
        """A halo lifts the whole glow above the ink threshold, so the eroded
        core can land in the glow instead of the lettering. White text inside a
        blue halo must still read back white."""
        style = extract_text_style(
            _render(
                background=(18, 18, 28),
                fill=(255, 255, 255),
                glow=(40, 130, 255),
                glow_radius=6,
            )
        )
        assert style is not None
        _close(style["fill_rgb"], (255, 255, 255), tolerance=45)


class TestFontSize:
    @pytest.mark.parametrize("drawn_size", [24, 40, 64])
    def test_estimate_tracks_the_drawn_size(self, drawn_size):
        style = extract_text_style(
            _render(size=(480, 200), font_size=drawn_size)
        )
        assert style is not None
        estimated = style["font_size_px"]
        assert estimated is not None
        # Cap-height-based estimation is approximate; ±30% keeps the ceiling in
        # the right neighbourhood, which is all the layout engine needs.
        assert 0.7 * drawn_size <= estimated <= 1.3 * drawn_size

    def test_estimate_is_monotonic_in_drawn_size(self):
        small = extract_text_style(_render(size=(480, 200), font_size=20))
        large = extract_text_style(_render(size=(480, 200), font_size=60))
        assert small is not None and large is not None
        assert large["font_size_px"] > small["font_size_px"]

    def test_multiline_text_counts_lines(self):
        style = extract_text_style(
            _render(size=(400, 260), font_size=36, text="ONE\nTWO\nTHREE")
        )
        assert style is not None
        assert style["line_count"] == 3


class TestRejection:
    def test_blank_region_is_rejected(self):
        blank = np.full((120, 320, 3), 255, dtype=np.uint8)
        assert extract_text_style(blank) is None

    def test_low_contrast_text_is_rejected(self):
        # Off-white on white: nothing a letterer intended to be read as styled.
        faint = _render(background=(255, 255, 255), fill=(252, 252, 252))
        assert extract_text_style(faint) is None

    def test_tiny_region_is_rejected(self):
        assert extract_text_style(np.zeros((4, 4, 3), dtype=np.uint8)) is None

    def test_malformed_input_is_rejected(self):
        assert extract_text_style(None) is None
        assert extract_text_style(np.zeros((10, 10), dtype=np.uint8)) is None

    def test_bubble_outline_alone_is_not_read_as_text(self):
        """A drawn bubble border spans the crop and must not pass as lettering."""
        canvas = Image.new("RGB", (200, 120), (255, 255, 255))
        ImageDraw.Draw(canvas).rectangle([2, 2, 197, 117], outline=(0, 0, 0), width=3)
        assert extract_text_style(_to_bgr(canvas)) is None


class TestResolveOverrides:
    """The bridge from a measurement to the values the renderer is handed."""

    def _style(self, **overrides):
        style = extract_text_style(
            _render(fill=(200, 30, 30), stroke_width=4, stroke_fill=(250, 230, 40))
        )
        assert style is not None
        style.update(overrides)
        return style

    def test_measured_style_becomes_render_overrides(self):
        ov = resolve_style_overrides(self._style(), base_min_font=8, base_max_font=16)
        assert ov
        assert ov.text_color_rgb is not None
        assert ov.outline_color_rgb is not None
        assert ov.outline_width > 0

    def test_font_size_is_a_ceiling_not_a_pin(self):
        """Translations run longer, so the layout engine must keep room to shrink."""
        ov = resolve_style_overrides(
            self._style(font_size_px=40.0), base_min_font=8, base_max_font=16, tolerance=0.25
        )
        assert ov.max_font_size == 50
        assert ov.max_font_size >= 8

    def test_measured_size_is_clamped_to_the_hard_ceiling(self):
        ov = resolve_style_overrides(
            self._style(font_size_px=5000.0),
            base_min_font=8,
            base_max_font=16,
            hard_max_font=384,
        )
        assert ov.max_font_size == 384

    def test_low_confidence_read_is_discarded(self):
        ov = resolve_style_overrides(
            self._style(confidence=0.10), base_min_font=8, base_max_font=16, min_confidence=0.35
        )
        assert not ov
        assert ov.text_color_rgb is None
        assert ov.max_font_size is None

    def test_missing_style_yields_no_overrides(self):
        ov = resolve_style_overrides(None, base_min_font=8, base_max_font=16)
        assert not ov
        assert ov.outline_width is None
        assert ov.summary()

    def test_absent_attributes_are_left_to_the_defaults(self):
        """Nothing measured means nothing overridden - the tool's own look wins."""
        plain = extract_text_style(_render())
        assert plain is not None
        ov = resolve_style_overrides(plain, base_min_font=8, base_max_font=16)
        assert ov.text_color_rgb is not None
        assert ov.outline_color_rgb is None
        assert ov.glow_color_rgb is None
