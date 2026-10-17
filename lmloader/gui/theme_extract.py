"""Turn any picture into a LM Loader palette — 取色 → 语义映射 → 对比度回校.

The GUI can already be repainted from a token map alone (see
``styles.apply_theme(app, base, overrides)``), so this module's whole job is to
produce that map from an image:

1. median-cut sample the picture into ~12 clusters with their coverage;
2. pick a canvas and an accent from those clusters, and decide whether the
   picture wants the dark or the light base (mean luminance);
3. build every one of the 47 semantic tokens out of those two anchors — the
   neutral ramp by lightness steps, the status hues kept recognisable but
   pulled toward the picture's saturation;
4. re-balance contrast: sampled colours off a photograph rarely clear WCAG, so
   text and accent are pushed until they do, and the nudges are reported.

Nothing here touches Qt, so the result is a plain dict and fully testable.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

RGB = tuple[int, int, int]

# WCAG targets this module guarantees. Body text 4.5:1, large text / UI 3:1,
# primary text 7:1 so a photo-derived theme never ships unreadable chrome.
CONTRAST_TARGETS = {
    "title_on_bg": 7.0,
    "fg_on_bg": 4.5,
    "muted_on_bg": 4.5,
    "fg_on_surface": 4.5,
    "accent_on_bg": 3.0,
    "on_accent": 4.5,
    "sidebar_fg_on_sidebar": 4.5,
    "sidebar_active_fg_on_sidebar": 4.5,
    "log_fg_on_log_bg": 4.5,
}

# Status hues stay in their semantic family; only chroma follows the picture.
STATUS_HUES = {"green": 145.0, "red": 356.0, "warn": 33.0}


# ------------------------------------------------------------------ colour math

def to_rgb(colour: str) -> RGB:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def to_hex(rgb: RGB) -> str:
    """A plain #RRGGBB. Deliberately no alpha slot — see the ``wash`` note in
    ``extract_palette`` for why an 8-digit token would paint the wrong colour."""
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(v))) for v in rgb))


def luminance(rgb: RGB) -> float:
    def channel(v: int) -> float:
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: RGB, b: RGB) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def hsv(rgb: RGB):
    """HSV with the hue in **degrees**, the unit ``from_hsv`` takes."""
    h, s, v = colorsys.rgb_to_hsv(*(val / 255.0 for val in rgb))
    return h * 360.0, s, v


def from_hsv(h: float, s: float, v: float) -> RGB:
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h % 360.0 / 360.0, s, v))  # type: ignore[return-value]


def mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def set_lum(rgb: RGB, target: float) -> RGB:
    """Keep hue/saturation, move value until the WCAG luminance matches."""
    h, s, v = hsv(rgb)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if luminance(from_hsv(h, s, mid)) < target:
            lo = mid
        else:
            hi = mid
    return from_hsv(h, s, (lo + hi) / 2)


def ensure_contrast(fg: RGB, bg: RGB, target: float, toward: float) -> tuple[RGB, float]:
    """Make `fg` clear `target` against `bg`, losing as little colour as possible.

    Phase 1 rides HSV value, which keeps hue *and* chroma — an indigo accent that
    is too dark for a dark canvas should become a lighter indigo. Only when the
    value ramp bottoms out (a saturated colour's luminance is capped by its hue)
    does phase 2 blend toward white/black, which is monotonic in contrast and so
    always terminates.
    """
    ratio = contrast(fg, bg)
    if ratio >= target:
        return fg, ratio

    h, s, v = hsv(fg)
    direction = 1.0 if toward > 0.5 else -1.0
    for i in range(1, 51):
        nv = min(1.0, max(0.0, v + direction * i * 0.02))
        cand = from_hsv(h, s, nv)
        if contrast(cand, bg) >= target:
            return cand, contrast(cand, bg)
        if nv in (0.0, 1.0):
            break

    reference = (255, 255, 255) if direction > 0 else (0, 0, 0)
    best = mix(fg, reference, 1.0)
    if contrast(best, bg) < target:
        return best, contrast(best, bg)
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if contrast(mix(fg, reference, mid), bg) >= target:
            hi = mid
        else:
            lo = mid
    out = mix(fg, reference, hi)
    return out, contrast(out, bg)


# --------------------------------------------------------------------- sampling

@dataclass
class Sample:
    rgb: RGB
    share: float

    @property
    def lum(self) -> float:
        return luminance(self.rgb)

    @property
    def chroma(self) -> float:
        return max(self.rgb) / 255 - min(self.rgb) / 255


@dataclass
class Extraction:
    base: str
    tokens: dict[str, str] = field(default_factory=dict)
    contrast: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"base = {self.base}"]
        lines.append("取样（覆盖率 / 色值）: " + ", ".join(
            f"{to_hex(s.rgb)} {s.share * 100:.1f}%" for s in self.samples[:6]))
        lines.append("对比度:")
        for name, ratio in self.contrast.items():
            need = CONTRAST_TARGETS[name]
            lines.append(f"  {name:34s} {ratio:5.2f}:1  (需 ≥{need})")
        if self.notes:
            lines.append("回校:")
            lines.extend(f"  {n}" for n in self.notes)
        return "\n".join(lines)


def sample_image(path: str | Path, k: int = 12) -> list[Sample]:
    im = Image.open(path).convert("RGB")
    im.thumbnail((180, 180))
    quantized = im.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = quantized.getpalette() or []
    counted = sorted(quantized.getcolors() or [], reverse=True)
    total = sum(count for count, _ in counted) or 1
    out = []
    for count, index in counted:
        rgb = tuple(palette[index * 3: index * 3 + 3])  # type: ignore[assignment]
        out.append(Sample(rgb, count / total))
    return out


def _pick_accent(samples: list[Sample]) -> Sample:
    """Most colourful cluster that is neither the canvas nor a highlight speck."""
    candidates = [s for s in samples if 0.03 <= s.lum <= 0.92 and s.share >= 0.015]
    if not candidates:
        candidates = [s for s in samples if s.chroma > 0] or samples
    return max(candidates, key=lambda s: s.chroma * (s.share ** 0.4))


# ---------------------------------------------------------------- the mapping

def extract_palette(path: str | Path, base: str | None = None, k: int = 12) -> Extraction:
    samples = sample_image(path, k=k)
    mean_lum = sum(s.lum * s.share for s in samples)
    base = base or ("dark" if mean_lum < 0.35 else "light")
    dark_base = base == "dark"
    accent_sample = _pick_accent(samples)
    _ah, asat, _av = hsv(accent_sample.rgb)
    # The canvas inherits the picture's own darkest (or lightest) wash, so a navy
    # photograph gives navy chrome instead of the mud a low-saturation average
    # would. Saturation is capped because full-strength colour fights text.
    ordered = sorted(samples, key=lambda s: s.lum)
    canvas_h, canvas_s, _ = hsv((ordered[0] if dark_base else ordered[-1]).rgb)
    neutral_hue = canvas_h
    canvas_sat = min(0.55, max(0.10, canvas_s * 0.85))
    neutral_sat = min(0.30, max(0.05, canvas_sat * 0.55))
    dark = dark_base

    notes: list[str] = []
    canvas_seed = from_hsv(neutral_hue, canvas_sat, 0.10 if dark else 0.97)
    bg = set_lum(canvas_seed, 0.055 if dark else 0.90)
    white = (255, 255, 255)

    def step(delta: float) -> RGB:
        """A neutral `delta` of the way from the canvas toward the text end.

        On a dark base that end is white, on a light base it is black — getting
        this backwards produces a ramp where "body text" is darker than the
        canvas, and only the contrast rebalancer notices.
        """
        return mix(bg, from_hsv(neutral_hue, neutral_sat * 0.6, 1.0 if dark else 0.0), delta)

    surface = step(0.06)
    surface_hover = step(0.10)
    btn = step(0.13) if dark else white
    btn_hover = step(0.20)
    btn_press = step(0.02)
    border = step(0.22)
    control_border = step(0.17)
    grid = step(0.10)
    chip_bg = btn
    chip_border = control_border
    scroll_handle = step(0.24)
    scroll_handle_hover = step(0.34)
    dim = step(0.30)
    muted = step(0.52)
    fg = step(0.78)
    title = step(0.95)

    accent = accent_sample.rgb
    accent, _ = ensure_contrast(accent, bg, CONTRAST_TARGETS["accent_on_bg"], 1.0 if dark else 0.0)
    # Both shipped themes put white on the accent, which only works because their
    # accents are dark. A photo-derived accent can land mid-luminance, where white
    # text misses 4.5:1 — so pick whichever end the accent actually contrasts with.
    # contrast(white,X) * contrast(X,black) is exactly 21, so the winner is always
    # ≥ 4.58:1 before the safety net below runs.
    paper, ink = white, from_hsv(neutral_hue, neutral_sat, 0.06)
    on_accent = paper if contrast(paper, accent) >= contrast(ink, accent) else ink
    on_accent, _ = ensure_contrast(on_accent, accent, CONTRAST_TARGETS["on_accent"],
                                   1.0 if luminance(on_accent) > 0.5 else 0.0)
    on_accent_disabled = mix(accent, bg, 0.55)

    # Accent-derived fills. Never an 8-digit hex: Qt parses "#RRGGBBAA" as
    # "#AARRGGBB", so "#1f6feb22" paints a 12%-alpha *green*. Translucent fills
    # are pre-composited onto the canvas instead — one rule for both bases, so a
    # tint can never land outside the canvas↔source range it is supposed to sit in.
    def wash(colour: RGB, alpha: float) -> RGB:
        return mix(bg, colour, alpha)

    accent_soft = bubble_user = to_hex(wash(accent, 0x22 / 255))
    bubble_border = to_hex(wash(accent, 0x44 / 255))
    accent_border = to_hex(wash(accent, 0x55 / 255))
    # Hover has to stay a *visible* step from the resting fill: on a dark canvas a
    # deeper wash reads, on a near-white one the wash is already at its ceiling.
    accent_hover = to_hex(wash(accent, 0x33 / 255) if dark else mix(accent, white, 0.18))
    accent_surface = to_hex(wash(accent, 0.16 if dark else 0.08))

    # Status colours: family fixed, saturation borrowed from the picture — but
    # floored, or a muted photo turns "success" into olive and "danger" into brick.
    signal_sat = min(0.85, max(0.60, asat))
    signals: dict[str, str] = {}
    for name, hue in STATUS_HUES.items():
        rgb = from_hsv(hue, signal_sat, 0.62 if dark else 0.60)
        rgb, _ = ensure_contrast(rgb, bg, 3.0, 1.0 if dark else 0.0)
        signals[name] = to_hex(rgb)
    green, red, warn = (to_rgb(signals[c]) for c in ("green", "red", "warn"))
    green_bg, red_bg = to_hex(wash(green, 0x22 / 255)), to_hex(wash(red, 0x22 / 255))
    if dark:
        red_border_rgb, green_btn_rgb = mix(green, (0, 0, 0), 0.56), mix(green, (0, 0, 0), 0.25)
        red_btn_rgb = red
    else:
        red_border_rgb, green_btn_rgb = mix(red, white, 0.62), green
        red_btn_rgb = mix(red, (0, 0, 0), 0.05)
    red_border = to_hex(red_border_rgb)
    green_btn = to_hex(green_btn_rgb)
    green_btn_hover = to_hex(mix(green_btn_rgb, white, 0.18))
    red_btn = to_hex(red_btn_rgb)
    red_btn_hover = to_hex(mix(red_btn_rgb, white, 0.18))

    # The rail sidebar is dark in both shipped themes; tint it from the picture.
    sidebar_rgb = set_lum(from_hsv(neutral_hue, min(0.45, neutral_sat * 1.6), 0.10), 0.045)
    sidebar = to_hex(sidebar_rgb)
    sidebar_hover = to_hex(mix(sidebar_rgb, white, 0.10))
    sidebar_fg = to_hex(mix(sidebar_rgb, white, 0.52))
    sidebar_active = to_hex(accent)
    sidebar_active_fg = to_hex(ensure_contrast(
        accent, sidebar_rgb, CONTRAST_TARGETS["sidebar_active_fg_on_sidebar"], 1.0)[0])
    sidebar_fg_hover = to_hex(mix(sidebar_rgb, white, 0.80))

    log_bg = to_hex(set_lum(canvas_seed, 0.04))
    log_fg = to_hex(mix(to_rgb(log_bg), white, 0.72))

    tokens = {
        "bg": to_hex(bg), "surface": to_hex(surface), "surface_hover": to_hex(surface_hover),
        "btn": to_hex(btn), "btn_hover": to_hex(btn_hover), "border": to_hex(border),
        "dim": to_hex(dim), "muted": to_hex(muted), "fg": to_hex(fg), "title": to_hex(title),
        "accent": to_hex(accent), "accent_soft": accent_soft,
        "accent_surface": accent_surface, "accent_hover": accent_hover,
        "accent_border": accent_border,
        "green": signals["green"], "green_bg": green_bg, "green_btn": green_btn,
        "green_btn_hover": green_btn_hover,
        "red": signals["red"], "red_btn": red_btn, "red_btn_hover": red_btn_hover,
        "red_bg": red_bg, "red_border": red_border, "warn": signals["warn"],
        "mem_compute": to_hex(mix(accent, from_hsv(280.0, signal_sat, 0.62), 0.5)),
        "mem_other": to_hex(dim), "mem_system": to_hex(step(0.16)),
        "log_bg": log_bg, "log_fg": log_fg,
        "sidebar": sidebar, "sidebar_hover": sidebar_hover, "sidebar_fg": sidebar_fg,
        "sidebar_active": sidebar_active,
        "sidebar_active_fg": sidebar_active_fg,
        "sidebar_fg_hover": sidebar_fg_hover,
        "chip_bg": to_hex(chip_bg), "chip_border": to_hex(chip_border),
        "btn_press": to_hex(btn_press), "control_border": to_hex(control_border),
        "scroll_handle": to_hex(scroll_handle), "scroll_handle_hover": to_hex(scroll_handle_hover),
        "bubble_user": bubble_user, "bubble_border": bubble_border,
        "grid": to_hex(grid),
        "on_accent": to_hex(on_accent), "on_accent_disabled": to_hex(on_accent_disabled),
    }

    # Re-balance every text/background pair the interface actually paints.
    for token, background, key in (
        ("fg", "bg", "fg_on_bg"),
        ("fg", "surface", "fg_on_surface"),      # cards are lighter than the canvas
        ("title", "bg", "title_on_bg"),
        ("muted", "bg", "muted_on_bg"),
        ("sidebar_fg", "sidebar", "sidebar_fg_on_sidebar"),
        ("sidebar_active_fg", "sidebar", "sidebar_active_fg_on_sidebar"),
        ("log_fg", "log_bg", "log_fg_on_log_bg"),
    ):
        target = CONTRAST_TARGETS[key]
        back = to_rgb(tokens[background])
        # Decide by the background, not by the base theme: the rail and the log
        # panel stay dark even in the light theme.
        rgb, _ratio = ensure_contrast(to_rgb(tokens[token]), back, target,
                                      1.0 if luminance(back) < 0.5 else 0.0)
        if to_hex(rgb) != tokens[token]:
            notes.append(f"{token}: {tokens[token]} → {to_hex(rgb)}"
                         f"（{key} 需 ≥{target}）")
            tokens[token] = to_hex(rgb)

    contrast_table = {
        "title_on_bg": contrast(to_rgb(tokens["title"]), bg),
        "fg_on_bg": contrast(to_rgb(tokens["fg"]), bg),
        "muted_on_bg": contrast(to_rgb(tokens["muted"]), bg),
        "fg_on_surface": contrast(to_rgb(tokens["fg"]), to_rgb(tokens["surface"])),
        "accent_on_bg": contrast(to_rgb(tokens["accent"]), bg),
        "on_accent": contrast(to_rgb(tokens["on_accent"]), to_rgb(tokens["accent"])),
        "sidebar_fg_on_sidebar": contrast(to_rgb(tokens["sidebar_fg"]), to_rgb(tokens["sidebar"])),
        "sidebar_active_fg_on_sidebar": contrast(
            to_rgb(tokens["sidebar_active_fg"]), to_rgb(tokens["sidebar"])),
        "log_fg_on_log_bg": contrast(to_rgb(tokens["log_fg"]), to_rgb(tokens["log_bg"])),
    }
    return Extraction(base=base, tokens=tokens, contrast=contrast_table, notes=notes, samples=samples)
