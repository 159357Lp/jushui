"""一张图 → 47 个调色板令牌：取色流程的契约。

``theme_extract.extract_palette()`` is the whole bridge between "user uploaded a
photo" and "the interface repaints" — ``styles.apply_theme(app, base, tokens)``
already knows how to paint from a token map. So this suite pins the four things
that bridge must never break:

* coverage — every token either theme has must be produced, and no token is
  invented that the QSS template doesn't know;
* legibility — every WCAG pair the interface paints clears its target, whatever
  the picture looks like (a flat white or flat black image included);
* determinism — the same file twice gives byte-identical tokens;
* structure — the neutral ramp stays a ramp and the status colours stay green,
  red and amber.

``test_contrast_judge_reports_a_planted_failure`` is the negative control: it
proves the contrast table can actually go red, so every "全部达标" above it means
something.
"""

import re

import pytest
from PIL import Image

from lmloader.gui import styles
from lmloader.gui.theme_extract import (
    CONTRAST_TARGETS, STATUS_HUES, contrast, extract_palette, hsv, luminance,
    sample_image, to_rgb,
)

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

# The images the pipeline must never fall over on: a dark photo with a warm
# accent, a light photo with a cool one, and the two degenerate cases a flat
# scan or a blank upload produces.
PHOTOS = {
    "navy_orange": [((10, 18, 52), 0.55), ((16, 28, 74), 0.25), ((232, 122, 32), 0.12),
                    ((208, 214, 232), 0.08)],
    "paper_teal": [((247, 246, 241), 0.6), ((222, 228, 226), 0.2), ((18, 128, 122), 0.15),
                   ((42, 48, 52), 0.05)],
    "flat_white": [((255, 255, 255), 1.0)],
    "flat_black": [((0, 0, 0), 1.0)],
    "grey_only": [((128, 128, 128), 1.0)],
}


def _photo(tmp_path, name: str) -> str:
    """Stripes in the declared proportions — median-cut gets exactly these clusters."""
    bands = PHOTOS[name]
    height = 200
    im = Image.new("RGB", (1, height))
    top = 0
    for rgb, share in bands:
        n = max(1, round(height * share))
        for y in range(top, min(height, top + n)):
            im.putpixel((0, y), rgb)
        top += n
    im = im.resize((200, height))
    path = tmp_path / f"{name}.png"
    im.save(path)
    return str(path)


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """Extract every photo once for the module — the pipeline is deterministic,
    and re-running median-cut per test would only slow the suite down."""
    tmp = tmp_path_factory.mktemp("theme_extract")
    return {name: extract_palette(_photo(tmp, name)) for name in PHOTOS}


# --------------------------------------------------------------------- coverage

@pytest.mark.parametrize("name", sorted(PHOTOS))
@pytest.mark.parametrize("base", ["dark", "light"])
def test_covers_every_token_both_bases(name, base, tmp_path):
    tokens = extract_palette(_photo(tmp_path, name), base=base).tokens
    shipped = set(styles.theme_tokens(base))
    assert not shipped - set(tokens), f"{name}/{base} 少了令牌: {sorted(shipped - set(tokens))}"
    assert not set(tokens) - shipped, f"{name}/{base} 凭空造了令牌: {sorted(set(tokens) - shipped)}"
    assert len(tokens) == len(shipped)


@pytest.mark.parametrize("name", sorted(PHOTOS))
@pytest.mark.parametrize("base", ["dark", "light"])
def test_extracted_tokens_render_the_qss_template(name, base, extracted):
    """The real integration: the extracted map must substitute into the shipped
    stylesheet without a missing key or a leftover placeholder."""
    qss = styles.render_theme(base, extracted[name].tokens)
    assert "${" not in qss
    assert extracted[name].tokens["accent"].upper() in qss


@pytest.mark.parametrize("name", sorted(PHOTOS))
def test_every_value_is_a_plain_hex_colour(name, extracted):
    """No 8-digit token may survive: Qt's QColor reads "#RRGGBBAA" as
    "#AARRGGBB", so an alpha-suffixed accent turns into a different hue."""
    for token, value in extracted[name].tokens.items():
        assert HEX.match(value), f"{token} = {value!r} 不是 #RRGGBB"


@pytest.mark.parametrize("name", sorted(PHOTOS))
def test_tinted_fills_are_washes_not_raw_accents(name, extracted):
    """The translucent fills are pre-composited onto the canvas, so each must sit
    *between* the canvas and its source colour — a token equal to the accent would
    paint a solid block where a tint belongs."""
    ex = extracted[name]
    bg = to_rgb(ex.tokens["bg"])
    for token, source in (("accent_soft", "accent"), ("accent_border", "accent"),
                          ("bubble_user", "accent"), ("bubble_border", "accent"),
                          ("accent_surface", "accent"), ("green_bg", "green"), ("red_bg", "red")):
        rgb, tint = to_rgb(ex.tokens[token]), to_rgb(ex.tokens[source])
        assert rgb != tint, f"{token} 就是原色，没有洗淡"
        for ch in range(3):
            lo, hi = sorted((bg[ch], tint[ch]))
            assert lo - 2 <= rgb[ch] <= hi + 2, (
                f"{token} 通道 {ch}={rgb[ch]} 不在 bg↔{source} 之间")
        assert abs(luminance(rgb) - luminance(bg)) < abs(luminance(tint) - luminance(bg)), \
            f"{token} 比 {source} 还浓"


# ------------------------------------------------------------------ legibility

@pytest.mark.parametrize("name", sorted(PHOTOS))
def test_contrast_table_meets_every_target(name, tmp_path):
    for base in ("dark", "light"):
        ex = extract_palette(_photo(tmp_path, name), base=base)
        assert set(ex.contrast) == set(CONTRAST_TARGETS)
        bad = {k: f"{v:.2f}:1 < {CONTRAST_TARGETS[k]}:1"
               for k, v in ex.contrast.items() if v + 1e-6 < CONTRAST_TARGETS[k]}
        assert not bad, f"{name}/{base} 对比度不达标: {bad}"


def test_midtone_accent_does_not_force_white_text(tmp_path):
    """Regression: both shipped themes put white on the accent, but a grey accent
    leaves white text at 3.7:1. The extractor must flip to dark ink instead."""
    ex = extract_palette(_photo(tmp_path, "grey_only"), base="light")
    on_accent = luminance(to_rgb(ex.tokens["on_accent"]))
    assert ex.contrast["on_accent"] + 1e-6 >= CONTRAST_TARGETS["on_accent"]
    assert on_accent < 0.5, "灰底accent还配白字，说明没按亮度选极端"


def test_judge_is_not_satisfied_by_a_single_hue(tmp_path):
    """A one-colour picture gives the extractor no neutral ramp to inherit; it must
    still end up with distinct background and text."""
    ex = extract_palette(_photo(tmp_path, "flat_black"), base="dark")
    assert ex.tokens["bg"] != ex.tokens["fg"]
    assert ex.contrast["fg_on_bg"] >= 4.5


# ------------------------------------------------------------------- structure

def test_neutral_ramp_is_ordered(extracted):
    light, dark = extracted["paper_teal"], extracted["navy_orange"]
    for ex, name in ((light, "light"), (dark, "dark")):
        order = [ex.tokens[k] for k in ("bg", "surface", "muted", "fg", "title")]
        lum = [luminance(to_rgb(v)) for v in order]
        if name == "light":
            assert lum == sorted(lum, reverse=True), f"浅色主题的灰阶反了: {list(zip(order, lum))}"
        else:
            assert lum == sorted(lum), f"深色主题的灰阶反了: {list(zip(order, lum))}"
        # "distinct" matters as much as "ordered" — a collapsed ramp is unreadable
        # even while sorted.
        assert len(set(order)) == len(order), f"{name} 灰阶塌成同色"
        assert abs(lum[0] - lum[-1]) > 0.5, f"{name} 首尾太接近"


def test_status_colours_stay_in_their_family(extracted):
    for name, ex in extracted.items():
        for colour, hue in STATUS_HUES.items():
            actual = hsv(to_rgb(ex.tokens[colour]))[0]
            delta = min(abs(actual - hue), 360 - abs(actual - hue))
            assert delta <= 40, f"{name}: {colour} 色相 {actual:.0f}° 偏离 {hue:.0f}°"


def test_status_colours_are_distinguishable(extracted):
    for name, ex in extracted.items():
        rgb = {k: to_rgb(ex.tokens[k]) for k in STATUS_HUES}
        hues = {k: hsv(v)[0] for k, v in rgb.items()}
        assert len({round(h / 30) for h in hues.values()}) == 3, f"{name}: 状态色糊成一片"


# ---------------------------------------------------------------- base choice

def test_base_follows_the_picture(tmp_path):
    assert extract_palette(_photo(tmp_path, "navy_orange")).base == "dark"
    assert extract_palette(_photo(tmp_path, "paper_teal")).base == "light"


@pytest.mark.parametrize("base", ["dark", "light"])
def test_explicit_base_overrides_the_picture(base, tmp_path):
    ex = extract_palette(_photo(tmp_path, "navy_orange"), base=base)
    assert ex.base == base
    assert set(ex.tokens) >= set(styles.theme_tokens(base))


# ----------------------------------------------------------------- determinism

def test_same_image_twice_is_identical(tmp_path):
    path = _photo(tmp_path, "navy_orange")
    a, b = extract_palette(path), extract_palette(path)
    assert a.tokens == b.tokens
    assert a.contrast == b.contrast
    assert a.notes == b.notes


def test_sampling_shares_sum_to_one(tmp_path):
    for name in PHOTOS:
        shares = sum(s.share for s in sample_image(_photo(tmp_path, name), k=12))
        assert 0.99 <= shares <= 1.01, f"{name} 覆盖率合计 {shares}"


# ------------------------------------------------------- negative control (must go red)

def test_contrast_judge_reports_a_planted_failure(extracted):
    """Every "全部达标" above is worthless unless the same arithmetic can report a
    broken palette — so plant one and check it is refused."""
    ex = extracted["navy_orange"]
    assert not [k for k, v in ex.contrast.items() if v < CONTRAST_TARGETS[k]]

    planted = dict(ex.tokens)
    for token, back, key in (("fg", "bg", "fg_on_bg"),
                             ("on_accent", "accent", "on_accent"),
                             ("sidebar_fg", "sidebar", "sidebar_fg_on_sidebar")):
        broken = dict(planted)
        broken[token] = broken[back]
        got = contrast(to_rgb(broken[token]), to_rgb(broken[back]))
        assert got < CONTRAST_TARGETS[key], (
            f"把 {token} 涂成 {back} 后判据仍然通过 —— 探针或阈值失效")
