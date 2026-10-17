"""Theme stylesheets for LM Loader GUI — dark + light, switchable at runtime.

Both stylesheets are ``string.Template`` bodies filled from a palette, so
`render_theme` / `apply_theme` can recolour the whole window from a token map
alone — that is how an image-derived theme gets injected. Add a colour to a
palette and reference it as ``${token}``; never paste a hex into the QSS, the
test suite fails on a hard-coded colour.

Pages that embed colors inside inline QSS should pull them from `c(name)`
instead of hard-coding, so a theme switch recolors them without rebuilding.

Design tokens follow the "LM Loader UI 重设计" spec:
  * light theme = light canvas (#F5F6FA) + white cards + blue accent (#2E6BE6)
    + DARK rail sidebar (#151A2C), 12px card radius, 10px control radius.
"""

from string import Template

# ---------------------------------------------------------------- palettes

_PALETTES = {
    "dark": {
        "bg": "#0d1117",            # window / input background
        "surface": "#161b22",       # cards, groups, sidebar
        "surface_hover": "#1c2129",
        "btn": "#21262d",
        "btn_hover": "#30363d",
        "border": "#30363d",
        "dim": "#484f58",           # disabled / placeholder
        "muted": "#8b949e",         # secondary text
        "fg": "#c9d1d9",
        "title": "#f0f6fc",
        "accent": "#58a6ff",
        "accent_soft": "#1f6feb22",      # translucent accent bg
        "accent_surface": "#17283b",
        "accent_hover": "#1f6feb33",
        "accent_border": "#1f6feb55",
        "green": "#3fb950",
        "green_bg": "#3fb95022",
        "green_btn": "#238636",
        "green_btn_hover": "#2ea043",
        "red": "#f85149",
        "red_btn": "#da3633",
        "red_btn_hover": "#f85149",
        "red_bg": "#da363322",
        "red_border": "#da363344",
        "warn": "#d29922",
        # memory-breakdown segments (monitor page): weights/kv/state reuse
        # accent/green/warn, these two have no existing semantic token.
        "mem_compute": "#a371f7",
        "mem_other": "#484f58",
        "mem_system": "#6E7681",
        "log_bg": "#0d1117",        # terminal log panel
        "log_fg": "#c9d1d9",
        # sidebar (dark rail) — dark theme keeps the same near-black rail
        "sidebar": "#0d1117",
        "sidebar_hover": "#21262d",
        "sidebar_fg": "#8b949e",
        "sidebar_active": "#1f6feb",
        "chip_bg": "#21262d",
        "chip_border": "#30363d",
        "btn_press": "#0d1117",           # pressed control fill
        "control_border": "#30363d",  # form-control outline
        "scroll_handle": "#30363d",
        "scroll_handle_hover": "#484f58",
        "bubble_user": "#1f6feb22",         # user message fill
        "bubble_border": "#1f6feb44",
        "sidebar_active_fg": "#58a6ff",
        "sidebar_fg_hover": "#c9d1d9",
        "grid": "#21262d",                       # gridlines / groove fill
        "on_accent": "white",             # text drawn on a filled accent
        "on_accent_disabled": "#484f58",
    },
    "light": {
        "bg": "#F5F6FA",            # app canvas
        "surface": "#FFFFFF",       # cards / panels
        "surface_hover": "#F8F9FC",
        "btn": "#FFFFFF",
        "btn_hover": "#F1F3F9",
        "border": "#E9ECF4",        # card + input border
        "dim": "#B3BAC8",           # placeholder / disabled
        "muted": "#5A6478",         # secondary text
        "fg": "#3A4154",            # body text
        "title": "#1A2032",         # headings
        "accent": "#2E6BE6",        # primary blue
        "accent_soft": "#EAF2FF",
        "accent_surface": "#F1F5FF",
        "accent_hover": "#4A82F0",
        "accent_border": "#C7DAFF",
        "green": "#1F9763",
        "green_bg": "#EAF7F0",
        "green_btn": "#22A06B",
        "green_btn_hover": "#1F9763",
        "red": "#E5484D",
        "red_btn": "#E5484D",
        "red_btn_hover": "#F0666A",
        "red_bg": "#FDECEC",
        "red_border": "#F7CDCF",
        "warn": "#E5853D",
        "mem_compute": "#7C3AED",
        "mem_other": "#B3BAC8",
        "mem_system": "#E4E8F1",
        "log_bg": "#0F1420",        # console-style log panel
        "log_fg": "#A9B1C6",
        # dark rail sidebar (design spec: sidebar stays dark in light theme)
        "sidebar": "#151A2C",
        "sidebar_hover": "#232A42",
        "sidebar_fg": "#8A93A8",
        "sidebar_active": "#2E6BE6",
        "chip_bg": "#F1F3F9",
        "chip_border": "#E4E8F1",
        "btn_press": "#E4E8F1",           # pressed control fill
        "control_border": "#E4E8F1",  # form-control outline
        "scroll_handle": "#DDE3EE",
        "scroll_handle_hover": "#C6CBD8",
        "bubble_user": "#E9F0FF",         # user message fill
        "bubble_border": "#D6E4FF",
        "sidebar_active_fg": "#FFFFFF",
        "sidebar_fg_hover": "#FFFFFF",
        "grid": "#F1F3F9",                       # gridlines / groove fill
        "on_accent": "#FFFFFF",             # text drawn on a filled accent
        "on_accent_disabled": "#FFFFFF",
    },
}

current_theme = "dark"


def c(name: str) -> str:
    """Theme-aware color lookup for inline QSS in pages."""
    return _PALETTES[current_theme].get(name, _PALETTES["dark"].get(name, "#000000"))


_DARK_QSS = r"""
QMainWindow {
    background-color: ${bg};
}

QWidget {
    color: ${fg};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

QSidebar, #sidebar {
    background-color: ${surface};
    border: none;
}

#sidebar QPushButton {
    background-color: transparent;
    color: ${sidebar_fg};
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-size: 14px;
}

#sidebar QPushButton:hover {
    background-color: ${btn};
    color: ${sidebar_fg_hover};
}

#sidebar QPushButton[active="true"] {
    background-color: ${accent_soft};
    color: ${sidebar_active_fg};
    border-left: 3px solid ${accent};
}

/* Collapsed rail: centered glyphs, no side padding */
#sidebar QPushButton[rail="true"] {
    padding: 10px 0;
    text-align: center;
    font-size: 15px;
}

#sidebar QPushButton#sidebarToggle {
    background-color: transparent;
    color: ${sidebar_fg};
    border: none;
    border-radius: 6px;
    font-size: 17px;
    text-align: left;
    padding: 4px 14px;
}

#sidebar QPushButton#sidebarToggle:hover {
    background-color: ${btn};
    color: ${sidebar_fg_hover};
}

#sidebar QPushButton#sidebarToggle[rail="true"] {
    padding: 4px 0;
    text-align: center;
}

QGroupBox {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: ${accent};
}

#modelCard {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 8px;
}

#modelCard:hover {
    background-color: ${surface_hover};
    border-color: ${muted};
}

QFrame#card {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 8px;
}

QPushButton {
    background-color: ${btn};
    color: ${fg};
    border: 1px solid ${control_border};
    border-radius: 6px;
    padding: 0 12px;
    min-height: 30px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: ${btn_hover};
    border-color: ${muted};
}

QPushButton:pressed {
    background-color: ${btn_press};
}

QPushButton:disabled {
    color: ${dim};
    border-color: ${btn};
}

QPushButton#primary {
    background-color: ${green_btn};
    border-color: ${green_btn_hover};
    color: ${on_accent};
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: ${green_btn_hover};
}

QPushButton#primary:disabled {
    background-color: ${btn};
    color: ${on_accent_disabled};
}

QPushButton#danger {
    background-color: ${red_btn};
    border-color: ${red};
    color: ${on_accent};
}

QPushButton#danger:hover {
    background-color: ${red_btn_hover};
}

/* Horizontal padding only. Fusion + QSS *vertical* padding mis-computes the
   text rect after a window resize and long text (e.g. a binary path) gets
   its glyphs clipped/overlapped; min-height keeps the height instead. */
QLineEdit {
    background-color: ${bg};
    border: 1px solid ${control_border};
    border-radius: 6px;
    padding: 0 8px;
    min-height: 30px;
    color: ${fg};
    selection-background-color: ${sidebar_active};
}

/* QSpinBox/QComboBox are complex widgets (border + editor + buttons). A QSS
   padding on the whole widget makes the Fusion style mis-compute the inner
   editor rect; after a window resize the value text is then painted into a
   wrong/clipped rect and its glyphs come out corrupted (999 -> "uuu").
   So: keep background/border, drop the padding. */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: ${bg};
    border: 1px solid ${control_border};
    border-radius: 6px;
    min-height: 30px;
    color: ${fg};
    selection-background-color: ${sidebar_active};
}

/* On Windows the native style can override the text color of the line
   edit embedded inside QSpinBox/QComboBox (values become black on the dark
   background and invisible). With the Fusion style the top-level color rule
   is honoured, and overriding the inner editor's background/border is
   harmful: Qt's documented widget-rendering issue with sub-control
   backgrounds makes the inner text vanish after a window resize. So we only
   pin the color here. */
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    color: ${fg};
    selection-background-color: ${sidebar_active};
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: ${btn};
    border: 1px solid ${control_border};
    border-top-right-radius: 5px;
    width: 18px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: ${btn};
    border: 1px solid ${control_border};
    border-bottom-right-radius: 5px;
    width: 18px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: ${btn_hover};
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: ${accent};
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: ${surface};
    border: 1px solid ${control_border};
    selection-background-color: ${sidebar_active};
    color: ${fg};
}

QTableWidget {
    background-color: ${bg};
    alternate-background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 8px;
    gridline-color: ${btn};
    selection-background-color: ${accent_hover};
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: ${accent_hover};
}

QHeaderView::section {
    background-color: ${surface};
    color: ${muted};
    border: none;
    border-bottom: 1px solid ${border};
    padding: 8px 6px;
    font-weight: bold;
}

QTextEdit, QTextEdit[readonly="true"] {
    background-color: ${bg};
    border: 1px solid ${control_border};
    border-radius: 8px;
    color: ${fg};
    selection-background-color: ${sidebar_active};
}

QScrollArea {
    background-color: ${bg};
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: ${bg};
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: ${scroll_handle};
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: ${scroll_handle_hover};
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: ${bg};
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background-color: ${scroll_handle};
    border-radius: 4px;
    min-width: 30px;
}

QTabWidget::pane {
    border: 1px solid ${control_border};
    border-radius: 8px;
    background-color: ${bg};
}

QTabBar::tab {
    background-color: ${surface};
    color: ${muted};
    border: none;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: ${accent};
    border-bottom-color: ${accent};
}

QTabBar::tab:hover {
    color: ${fg};
    background-color: ${btn};
}

QProgressBar {
    background-color: ${btn};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: ${fg};
    height: 8px;
}

QProgressBar::chunk {
    background-color: ${green_btn};
    border-radius: 4px;
}

QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: ${title};
}

QLabel#subtitle {
    font-size: 12px;
    color: ${muted};
}

QLabel#status-loaded {
    color: ${green};
    font-weight: bold;
}

QLabel#status-unloaded {
    color: ${muted};
}

QLabel#status-loading {
    color: ${warn};
}

QLabel#status-failed {
    color: ${red};
}

QSplitter::handle {
    background-color: ${btn};
}

QSplitter::handle:horizontal {
    width: 1px;
}

#chat_input {
    background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 8px;
    padding: 10px;
    color: ${fg};
}

#chat_input:focus {
    border-color: ${accent};
}

#message_user {
    background-color: ${bubble_user};
    border: 1px solid ${bubble_border};
    border-radius: 8px;
    padding: 10px;
}

#message_assistant {
    background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 8px;
    padding: 10px;
}
"""

# ------------------------------------------------------------- light theme
# Design spec: light canvas, white cards (12px radius), blue accent #2E6BE6,
# dark rail sidebar (#151A2C) with a blue active pill.

_LIGHT_QSS = r"""
QMainWindow {
    background-color: ${bg};
}

QWidget {
    color: ${fg};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ---- dark rail sidebar (stays dark on purpose) ---- */
QSidebar, #sidebar {
    background-color: ${sidebar};
    border: none;
}

#sidebar QLabel {
    color: ${sidebar_fg};
    background: transparent;
}

#sidebar QPushButton {
    background-color: transparent;
    color: ${sidebar_fg};
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: left;
    font-size: 14px;
}

#sidebar QPushButton:hover {
    background-color: ${sidebar_hover};
    color: ${sidebar_fg_hover};
}

#sidebar QPushButton[active="true"] {
    background-color: ${sidebar_active};
    color: ${sidebar_active_fg};
    border: none;
}

/* Collapsed rail: centered glyphs, no side padding */
#sidebar QPushButton[rail="true"] {
    padding: 10px 0;
    text-align: center;
    font-size: 15px;
}

#sidebar QPushButton#sidebarToggle {
    background-color: transparent;
    color: ${sidebar_fg};
    border: none;
    border-radius: 10px;
    font-size: 17px;
    text-align: left;
    padding: 4px 14px;
}

#sidebar QPushButton#sidebarToggle:hover {
    background-color: ${sidebar_hover};
    color: ${sidebar_fg_hover};
}

#sidebar QPushButton#sidebarToggle[rail="true"] {
    padding: 4px 0;
    text-align: center;
}

/* ---- cards ---- */
QGroupBox {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 12px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: ${title};
}

#modelCard {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 12px;
}

#modelCard:hover {
    background-color: ${surface};
    border-color: ${accent_border};
}

/* Generic white card used by pages that group controls outside a QGroupBox
   (e.g. the download page's manual-fetch panel). */
QFrame#card {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 12px;
}

/* ---- buttons ---- */
QPushButton {
    background-color: ${btn};
    color: ${fg};
    border: 1px solid ${control_border};
    border-radius: 10px;
    padding: 0 14px;
    min-height: 32px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: ${btn_hover};
    border-color: ${accent_border};
}

QPushButton:pressed {
    background-color: ${btn_press};
}

QPushButton:disabled {
    color: ${dim};
    border-color: ${border};
    background-color: ${surface_hover};
}

QPushButton#primary {
    background-color: ${accent};
    border-color: ${accent};
    color: ${on_accent};
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: ${accent_hover};
    border-color: ${accent_hover};
}

QPushButton#primary:disabled {
    background-color: ${accent_border};
    border-color: ${accent_border};
    color: ${on_accent_disabled};
}

QPushButton#danger {
    background-color: ${red_btn};
    border-color: ${red_btn};
    color: ${on_accent};
}

QPushButton#danger:hover {
    background-color: ${red_btn_hover};
    border-color: ${red_btn_hover};
}

/* ---- inputs ----
   Horizontal padding only. Fusion + QSS *vertical* padding mis-computes the
   text rect after a window resize and long text (e.g. a binary path) gets
   its glyphs clipped/overlapped; min-height keeps the height instead. */
QLineEdit {
    background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 10px;
    padding: 0 12px;
    min-height: 34px;
    color: ${fg};
    selection-background-color: ${accent_border};
}

/* Complex widgets: keep background/border, drop padding (see DARK_THEME). */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 10px;
    min-height: 34px;
    color: ${fg};
    selection-background-color: ${accent_border};
}

QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    color: ${fg};
    selection-background-color: ${accent_border};
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: ${surface_hover};
    border: 1px solid ${control_border};
    border-top-right-radius: 9px;
    width: 18px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: ${surface_hover};
    border: 1px solid ${control_border};
    border-bottom-right-radius: 9px;
    width: 18px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: ${btn_hover};
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: ${accent};
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: ${surface};
    border: 1px solid ${control_border};
    border-radius: 10px;
    selection-background-color: ${accent_soft};
    selection-color: ${title};
    color: ${fg};
}

/* ---- tables ---- */
QTableWidget {
    background-color: ${surface};
    alternate-background-color: ${surface_hover};
    border: 1px solid ${border};
    border-radius: 12px;
    gridline-color: ${grid};
    selection-background-color: ${accent_soft};
    selection-color: ${title};
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: ${accent_soft};
}

QHeaderView::section {
    background-color: ${surface_hover};
    color: ${muted};
    border: none;
    border-bottom: 1px solid ${border};
    padding: 10px 8px;
    font-weight: bold;
}

/* ---- text / scroll ---- */
QTextEdit, QTextEdit[readonly="true"] {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 12px;
    color: ${fg};
    selection-background-color: ${accent_border};
}

QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: ${scroll_handle};
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: ${scroll_handle_hover};
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: transparent;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: ${scroll_handle};
    border-radius: 5px;
    min-width: 30px;
}

/* ---- tabs ---- */
QTabWidget::pane {
    border: 1px solid ${border};
    border-radius: 12px;
    background-color: ${surface};
    top: -1px;
}

QTabBar::tab {
    background-color: transparent;
    color: ${muted};
    border: none;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: ${accent};
    border-bottom-color: ${accent};
}

QTabBar::tab:hover {
    color: ${title};
}

/* ---- progress ---- */
QProgressBar {
    background-color: ${border};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: ${muted};
    height: 8px;
}

QProgressBar::chunk {
    background-color: ${green_btn};
    border-radius: 4px;
}

/* ---- labels ---- */
QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: ${title};
}

QLabel#subtitle {
    font-size: 12px;
    color: ${muted};
}

QLabel#status-loaded {
    color: ${green};
    font-weight: bold;
}

QLabel#status-unloaded {
    color: ${muted};
}

QLabel#status-loading {
    color: ${warn};
}

QLabel#status-failed {
    color: ${red};
}

QSplitter::handle {
    background-color: ${border};
}

QSplitter::handle:horizontal {
    width: 1px;
}

/* ---- chat ---- */
#chat_input {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 14px;
    padding: 12px;
    color: ${fg};
}

#chat_input:focus {
    border-color: ${accent};
}

#message_user {
    background-color: ${bubble_user};
    border: none;
    border-radius: 14px;
    padding: 12px;
}

#message_assistant {
    background-color: ${surface};
    border: 1px solid ${border};
    border-radius: 14px;
    padding: 12px;
}
"""

_QSS_TEMPLATES = {"dark": Template(_DARK_QSS), "light": Template(_LIGHT_QSS)}


def theme_tokens(name: str = "dark", overrides: dict | None = None) -> dict:
    """A complete token map: the palette of `name` with `overrides` on top."""
    tokens = dict(_PALETTES.get(name, _PALETTES["dark"]))
    if overrides:
        tokens.update({k: v for k, v in overrides.items() if v})
    return tokens


def render_theme(name: str = "dark", overrides: dict | None = None) -> str:
    """Fill the stylesheet template with a palette.

    Raises KeyError if a token the template needs is missing, so a half-built
    custom palette fails loudly instead of shipping a half-coloured UI.
    """
    template = _QSS_TEMPLATES.get(name, _QSS_TEMPLATES["dark"])
    return template.substitute(theme_tokens(name, overrides))


DARK_THEME = render_theme("dark")
LIGHT_THEME = render_theme("light")
THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}


def apply_theme(app, name: str, overrides: dict | None = None):
    """Switch the running app's theme; safe to call anytime.

    `overrides` lets a custom palette ride on top of `name`'s base shape. The
    merged palette is published as "custom" so c() and the rail QSS in
    window.py resolve exactly the colours the stylesheet was rendered with.
    """
    global current_theme
    if overrides:
        _PALETTES["custom"] = theme_tokens(name, overrides)
        current_theme = "custom"
        app.setStyleSheet(render_theme(name, overrides))
        return
    name = name if name in THEMES else "dark"
    current_theme = name
    app.setStyleSheet(THEMES[name])
