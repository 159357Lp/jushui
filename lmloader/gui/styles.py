"""Theme stylesheets for LM Loader GUI — dark + light, switchable at runtime.

Pages that embed colors inside inline QSS should pull them from `c(name)`
instead of hard-coding, so a theme switch recolors them without rebuilding.

Design tokens follow the "LM Loader UI 重设计" spec:
  * light theme = light canvas (#F5F6FA) + white cards + blue accent (#2E6BE6)
    + DARK rail sidebar (#151A2C), 12px card radius, 10px control radius.
"""

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
        "log_bg": "#0d1117",        # terminal log panel
        "log_fg": "#c9d1d9",
        # sidebar (dark rail) — dark theme keeps the same near-black rail
        "sidebar": "#0d1117",
        "sidebar_hover": "#21262d",
        "sidebar_fg": "#8b949e",
        "sidebar_active": "#1f6feb",
        "chip_bg": "#21262d",
        "chip_border": "#30363d",
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
        "log_bg": "#0F1420",        # console-style log panel
        "log_fg": "#A9B1C6",
        # dark rail sidebar (design spec: sidebar stays dark in light theme)
        "sidebar": "#151A2C",
        "sidebar_hover": "#232A42",
        "sidebar_fg": "#8A93A8",
        "sidebar_active": "#2E6BE6",
        "chip_bg": "#F1F3F9",
        "chip_border": "#E4E8F1",
    },
}

current_theme = "dark"


def c(name: str) -> str:
    """Theme-aware color lookup for inline QSS in pages."""
    return _PALETTES[current_theme].get(name, _PALETTES["dark"].get(name, "#000000"))


DARK_THEME = """
QMainWindow {
    background-color: #0d1117;
}

QWidget {
    color: #c9d1d9;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

QSidebar, #sidebar {
    background-color: #161b22;
    border: none;
}

#sidebar QPushButton {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-size: 14px;
}

#sidebar QPushButton:hover {
    background-color: #21262d;
    color: #c9d1d9;
}

#sidebar QPushButton[active="true"] {
    background-color: #1f6feb22;
    color: #58a6ff;
    border-left: 3px solid #58a6ff;
}

/* Collapsed rail: centered glyphs, no side padding */
#sidebar QPushButton[rail="true"] {
    padding: 10px 0;
    text-align: center;
    font-size: 15px;
}

#sidebar QPushButton#sidebarToggle {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-radius: 6px;
    font-size: 17px;
    text-align: left;
    padding: 4px 14px;
}

#sidebar QPushButton#sidebarToggle:hover {
    background-color: #21262d;
    color: #c9d1d9;
}

#sidebar QPushButton#sidebarToggle[rail="true"] {
    padding: 4px 0;
    text-align: center;
}

QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #58a6ff;
}

#modelCard {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

#modelCard:hover {
    background-color: #1c2129;
    border-color: #8b949e;
}

QFrame#card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0 12px;
    min-height: 30px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background-color: #0d1117;
}

QPushButton:disabled {
    color: #484f58;
    border-color: #21262d;
}

QPushButton#primary {
    background-color: #238636;
    border-color: #2ea043;
    color: white;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #2ea043;
}

QPushButton#primary:disabled {
    background-color: #21262d;
    color: #484f58;
}

QPushButton#danger {
    background-color: #da3633;
    border-color: #f85149;
    color: white;
}

QPushButton#danger:hover {
    background-color: #f85149;
}

/* Horizontal padding only. Fusion + QSS *vertical* padding mis-computes the
   text rect after a window resize and long text (e.g. a binary path) gets
   its glyphs clipped/overlapped; min-height keeps the height instead. */
QLineEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0 8px;
    min-height: 30px;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}

/* QSpinBox/QComboBox are complex widgets (border + editor + buttons). A QSS
   padding on the whole widget makes the Fusion style mis-compute the inner
   editor rect; after a window resize the value text is then painted into a
   wrong/clipped rect and its glyphs come out corrupted (999 -> "uuu").
   So: keep background/border, drop the padding. */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    min-height: 30px;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}

/* On Windows the native style can override the text color of the line
   edit embedded inside QSpinBox/QComboBox (values become black on the dark
   background and invisible). With the Fusion style the top-level color rule
   is honoured, and overriding the inner editor's background/border is
   harmful: Qt's documented widget-rendering issue with sub-control
   backgrounds makes the inner text vanish after a window resize. So we only
   pin the color here. */
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-top-right-radius: 5px;
    width: 18px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-bottom-right-radius: 5px;
    width: 18px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #58a6ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #1f6feb;
    color: #c9d1d9;
}

QTableWidget {
    background-color: #0d1117;
    alternate-background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    gridline-color: #21262d;
    selection-background-color: #1f6feb33;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #1f6feb33;
}

QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    border: none;
    border-bottom: 1px solid #30363d;
    padding: 8px 6px;
    font-weight: bold;
}

QTextEdit, QTextEdit[readonly="true"] {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}

QScrollArea {
    background-color: #0d1117;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #0d1117;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 4px;
    min-width: 30px;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 8px;
    background-color: #0d1117;
}

QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: none;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: #58a6ff;
    border-bottom-color: #58a6ff;
}

QTabBar::tab:hover {
    color: #c9d1d9;
    background-color: #21262d;
}

QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #c9d1d9;
    height: 8px;
}

QProgressBar::chunk {
    background-color: #238636;
    border-radius: 4px;
}

QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #f0f6fc;
}

QLabel#subtitle {
    font-size: 12px;
    color: #8b949e;
}

QLabel#status-loaded {
    color: #3fb950;
    font-weight: bold;
}

QLabel#status-unloaded {
    color: #8b949e;
}

QLabel#status-loading {
    color: #d29922;
}

QLabel#status-failed {
    color: #f85149;
}

QSplitter::handle {
    background-color: #21262d;
}

QSplitter::handle:horizontal {
    width: 1px;
}

#chat_input {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
    color: #c9d1d9;
}

#chat_input:focus {
    border-color: #58a6ff;
}

#message_user {
    background-color: #1f6feb22;
    border: 1px solid #1f6feb44;
    border-radius: 8px;
    padding: 10px;
}

#message_assistant {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
}
"""

# ------------------------------------------------------------- light theme
# Design spec: light canvas, white cards (12px radius), blue accent #2E6BE6,
# dark rail sidebar (#151A2C) with a blue active pill.

LIGHT_THEME = """
QMainWindow {
    background-color: #F5F6FA;
}

QWidget {
    color: #3A4154;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ---- dark rail sidebar (stays dark on purpose) ---- */
QSidebar, #sidebar {
    background-color: #151A2C;
    border: none;
}

#sidebar QLabel {
    color: #8A93A8;
    background: transparent;
}

#sidebar QPushButton {
    background-color: transparent;
    color: #8A93A8;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: left;
    font-size: 14px;
}

#sidebar QPushButton:hover {
    background-color: #232A42;
    color: #FFFFFF;
}

#sidebar QPushButton[active="true"] {
    background-color: #2E6BE6;
    color: #FFFFFF;
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
    color: #8A93A8;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    text-align: left;
    padding: 4px 14px;
}

#sidebar QPushButton#sidebarToggle:hover {
    background-color: #232A42;
    color: #FFFFFF;
}

#sidebar QPushButton#sidebarToggle[rail="true"] {
    padding: 4px 0;
    text-align: center;
}

/* ---- cards ---- */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 12px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #1A2032;
}

#modelCard {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 12px;
}

#modelCard:hover {
    background-color: #FFFFFF;
    border-color: #C7DAFF;
}

/* Generic white card used by pages that group controls outside a QGroupBox
   (e.g. the download page's manual-fetch panel). */
QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 12px;
}

/* ---- buttons ---- */
QPushButton {
    background-color: #FFFFFF;
    color: #3A4154;
    border: 1px solid #E4E8F1;
    border-radius: 10px;
    padding: 0 14px;
    min-height: 32px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #F1F3F9;
    border-color: #C7DAFF;
}

QPushButton:pressed {
    background-color: #E4E8F1;
}

QPushButton:disabled {
    color: #B3BAC8;
    border-color: #E9ECF4;
    background-color: #F8F9FC;
}

QPushButton#primary {
    background-color: #2E6BE6;
    border-color: #2E6BE6;
    color: #FFFFFF;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #4A82F0;
    border-color: #4A82F0;
}

QPushButton#primary:disabled {
    background-color: #C7DAFF;
    border-color: #C7DAFF;
    color: #FFFFFF;
}

QPushButton#danger {
    background-color: #E5484D;
    border-color: #E5484D;
    color: #FFFFFF;
}

QPushButton#danger:hover {
    background-color: #F0666A;
    border-color: #F0666A;
}

/* ---- inputs ----
   Horizontal padding only. Fusion + QSS *vertical* padding mis-computes the
   text rect after a window resize and long text (e.g. a binary path) gets
   its glyphs clipped/overlapped; min-height keeps the height instead. */
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #E4E8F1;
    border-radius: 10px;
    padding: 0 12px;
    min-height: 34px;
    color: #3A4154;
    selection-background-color: #C7DAFF;
}

/* Complex widgets: keep background/border, drop padding (see DARK_THEME). */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #E4E8F1;
    border-radius: 10px;
    min-height: 34px;
    color: #3A4154;
    selection-background-color: #C7DAFF;
}

QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    color: #3A4154;
    selection-background-color: #C7DAFF;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #F8F9FC;
    border: 1px solid #E4E8F1;
    border-top-right-radius: 9px;
    width: 18px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #F8F9FC;
    border: 1px solid #E4E8F1;
    border-bottom-right-radius: 9px;
    width: 18px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #F1F3F9;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2E6BE6;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #E4E8F1;
    border-radius: 10px;
    selection-background-color: #EAF2FF;
    selection-color: #1A2032;
    color: #3A4154;
}

/* ---- tables ---- */
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8F9FC;
    border: 1px solid #E9ECF4;
    border-radius: 12px;
    gridline-color: #F1F3F9;
    selection-background-color: #EAF2FF;
    selection-color: #1A2032;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #EAF2FF;
}

QHeaderView::section {
    background-color: #F8F9FC;
    color: #5A6478;
    border: none;
    border-bottom: 1px solid #E9ECF4;
    padding: 10px 8px;
    font-weight: bold;
}

/* ---- text / scroll ---- */
QTextEdit, QTextEdit[readonly="true"] {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 12px;
    color: #3A4154;
    selection-background-color: #C7DAFF;
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
    background-color: #DDE3EE;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #C6CBD8;
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
    background-color: #DDE3EE;
    border-radius: 5px;
    min-width: 30px;
}

/* ---- tabs ---- */
QTabWidget::pane {
    border: 1px solid #E9ECF4;
    border-radius: 12px;
    background-color: #FFFFFF;
    top: -1px;
}

QTabBar::tab {
    background-color: transparent;
    color: #5A6478;
    border: none;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: #2E6BE6;
    border-bottom-color: #2E6BE6;
}

QTabBar::tab:hover {
    color: #1A2032;
}

/* ---- progress ---- */
QProgressBar {
    background-color: #E9ECF4;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #5A6478;
    height: 8px;
}

QProgressBar::chunk {
    background-color: #22A06B;
    border-radius: 4px;
}

/* ---- labels ---- */
QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #1A2032;
}

QLabel#subtitle {
    font-size: 12px;
    color: #5A6478;
}

QLabel#status-loaded {
    color: #1F9763;
    font-weight: bold;
}

QLabel#status-unloaded {
    color: #5A6478;
}

QLabel#status-loading {
    color: #E5853D;
}

QLabel#status-failed {
    color: #E5484D;
}

QSplitter::handle {
    background-color: #E9ECF4;
}

QSplitter::handle:horizontal {
    width: 1px;
}

/* ---- chat ---- */
#chat_input {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 14px;
    padding: 12px;
    color: #3A4154;
}

#chat_input:focus {
    border-color: #2E6BE6;
}

#message_user {
    background-color: #E9F0FF;
    border: none;
    border-radius: 14px;
    padding: 12px;
}

#message_assistant {
    background-color: #FFFFFF;
    border: 1px solid #E9ECF4;
    border-radius: 14px;
    padding: 12px;
}
"""

THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}


def apply_theme(app, name: str):
    """Switch the running app's theme; safe to call anytime."""
    global current_theme
    name = name if name in THEMES else "dark"
    current_theme = name
    app.setStyleSheet(THEMES[name])
