"""Regression guards for the model page's loaded-state polling.

Context — the bug these lock down:

The model page computed ``_loaded_names`` exactly once, inside ``_refresh()``,
which only ran when the page was navigated to or the user pressed刷新. So an
engine that died *after* that snapshot (taskkill, a crash, or the GUI's
kill-on-close job object taking it down) left its card wearing a green
"已加载 primary 8100" badge indefinitely, with the 卸载 button still offered.
The user could not tell a live engine from a dead one.

The backend side was already fixed (``get_loaded_models`` reaps dead instances);
what was missing was the GUI re-reading that answer.

What is asserted here:

* a loaded-set change must reach ``_loaded_names`` without a manual refresh;
* an *unchanged* set must NOT rebuild the cards (rebuilding every tick would
  reset scroll position and flicker the buttons);
* a failed poll must never be interpreted as "nothing is loaded";
* the poll must not start a second request while one is in flight;
* the poll must stop when the page is navigated away from.
"""

import re

import pytest

from lmloader.gui.pages import models as models_page
from lmloader.gui.pages.models import POLL_INTERVAL_MS


# ------------------------------------------------------------------ structure

def test_poll_timer_is_created_and_wired():
    """The page must own a poll timer connected to the loaded-set fetcher."""
    src = models_page.__file__
    text = open(src, encoding="utf-8").read()
    assert "_poll_timer = QTimer(self)" in text
    assert "_poll_timer.timeout.connect(self._poll_loaded)" in text


def test_poll_timer_is_started_on_navigate():
    text = open(models_page.__file__, encoding="utf-8").read()
    navigate = text.split("def on_navigate")[1].split("\n    def ")[0]
    assert "_poll_timer.start" in navigate, (
        "on_navigate must start the poll, otherwise the badges never update "
        "while the user sits on the page"
    )


def test_poll_interval_is_sane():
    """Too fast hammers the backend; too slow defeats the purpose."""
    assert 500 <= POLL_INTERVAL_MS <= 10000


def test_on_leave_stops_the_timer():
    text = open(models_page.__file__, encoding="utf-8").read()
    leave = text.split("def on_leave")[1].split("\n    def ")[0]
    assert "_poll_timer.stop()" in leave


def test_navigation_calls_on_leave():
    """The window must notify the page it navigated away from."""
    src = models_page.__file__
    import lmloader.gui.window as win
    text = open(win.__file__, encoding="utf-8").read()
    navigate = text.split("def _navigate")[1].split("\ndef ")[0]
    assert "on_leave" in navigate, (
        "window._navigate must call on_leave, otherwise a hidden model page "
        "keeps polling forever"
    )


# ------------------------------------------------------------ refresh guarding

def test_refresh_is_guarded_against_concurrent_polls():
    """A full refresh and the poll can both be in flight; only one at a time."""
    text = open(models_page.__file__, encoding="utf-8").read()
    refresh = text.split("def _refresh")[1].split("\n    def ")[0]
    assert "if self._refreshing" in refresh, (
        "_refresh must bail out when a request is already running, otherwise a "
        "poll landing mid-refresh can repopulate the list from a stale snapshot"
    )


def test_poll_is_guarded_against_concurrent_polls():
    poll = open(models_page.__file__, encoding="utf-8").read().split(
        "def _poll_loaded")[1].split("\n    def ")[0]
    assert "if self._refreshing" in poll
    assert "return" in poll.split("if self._refreshing")[1].split("\n")[0] + \
        poll.split("if self._refreshing")[1][:40]


def test_refresh_clears_the_busy_flag_on_success():
    text = open(models_page.__file__, encoding="utf-8").read()
    refresh = text.split("def _refresh")[1].split("\n    # ---")[0]
    assert refresh.count("self._refreshing = False") >= 2, (
        "both the success and the error callback must clear _refreshing, or "
        "the page stops polling forever after a single failure"
    )


# ------------------------------------------------------- failure must not lie

def test_failed_poll_does_not_clear_loaded_state():
    """A network error must not be read as "the engine is gone".

    Otherwise a momentary timeout makes every badge flip to未加载 while the
    engine is still running — the exact inverse of the bug being fixed.
    """
    text = open(models_page.__file__, encoding="utf-8").read()
    poll = open(models_page.__file__, encoding="utf-8").read().split(
        "def _poll_loaded")[1].split("\n    def ")[0]
    on_error = poll.split("def on_error")[1]
    assert "_loaded_names = set()" not in on_error
    assert "_populate_list()" not in on_error, (
        "the error path must leave the cards exactly as they were"
    )


def test_poll_only_rebuilds_cards_when_the_set_changes():
    """Unconditional _populate_list() on every tick would reset scroll."""
    poll = open(models_page.__file__, encoding="utf-8").read().split(
        "def _poll_loaded")[1].split("\n    def ")[0]
    result = poll.split("def on_result")[1].split("def on_error")[0]
    assert "if names == self._loaded_names:" in result
    guard = result.split("if names == self._loaded_names:")[1]
    early_return, _, after = guard.partition("return")
    assert early_return.strip(), "the unchanged branch must return early"
    # _populate_list must live in the *changed* branch, i.e. after the return.
    assert "_populate_list()" in after.split("return")[-1] or \
        "_populate_list()" in after
    assert "_populate_list()" not in early_return


def test_poll_still_refreshes_metrics_when_set_is_unchanged():
    """Even with the same names, tg/s figures must keep updating."""
    poll = open(models_page.__file__, encoding="utf-8").read().split(
        "def _poll_loaded")[1].split("\n    def ")[0]
    result = poll.split("def on_result")[1].split("def on_error")[0]
    guard = result.split("if names == self._loaded_names:")[1]
    early_return = guard.partition("return")[0]
    assert "self._loaded_info = info" in early_return, (
        "the loaded detail dict carries live metrics (tg t/s, MTP acceptance); "
        "it must be refreshed even when the name set is unchanged"
    )


# ------------------------------------------------------------------- behaviour

class _FakeApi:
    """Stands in for LMLoaderAPI; records how often it was polled.

    ``fetch_loaded_models_or_none`` mirrors the real contract: a payload of
    ``None`` means the request failed, which is NOT the same as an empty list.
    """
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def fetch_loaded_models_or_none(self):
        self.calls += 1
        if not self._payloads:
            return []
        return self._payloads.pop(0)


def _entry(name, port=8100, pid=1):
    return {"model": {"name": name}, "port": port, "pid": pid}


def _make_page(loaded_names, payloads):
    """A ModelsPage wired for polling without running __init__.

    __init__ is skipped deliberately: it builds the whole card list and needs a
    live style stack. Everything _poll_loaded touches is stubbed instead, so the
    test exercises the real polling logic rather than a re-implementation.
    """
    page = models_page.ModelsPage.__new__(models_page.ModelsPage)
    page._refreshing = False
    page._refresh_thread = None
    page._models = []
    page._loaded_names = set(loaded_names)
    page._loaded_info = {n: _entry(n) for n in loaded_names}
    page.api = _FakeApi(payloads)
    page.rebuilt = []
    page._populate_list = lambda: page.rebuilt.append(True)
    page.status_label = type("S", (), {"setText": lambda self, t: None})()
    return page


@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_poll_drops_a_dead_engine_from_the_loaded_set(qt_app):
    """The reported bug: engine died, badge stayed green."""
    page = _make_page(["qwen35-9b-mtp-nvfp4.gguf"], [[]])

    models_page.ModelsPage._poll_loaded(page)
    _settle(qt_app, page)

    assert page._loaded_names == set(), "dead engine must leave the loaded set"
    assert page.rebuilt == [True], "the cards must be rebuilt so the badge clears"
    assert page._refreshing is False


def test_poll_adds_a_newly_loaded_engine(qt_app):
    page = _make_page([], [[_entry("gemma-4-12b-it-nvfp4")]])

    models_page.ModelsPage._poll_loaded(page)
    _settle(qt_app, page)

    assert page._loaded_names == {"gemma-4-12b-it-nvfp4"}
    assert page.rebuilt == [True]


def test_poll_skips_rebuild_when_nothing_changed(qt_app):
    page = _make_page(["a.gguf"], [[_entry("a.gguf", port=8100, pid=99)]])

    models_page.ModelsPage._poll_loaded(page)
    _settle(qt_app, page)

    assert page.rebuilt == [], "an unchanged set must not rebuild the cards"
    assert page.api.calls == 1


def test_failed_poll_leaves_the_badge_alone(qt_app):
    """A failed query must not be read as "every engine is gone".

    This is the regression that made the original fix actively harmful: the
    client used to swallow the exception and return [], so the poll saw an empty
    list, concluded nothing was loaded, and blanked every badge on any timeout.

    The assertion has to be on the *outcome*, not merely on "no crash": with the
    ``if loaded is None`` guard removed, the poll raises TypeError and lands in
    on_error — which also happens to leave the badges alone. Both paths pass a
    naive check, so this test also pins the call count and the absence of any
    rebuild, and is paired with the mutation script which proves the guard is
    load-bearing.
    """
    page = _make_page(["a.gguf"], [None])          # None == query failed

    models_page.ModelsPage._poll_loaded(page)
    _settle(qt_app, page)

    assert page._loaded_names == {"a.gguf"}, (
        "a failed query must leave the loaded set untouched"
    )
    assert page.rebuilt == [], "no rebuild on a failed query"
    assert page.api.calls == 1, "exactly one request, no silent retry loop"
    assert page._refreshing is False


def test_failed_poll_short_circuits_before_touching_the_payload(qt_app):
    """The None guard must return BEFORE the payload is walked.

    Why this needs its own test: removing the guard does NOT blank the badges.
    The comprehension raises TypeError inside the worker, run_in_thread catches
    it and routes to on_error, and on_error also happens to leave the badges
    alone. So the behavioural test above passes either way -- only a check on the
    actual control flow can tell the two apart.
    """
    src = open(models_page.__file__, encoding="utf-8").read()
    poll = src.split("def _poll_loaded")[1].split("\n    def ")[0]
    result = poll.split("def on_result")[1].split("def on_error")[0]

    assert "if loaded is None:" in result, (
        "on_result must guard against a None payload before iterating it"
    )
    guard_body = result.split("if loaded is None:")[1]
    before_comprehension = guard_body.split("names =")[0]
    assert "return" in before_comprehension, (
        "the None guard must return before the set comprehension runs; "
        "otherwise a failed query raises TypeError and becomes "
        "indistinguishable from a deliberate early-out"
    )

def test_empty_list_does_clear_the_badges(qt_app):
    """The mirror image: a genuine empty list MUST clear them.

    Without this, "leave it alone on failure" could be implemented by never
    clearing anything, which is the same bug wearing a different hat.
    """
    page = _make_page(["a.gguf"], [[]])

    models_page.ModelsPage._poll_loaded(page)
    _settle(qt_app, page)

    assert page._loaded_names == set(), (
        "an authoritative empty list means the engines are gone"
    )
    assert page.rebuilt == [True]


def test_second_poll_while_one_is_in_flight_is_skipped(qt_app):
    page = _make_page(["a.gguf"], [[], []])
    page._refreshing = True          # a request is already running

    models_page.ModelsPage._poll_loaded(page)

    assert page.api.calls == 0, "the poll must not start a concurrent request"


def _settle(app, page, timeout_ms=5000):
    """Spin the event loop until the worker thread delivers its callback.

    run_in_thread() uses a real QThread with a queued cross-thread signal, so
    the callback lands on the next event-loop turn — it is never synchronous.
    """
    import time
    waited = 0.0
    while waited < timeout_ms / 1000.0:
        app.processEvents()
        if not page._refreshing:
            return
        time.sleep(0.005)
        waited += 0.005
    raise AssertionError("poll callback never fired")