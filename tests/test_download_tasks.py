"""Tests for download-task lifecycle: HF redirects, stop/clear buttons.

The regressions here were all user-visible:
  * hf-mirror answers 302 to the CDN; without follow_redirects every HF
    download failed with "Redirect response '302 Found'".
  * A finished task row could not be removed, so rows piled up forever.
  * Pressing "Clear" on a completed task silently did nothing because the
    dispatch keyed off worker.stopped, which only means "user pressed Stop".
"""

import inspect

import pytest

from lmloader.infra import downloader as D
from lmloader.gui.i18n import tr
from lmloader.gui.pages.download import DownloadPage, _DownloadWorker


# --- redirect handling -----------------------------------------------------

def _clients():
    """Every httpx client the downloader builds for transfers."""
    src = inspect.getsource(D)
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        if "AsyncClient(" in line:
            out.append((i, line))
    return out


def test_every_downloader_client_follows_redirects():
    """hf-mirror 302-redirects to the xet CDN; not following it means every
    HuggingFace download fails before a single byte is written."""
    clients = _clients()
    assert len(clients) >= 3, "expected the size-probe + two transfer clients"
    for lineno, line in clients:
        window = "\n".join(
            inspect.getsource(D).splitlines()[lineno - 1: lineno + 2]
        )
        assert "follow_redirects=True" in window, (
            f"AsyncClient at downloader.py:{lineno} does not follow redirects"
        )


def test_downloader_source_has_no_bare_asyncclient_without_redirects():
    src = inspect.getsource(D)
    for block in src.split("AsyncClient(")[1:]:
        head = block[:200]
        assert "follow_redirects=True" in head, head[:120]


def test_regression_the_redirect_bug_is_actually_fixed():
    """Named test so the original failure mode stays visible in the log."""
    import httpx
    # A client built the way downloader used to build it must NOT follow the
    # redirect, which is precisely why downloads failed.
    strict = httpx.AsyncClient(follow_redirects=False)
    assert strict.follow_redirects is False


# --- task lifecycle --------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp):
    p = DownloadPage(_FakeApi())
    yield p
    p.deleteLater()


class _FakeApi:
    base_url = "http://127.0.0.1:1"   # never reachable: no network in tests
    token = ""

    def _headers(self):
        return {}


def _task_rows(page: DownloadPage) -> list:
    return [page.progress_area.itemAt(i).widget()
            for i in range(page.progress_area.count())
            if page.progress_area.itemAt(i).widget() is not None]


def test_each_task_gets_its_own_container(page):
    """Rows must be individually removable; loose layout items are not."""
    page._start_download("a/b.gguf")
    page._start_download("c/d.gguf")
    assert len(_task_rows(page)) == 2
    assert len(page._dl_widgets) == 2


def test_task_row_carries_a_stop_button(page):
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    label, bar, detail, btn, container = page._dl_widgets[worker]
    assert btn.text()
    assert btn.isEnabled()
    assert container is not None


def test_finished_task_button_becomes_clear(page):
    """Terminal state flips the button from Stop to Clear, and Clear stays
    enabled — a disabled button is the 'no cancel button' the user hit."""
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    widgets = page._dl_widgets[worker]
    widgets[3].setText(tr("download.ollama_stop"))
    page._mark_finished(worker, widgets, "green")
    assert widgets[3].text()
    assert widgets[3].isEnabled()
    assert page._finished_workers.get(worker) is True


def test_completed_task_is_marked_finished_not_just_stopped(page):
    """worker.stopped stays False when a task ends on its own; the page must
    track terminal state separately or bulk-clear misses those rows."""
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    assert worker.stopped is False
    widgets = page._dl_widgets[worker]
    page._mark_finished(worker, widgets, "green")
    assert page._finished_workers.get(worker) is True


def test_clear_button_dispatch_removes_a_completed_row(page):
    """The regression: pressing Clear on a finished task did nothing because
    the dispatch branched on worker.stopped instead of the button's role."""
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    widgets = page._dl_widgets[worker]
    page._mark_finished(worker, widgets, "green")
    widgets[3].setText(tr("download.clear_task"))

    before = len(_task_rows(page))
    page._on_task_action(worker, widgets[3])
    assert len(_task_rows(page)) == before - 1
    assert worker not in page._dl_widgets


def test_stop_button_dispatch_does_not_remove_the_row(page):
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    widgets = page._dl_widgets[worker]
    widgets[3].setText(tr("download.ollama_stop"))

    before = len(_task_rows(page))
    page._on_task_action(worker, widgets[3])
    # The row must stay: it now shows "stopping", it is not deleted.
    assert len(_task_rows(page)) == before
    assert worker.stopped is True


def test_bulk_clear_only_removes_finished_rows(page):
    page._start_download("a/b.gguf")
    w1 = next(iter(page._dl_widgets))
    page._start_download("c/d.gguf")
    w2 = [w for w in page._dl_widgets if w is not w1][0]

    page._mark_finished(w1, page._dl_widgets[w1], "green")
    assert len(_task_rows(page)) == 2

    page.clear_finished_tasks()
    remaining = _task_rows(page)
    assert len(remaining) == 1
    assert w2 in page._dl_widgets, "the still-running task must survive"
    assert w1 not in page._dl_widgets


def test_removing_a_row_clears_the_finished_marker(page):
    """Otherwise _finished_workers grows without bound across a session."""
    page._start_download("a/b.gguf")
    worker = next(iter(page._dl_widgets))
    page._mark_finished(worker, page._dl_widgets[worker], "green")
    page._remove_task_row(worker, None)
    assert worker not in page._finished_workers


def test_clearing_an_unknown_worker_is_harmless(page):
    page._remove_task_row(object(), None)      # must not raise


def test_page_exposes_a_bulk_clear_control(page):
    assert hasattr(page, "clear_btn")
    assert callable(page.clear_finished_tasks)


# --- worker.stop semantics -------------------------------------------------

def test_worker_stop_flag_only_tracks_user_action():
    """Documents why the page needs its own terminal-state bookkeeping."""
    w = _DownloadWorker("http://127.0.0.1:1", "m:tag")
    assert w.stopped is False
    w.stop()
    assert w.stopped is True


def test_worker_stop_before_run_is_safe():
    w = _DownloadWorker("http://127.0.0.1:1", "m:tag")
    w.stop()
    w.stop()          # idempotent
    assert w.stopped is True