"""Repetition-loop guard: detector, request payload, engine sampling args."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.gui.repetition import (  # noqa: E402
    MIN_SCAN_CHARS,
    REPEAT_THRESHOLD,
    detect_repetition_loop,
)


# ---------- detector: healthy output must never trip ----------

def test_short_text_never_trips():
    assert detect_repetition_loop("") is None
    assert detect_repetition_loop("好的，我来帮你。") is None


def test_normal_answer_does_not_trip():
    text = (
        "嗯，用户问我能不能读取 PDF。我需要先确认自己的能力边界。"
        "作为 Qwen3.5，我确实支持处理 PDF 文件，但需要用户主动上传。"
        "不能直接读取本地文件，必须通过上传。然后要强调功能，"
        "比如全文分析、信息提取，可能还有图表、公式这些。"
    ) * 3
    assert detect_repetition_loop(text) is None


def test_code_block_does_not_trip():
    """Legitimately repetitive content: a code sample repeated by the user."""
    snippet = "for i in range(10):\n    print(i)\n"
    assert detect_repetition_loop("示例代码：\n" + snippet * 5) is None


def test_markdown_table_does_not_trip():
    """Dense material/order table: repeated column values, distinct whole rows.

    The old 16-char detector false-tripped here — a column value such as
    " | dense busbar | 2000" recurred on every row — and cut the table
    half-read. The 48-char whole-row detector must pass, while a verbatim
    sentence loop still trips.
    """
    header = ("|序号|产品名称|产品规格|单位|数量|安装序号|单位长度|总长度|\n"
              "|:--|:--|:--|:--|:--|:--|:--|:--|\n")
    specs = ["垂直弯","直段","水平弯","水平弯","垂直弯","直段"]
    vals = [1.65,1.745,1.39,1.39,1.65,1.745]
    rows = []
    for i in range(26, 66):
        sp = specs[(i - 26) % 6]; v = vals[(i - 26) % 6]
        rows.append(f"| {i} | 密集型母线槽 | 2000A/5P{sp} | 段 | 1 | C{i} | {v} | {v} |")
    text = header + "\n".join(rows)
    assert len(text) > MIN_SCAN_CHARS
    assert detect_repetition_loop(text) is None


def test_long_prose_with_repeated_words_does_not_trip():
    """Same 16-char fragment appearing a few times is normal English/Chinese."""
    unit = "这是一个很常见的开头部分"
    text = (unit + "，后面接不同的内容说明。" * 5) * 4
    assert detect_repetition_loop(text) is None


def test_verbatim_paragraph_repeat_boundary():
    """Quoting a paragraph a few times passes; a long run of it trips.

    The boundary matters both ways: a model restating a definition or a code
    sample verbatim is legitimate, while the same paragraph repeated for
    hundreds of characters is degenerate by any reading.

    Note the short-run case stays below MIN_SCAN_CHARS on purpose — a handful
    of repeats inside a 200-character answer carries too little signal to be
    worth truncating. Measured on a 30-char paragraph the guard trips at 14
    repeats (~420 chars), versus 4186 tokens in the original bug report.
    """
    para = "嗯，用户问我能不能读取 PDF。我需要先确认自己的能力边界。"
    assert detect_repetition_loop(para * 4) is None
    assert detect_repetition_loop(para * 6) is None       # still under the floor
    assert detect_repetition_loop(para * 14) is not None  # 420 chars: trips


def test_distinct_sentences_never_trip():
    """Real reasoning prose: overlapping words, no repeated 16-char span."""
    text = "".join([
        "嗯，用户问我能不能读取 PDF 文件。",
        "我需要先确认自己的能力边界和格式支持情况。",
        "作为 Qwen3.5，我确实可以处理 PDF 的文字内容。",
        "但这需要用户主动把文件上传到对话里才行。",
        "我不能直接读取用户本地磁盘上的任意路径文件。",
        "如果用户不方便上传，也可以把文字直接粘贴过来。",
        "另外扫描件需要先经过 OCR 处理才能提取文字。",
        "提取出来之后我可以做总结、翻译或者按要点问答。",
    ])
    assert detect_repetition_loop(text) is None


# ---------- detector: the real failure must trip ----------

DEGEN_UNIT = "Wait. I'll write a story. Okay. "


def test_degenerate_reasoning_trips():
    """The exact shape from the bug report: 'Wait. I'll write a story.' on repeat."""
    text = DEGEN_UNIT * 16
    assert len(text) > MIN_SCAN_CHARS
    hit = detect_repetition_loop(text)
    assert hit is not None
    assert hit["reason"] == "repetition_loop"
    assert hit["occurrences"] >= REPEAT_THRESHOLD
    assert len(hit["fragment"]) == 48


def test_degenerate_content_trips():
    text = "这是一个非常无聊的重复回答。" * 40
    assert len(text) > MIN_SCAN_CHARS
    hit = detect_repetition_loop(text)
    assert hit is not None
    assert hit["occurrences"] >= REPEAT_THRESHOLD


def test_repetition_below_min_chars_does_not_trip():
    """A loop shorter than MIN_SCAN_CHARS is intentionally not judged.

    Early output is short and legitimately repetitive; tripping here would
    truncate answers before they have any content.
    """
    text = "这是一个非常无聊的重复回答。" * 8
    assert len(text) < MIN_SCAN_CHARS
    assert detect_repetition_loop(text) is None


def test_short_repetition_below_min_chars_does_not_trip():
    """Looping only 3 times at the very start is not enough signal."""
    text = "Wait. I'll write a story. Okay. Wait. I'll write a story. Okay."
    assert len(text) < MIN_SCAN_CHARS
    assert detect_repetition_loop(text) is None


def test_barely_below_min_chars_does_not_trip():
    """12 repeats = 384 chars, just under the 400-char scan floor."""
    text = DEGEN_UNIT * 12
    assert len(text) < MIN_SCAN_CHARS
    assert detect_repetition_loop(text) is None


def test_loop_below_threshold_does_not_trip():
    """A fragment repeated 5x must pass — 6x is the trip point."""
    frag = "abcdefghijklmnop"
    text = frag * (REPEAT_THRESHOLD - 1) + "结尾各不相同的内容在这里"
    text = text + "补充一些别的文字让它超过最小扫描长度"
    assert detect_repetition_loop(text) is None


def test_diagnostic_payload_shape():
    hit = detect_repetition_loop(DEGEN_UNIT * 16)
    assert set(hit) == {"reason", "fragment", "occurrences", "scanned_chars"}
    assert hit["scanned_chars"] > 0


def test_detector_window_is_bounded():
    """A long healthy prefix must not make the scan cost grow unbounded."""
    text = "健康的内容。" * 5000 + (DEGEN_UNIT * 16)
    hit = detect_repetition_loop(text)
    assert hit is not None
    assert hit["scanned_chars"] <= 4000


def test_guard_reacts_within_a_few_hundred_chars():
    """The screenshot showed 4186 tokens of runaway; the guard must fire far earlier."""
    hit = detect_repetition_loop(DEGEN_UNIT * 16)
    assert hit["scanned_chars"] <= 600


# ---------- request payload: max_tokens must always be sent ----------

class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(["data: [DONE]"])

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, captured, **kw):
        self._captured = captured
        self.kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, json=None, headers=None):
        self._captured.update(json or {})
        return _FakeResponse()


def test_sse_payload_includes_max_tokens(monkeypatch):
    import lmloader.gui.api_client as ac

    captured: dict = {}

    def fake_client(timeout=None, **kw):
        return _FakeClient(captured, timeout=timeout, **kw)

    monkeypatch.setattr(ac.httpx, "Client", fake_client)
    api = ac.LMLoaderAPI()
    list(api.chat_stream_sse("m", [{"role": "user", "content": "hi"}], 0.7,
                             True, ac.DEFAULT_MAX_TOKENS))
    assert captured["max_tokens"] == 8192


def test_sse_payload_omits_max_tokens_when_not_given(monkeypatch):
    """Documents the runaway: without max_tokens the engine streams unbounded."""
    import lmloader.gui.api_client as ac

    captured: dict = {}
    monkeypatch.setattr(ac.httpx, "Client",
                        lambda timeout=None, **kw: _FakeClient(captured, timeout=timeout, **kw))
    api = ac.LMLoaderAPI()
    list(api.chat_stream_sse("m", [{"role": "user", "content": "hi"}], 0.7, True, None))
    assert "max_tokens" not in captured


def test_default_max_tokens_value():
    from lmloader.gui.api_client import DEFAULT_MAX_TOKENS
    assert DEFAULT_MAX_TOKENS == 8192


def test_non_stream_payload_includes_max_tokens(monkeypatch):
    import lmloader.gui.api_client as ac

    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json or {})
        return _Resp()

    monkeypatch.setattr(ac.httpx, "post", fake_post)
    api = ac.LMLoaderAPI()
    assert api.chat_stream("m", [{"role": "user", "content": "hi"}], 0.7,
                            True, 8192) == "ok"
    assert captured["max_tokens"] == 8192


# ---------- engine sampling args ----------

def test_router_injects_repetition_args():
    """Engine must launch with repetition suppression on."""
    from lmloader.engine.router import EngineRouter
    from lmloader.config import EngineDefaultsConfig

    assert EngineDefaultsConfig().repeat_penalty == 1.1
    assert EngineDefaultsConfig().dry_multiplier == 0.8

    router = EngineRouter()
    args = _launch_args(router)
    assert "--repeat-penalty" in args
    assert args[args.index("--repeat-penalty") + 1] == "1.1"
    assert "--dry-multiplier" in args
    assert args[args.index("--dry-multiplier") + 1] == "0.8"
    assert "--dry-penalty-last-n" in args


def test_router_omits_args_when_disabled():
    """repeat_penalty=1.0 / dry=0.0 means the user wants raw sampling."""
    from lmloader.engine.router import EngineRouter

    router = EngineRouter()
    args = _launch_args(router, run_config={"repeat_penalty": 1.0, "dry_multiplier": 0.0})
    assert "--repeat-penalty" not in args
    assert "--dry-multiplier" not in args


def test_router_honours_custom_values():
    from lmloader.engine.router import EngineRouter

    router = EngineRouter()
    args = _launch_args(router, run_config={"repeat_penalty": 1.25, "dry_multiplier": 0.5})
    assert args[args.index("--repeat-penalty") + 1] == "1.25"
    assert args[args.index("--dry-multiplier") + 1] == "0.5"


def _launch_args(router, run_config=None):
    from lmloader.engine.router import ModelRecord

    model = ModelRecord(
        id=999, name="probe-model.gguf", path="C:/tmp/probe-model.gguf",
        size=1024, architecture="llama", quantization="NVFP4",
        parameters="9B", metadata={"context_length": 32768}, backend="llama",
    )
    cfg = router.build_launch_config(model, 8199, mmproj_path="", run_config=run_config)
    return cfg.args


# ---------- i18n parity ----------

def test_repetition_keys_exist_in_both_locales():
    from lmloader.gui.i18n import TRANSLATIONS

    zh, en = TRANSLATIONS["zh"], TRANSLATIONS["en"]
    keys = ["chat.stop", "chat.stop_tip", "chat.stopped_notice", "chat.repetition_notice"]
    for k in keys:
        assert k in zh, f"missing zh key {k}"
        assert k in en, f"missing en key {k}"
