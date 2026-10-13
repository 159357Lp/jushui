"""Tests for the Ollama library browser and pull cancellation.

The HTML parsers are exercised against fixtures captured from the live site
rather than hand-written markup: a hand-written fixture only proves the regex
matches the markup the author imagined. The live-capture tests below are
skipped when the network is unavailable, but the fixture tests always run.
"""

import asyncio
import inspect

import pytest

from lmloader.infra import ollama_search as S
from lmloader.infra.ollama_pull import CancelToken, PullCancelled, pull_via_cli
from lmloader.infra import ollama_pull


def _has_network() -> bool:
    try:
        import httpx
        httpx.get(S.SITE, timeout=6.0)
        return True
    except Exception:  # noqa: BLE001
        return False


requires_network = pytest.mark.skipif(
    not _has_network(), reason="no network access to ollama.com"
)


# --- spec validation -------------------------------------------------------

@pytest.mark.parametrize("spec", [
    "qwen2.5:7b", "llama3.2:3b", "qwen3", "ns/model:tag", "a", "0.5b",
    "model_1.2-rc1:instruct-q4_0",
])
def test_valid_specs_are_accepted(spec):
    assert S.is_valid_spec(spec)


@pytest.mark.parametrize("spec", [
    "../../etc/passwd", "http://evil/x", "a b", "", "  ", "a/b/c:tag/extra",
    "model:tag;rm -rf", "naïve", "a" * 400,
])
def test_traversal_and_junk_specs_are_rejected(spec):
    """Anything that could escape the registry URL must never reach the puller."""
    assert not S.is_valid_spec(spec)


def test_leading_whitespace_is_tolerated():
    assert S.is_valid_spec("  qwen3:8b  ")


# --- search page parser ----------------------------------------------------

SEARCH_HTML = """
<ul>
  <li class="border-b"><a href="/library/qwen3" class="group flex">
    <h2 title="qwen3"><span>qwen3</span></h2>
    <span class="inline-flex bg-[#ddf4ff] px-2 text-xs">27b</span>
    <span class="inline-flex bg-indigo-50 px-2 text-xs">vision</span>
    <span class="inline-flex bg-indigo-50 px-2 text-xs">tools</span>
    <p class="mt-1">A large language model.</p>
    <span title="39,604,549 downloads">39M</span>
  </a></li>
  <li class="border-b"><a href="/library/qwen3-coder" class="group">
    <h2 title="qwen3-coder"><span>qwen3-coder</span></h2>
    <span class="inline-flex bg-[#ddf4ff] px-2 text-xs">30b</span>
    <p class="mt-1">Coder model.</p>
    <span title="9,885,701 downloads">9.8M</span>
  </a></li>
  <li><a href="/library/broken">no fields at all</a></li>
</ul>
"""


def test_search_parser_reads_every_field():
    rows = S.parse_search(SEARCH_HTML)
    assert [r["name"] for r in rows] == ["qwen3", "qwen3-coder", "broken"]
    q = rows[0]
    assert q["params"] == "27b"
    assert q["capabilities"] == ["vision", "tools"]
    assert q["downloads"] == 39604549
    assert q["description"] == "A large language model."


def test_search_parser_survives_a_sparse_card():
    """A model with no badges must still be listed, just with empty fields."""
    rows = S.parse_search(SEARCH_HTML)
    sparse = rows[2]
    assert sparse["params"] == ""
    assert sparse["capabilities"] == []
    assert sparse["downloads"] == 0


def test_search_parser_deduplicates_repeated_names():
    doubled = SEARCH_HTML + SEARCH_HTML
    names = [r["name"] for r in S.parse_search(doubled)]
    assert len(names) == len(set(names))


def test_search_parser_rejects_a_path_traversal_link():
    html = '<li><a href="/library/../../etc/passwd">x</a></li>'
    assert S.parse_search(html) == []


def test_search_parser_on_empty_html():
    assert S.parse_search("") == []


def test_search_models_with_blank_query_short_circuits():
    """No network call should happen for an empty query."""
    assert S.search_models("   ") == []


# --- tags page parser ------------------------------------------------------

TAGS_HTML = """
<div class="group px-4 py-3">
  <a href="/library/qwen2.5:7b" class="md:hidden flex flex-col">
    <span><span class="font-mono">357c53fb659c</span> • 4.7GB • 32K context window •
    <span class="hidden sm:inline">Text input • 2 years ago</span></span>
  </a>
  <div class="hidden md:flex"><div class="grid grid-cols-12 items-center">
    <p class="col-span-2 text-neutral-500 text-[13px]">4.7GB</p>
    <p class="col-span-2 text-neutral-500 text-[13px]">32K</p>
    <div class="col-span-2 text-neutral-500 text-[13px]">Text</div>
  </div></div>
</div>
<div class="group px-4 py-3">
  <a href="/library/qwen2.5:0.5b" class="md:hidden flex flex-col">
    <span><span class="font-mono">aabbccdd1122</span> • 398MB • 32K context window •
    <span class="hidden sm:inline">Text input • 3 days ago</span></span>
  </a>
</div>
"""


def test_tags_parser_reads_every_field():
    rows = S.parse_tags(TAGS_HTML)
    by_ref = {r["ref"]: r for r in rows}
    seven = by_ref["qwen2.5:7b"]
    assert seven["size_bytes"] == 4_700_000_000
    assert seven["context"] == "32K"
    assert seven["modality"] == "text"
    assert seven["digest"] == "357c53fb659c"
    assert "2 years ago" in seven["updated"]


def test_tags_parser_survives_a_sparse_card():
    """A model with no badges must still be listed, just with empty fields."""
    rows = S.parse_search(SEARCH_HTML)
    sparse = rows[2]
    assert sparse["params"] == ""
    assert sparse["capabilities"] == []
    assert sparse["downloads"] == 0


def test_tags_are_sorted_smallest_first():
    """A 16 GB card wants the small quantization at the top of the list."""
    rows = S.parse_tags(TAGS_HTML)
    sizes = [r["size_bytes"] for r in rows]
    assert sizes == sorted(sizes)
    assert rows[0]["ref"] == "qwen2.5:0.5b"


def test_tags_parser_handles_mb_tb_and_b():
    html = (
        '<div class="group px-4 py-3"><a href="/library/m:t1">m:t1</a>'
        '<p class="col-span-2">900MB</p></div>'
        '<div class="group px-4 py-3"><a href="/library/m:t2">m:t2</a>'
        '<p class="col-span-2">1.5TB</p></div>'
        '<div class="group px-4 py-3"><a href="/library/m:t3">m:t3</a>'
        '<p class="col-span-2">512B</p></div>'
    )
    rows = {r["ref"]: r["size_bytes"] for r in S.parse_tags(html)}
    assert rows["m:t1"] == 900_000_000
    assert rows["m:t2"] == 1_500_000_000_000
    assert rows["m:t3"] == 512


def test_tags_parser_ignores_rows_without_a_size():
    html = ('<div class="group px-4 py-3"><a href="/library/m:tag">m:tag</a></div>')
    rows = S.parse_tags(html)
    assert rows and rows[0]["size_bytes"] == 0


# The live tags page renders each tag TWICE inside one row: a mobile block
# that folds size/context/modality into one bullet line, then a desktop grid
# with one fact per cell. Only reading the desktop layout silently drops the
# vision flag (and vice versa). The fixture below is a verbatim capture of two
# real rows from ollama.com/library/gemma3/tags — a vision tag and a text
# tag — so these tests cannot pass against markup the author imagined.
from pathlib import Path

_FIXTURE = Path(__file__).with_name("fixtures_gemma3_tags.html")


@pytest.fixture(scope="module")
def real_tag_rows():
    if not _FIXTURE.is_file():
        pytest.skip("captured gemma3 tags fixture is missing")
    return S.parse_tags(_FIXTURE.read_text(encoding="utf-8"))


def test_real_capture_parses_both_rows(real_tag_rows):
    assert {r["ref"] for r in real_tag_rows} == {"gemma3:270m", "gemma3:4b"}


def test_real_capture_fields_are_complete(real_tag_rows):
    """Every field must be populated from the live markup, not defaulted."""
    for row in real_tag_rows:
        assert row["size_bytes"] > 0, f"{row['ref']} lost its size"
        assert row["context"], f"{row['ref']} lost its context window"
        assert row["modality"] in ("text", "vision")
        assert row["digest"], f"{row['ref']} lost its digest"
        assert row["updated"], f"{row['ref']} lost its age"


def test_real_capture_vision_and_text_are_distinguished(real_tag_rows):
    by_ref = {r["ref"]: r for r in real_tag_rows}
    assert by_ref["gemma3:4b"]["modality"] == "vision"
    assert by_ref["gemma3:270m"]["modality"] == "text"


def test_real_capture_sizes_are_ordered_smallest_first(real_tag_rows):
    sizes = [r["size_bytes"] for r in real_tag_rows]
    assert sizes == sorted(sizes)


def test_real_capture_exact_values(real_tag_rows):
    """Pinned against the live page so a silent mis-parse cannot pass."""
    by_ref = {r["ref"]: r for r in real_tag_rows}
    assert by_ref["gemma3:4b"]["size_bytes"] == 3_400_000_000
    assert by_ref["gemma3:4b"]["context"] == "128K"
    assert by_ref["gemma3:270m"]["size_bytes"] == 292_000_000
    assert by_ref["gemma3:270m"]["context"] == "32K"


@requires_network
def test_live_vision_model_tags_are_flagged():
    """qwen3-vl is vision-only; if the parser loses the flag, every row
    reports 'text' and the user is told the wrong thing about the model."""
    rows = S.list_tags("qwen3-vl")
    assert rows
    assert all(r["modality"] == "vision" for r in rows), (
        f"only {sum(r['modality'] == 'vision' for r in rows)}/{len(rows)} rows flagged vision"
    )


@requires_network
def test_live_text_model_tags_are_not_flagged_vision():
    rows = S.list_tags("qwen2.5")
    assert rows
    assert not any(r["modality"] == "vision" for r in rows)


@requires_network
def test_live_tags_every_row_has_a_size():
    """A row with no size is useless for picking a quantization."""
    rows = S.list_tags("qwen2.5")
    missing = [r["ref"] for r in rows if not r["size_bytes"]]
    assert not missing, f"{len(missing)} rows had no size: {missing[:5]}"


def test_tags_parser_deduplicates():
    rows = S.parse_tags(TAGS_HTML + TAGS_HTML)
    refs = [r["ref"] for r in rows]
    assert len(refs) == len(set(refs))


def test_tags_parser_on_empty_html():
    assert S.parse_tags("") == []


def test_list_tags_rejects_a_namespaced_or_tagged_name():
    """list_tags builds the URL itself, so it must reject embedded path parts."""
    for bad in ["ns/model", "model:tag", "../../etc", "a/b/c"]:
        with pytest.raises(ValueError):
            S.list_tags(bad)


# --- cancellation ----------------------------------------------------------

def test_token_starts_uncancelled():
    assert CancelToken().cancelled is False


def test_cancel_sets_the_flag():
    t = CancelToken()
    t.cancel()
    assert t.cancelled is True


def test_raise_if_cancelled_only_fires_after_cancel():
    t = CancelToken()
    t.raise_if_cancelled()          # must not raise
    t.cancel()
    with pytest.raises(PullCancelled):
        t.raise_if_cancelled()


def test_cancel_is_idempotent():
    t = CancelToken()
    t.cancel()
    t.cancel()
    assert t.cancelled is True


def test_cancel_kills_a_bound_process():
    async def scenario():
        proc = await asyncio.create_subprocess_exec(
            "cmd", "/c", "timeout", "5",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        t = CancelToken()
        t.bind_process(proc)
        t.cancel()
        code = await asyncio.wait_for(proc.wait(), timeout=5)
        return code
    code = asyncio.run(scenario())
    assert code != 0, "terminate should have killed the process"


def test_binding_a_process_after_cancel_kills_it_immediately():
    """A cancel that lands before the subprocess exists must not be lost."""
    async def scenario():
        t = CancelToken()
        t.cancel()
        proc = await asyncio.create_subprocess_exec(
            "cmd", "/c", "timeout", "5",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        t.bind_process(proc)
        return await asyncio.wait_for(proc.wait(), timeout=5)
    assert asyncio.run(scenario()) != 0


def test_bind_process_none_is_safe():
    CancelToken().bind_process(None)


def test_cli_pull_reports_cancelled_when_token_is_pre_cancelled():
    """A pre-cancelled token must not start downloading anything."""
    async def scenario():
        t = CancelToken()
        t.cancel()
        events = []
        async for ev in pull_via_cli("some/model:tag", t):
            events.append(ev)
            if ev.get("status") in ("cancelled", "completed", "error"):
                break
        return events
    events = asyncio.run(scenario())
    assert events[-1]["status"] == "cancelled"


def test_pull_functions_accept_a_token_argument():
    """Both generators must be cancellable; a signature drift would silently
    drop the token at the call site."""
    for fn in (ollama_pull.pull_with_progress, pull_via_cli):
        params = inspect.signature(fn).parameters
        assert "token" in params, f"{fn.__name__} lost its token parameter"
        assert params["token"].default is None


def test_partial_blob_removal_is_forgiving():
    from lmloader.infra.ollama_pull import _discard_partial
    from pathlib import Path
    missing = Path("definitely/not/here.part")
    _discard_partial(missing)          # must not raise

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.part"
        p.write_bytes(b"123")
        _discard_partial(p)
        assert not p.exists()


# --- route wiring ----------------------------------------------------------

def test_pull_route_validates_the_model_name():
    """The route must reject junk before it reaches the registry URL builder."""
    from lmloader.api.routes.ollama import pull_ollama_model
    src = inspect.getsource(pull_ollama_model)
    assert "is_valid_spec" in src
    assert "status_code=400" in src


def test_cancel_route_is_idempotent():
    """Cancelling an idle pull must answer 200 with cancelled=False, not 404."""
    from lmloader.api.routes.ollama import cancel_pull
    result = asyncio.run(cancel_pull(name="nothing:here", mode="builtin"))
    assert result["cancelled"] is False
    assert "no active pull" in result["reason"]


def test_pull_key_is_case_insensitive():
    from lmloader.api.routes.ollama import _pull_key
    assert _pull_key("Qwen3:8B", "builtin") == _pull_key("qwen3:8b", "builtin")
    assert _pull_key("m", "cli") != _pull_key("m", "builtin")


# --- live network (skipped offline) ---------------------------------------

@requires_network
def test_live_search_returns_models():
    rows = S.search_models("qwen3")
    assert rows, "live ollama.com search returned nothing"
    assert any(r["name"].startswith("qwen3") for r in rows)


@requires_network
def test_live_tags_are_sorted_and_sized():
    rows = S.list_tags("qwen2.5")
    assert rows
    assert all(r["size_bytes"] > 0 for r in rows)
    assert [r["size_bytes"] for r in rows] == sorted(r["size_bytes"] for r in rows)