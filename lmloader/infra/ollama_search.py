"""Browse the Ollama model library (search models, list their tags).

Ollama publishes no public JSON search API for the library — the registry
speaks Docker Registry v2 only and answers `GET /v2/<repo>/tags/list` with 404.
What *is* available is ollama.com's server-rendered HTML, which carries the
same data the web UI renders:

  https://ollama.com/search?q=<query>          -> <li> per model
  https://ollama.com/library/<name>/tags       -> one block per tag

Both pages are parsed defensively: every field is optional and a missing
field degrades to "" rather than raising, because a markup tweak upstream
must not take the download page down. Only model names and tag names are
ever pulled out — they are re-validated against the spec grammar before
being handed to the puller.
"""

import html
import re
from typing import Optional

import httpx

from lmloader.config import settings
from lmloader.logger import logger

SITE = "https://ollama.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) lmloader/1.0"

# A model reference is '<namespace>/<model>[:<tag>]'. Namespace and model are
# lowercase alnum plus . _ - ; tags additionally allow those chars. Anything
# else is rejected before it can reach the registry URL builder. The 128-char
# segment cap is well above any real Ollama name and keeps a pasted blob of
# text from turning into a multi-kilobyte registry URL.
_SEGMENT = r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}"
SPEC_RE = re.compile(rf"^{_SEGMENT}(?:/{_SEGMENT})?(?::{_SEGMENT})?$")


def is_valid_spec(spec: str) -> bool:
    """True when `spec` is a safe model reference (no path/URL smuggling)."""
    return bool(spec) and bool(SPEC_RE.match(spec.strip()))


def _proxy() -> Optional[str]:
    return settings.proxy.strip() or None


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        proxy=_proxy(),
        headers={"User-Agent": UA, "Accept-Language": "en"},
    )


def _strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- search page -----------------------------------------------------------

# One model card: <li ...><a href="/library/NAME" ...> ... </a></li>
_CARD_RE = re.compile(
    r'<li[^>]*>\s*<a href="/library/(?P<name>[^"/?#]+)"(?P<body>.*?)</a>',
    re.S,
)
_DL_RE = re.compile(r'title="([\d,]+)\s+downloads"')
# The card carries two badge families that share one markup shape:
#   bg-[#ddf4ff]  -> the parameter-size badge ("27b")
#   bg-indigo-50  -> capability badges ("vision", "tools", "thinking")
# They differ only by background utility class, so both are captured and
# classified by which class they carry.
_BADGE_RE = re.compile(
    r'<span[^>]*class="[^"]*?(?P<kind>bg-\[#ddf4ff\]|bg-indigo-50)[^"]*"[^>]*>\s*(?P<text>[^<]+?)\s*</span>'
)
# The description is the only <p> inside the card.
_DESC_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)


def parse_search(html_text: str) -> list[dict]:
    """Extract model cards from the ollama.com search page.

    Returns [{name, params, capabilities, downloads, description}] in page
    order. `capabilities` is e.g. ["vision", "tools", "thinking"].
    """
    out: list[dict] = []
    seen: set[str] = set()
    for m in _CARD_RE.finditer(html_text):
        name = html.unescape(m.group("name")).strip()
        if not name or not is_valid_spec(name) or name in seen:
            continue
        seen.add(name)
        body = m.group("body")

        params = ""
        caps: list[str] = []
        for badge in _BADGE_RE.finditer(body):
            text = _collapse(_strip_tags(badge.group("text")))
            if not text:
                continue
            if badge.group("kind") == "bg-indigo-50":
                caps.append(text)
            elif not params:
                params = text

        downloads = 0
        dl = _DL_RE.search(body)
        if dl:
            downloads = int(dl.group(1).replace(",", ""))

        desc = ""
        d = _DESC_RE.search(body)
        if d:
            desc = _collapse(_strip_tags(d.group(1)))[:300]

        out.append({
            "name": name,
            "params": params,
            "capabilities": caps,
            "downloads": downloads,
            "description": desc,
        })
    return out


def search_models(query: str, limit: int = 30) -> list[dict]:
    """Search the Ollama library. Raises httpx errors on transport failure."""
    q = query.strip()
    if not q:
        return []
    with _client() as client:
        resp = client.get(f"{SITE}/search", params={"q": q})
        resp.raise_for_status()
    return parse_search(resp.text)[:limit]


# --- tags page -------------------------------------------------------------

# The live tags page renders each tag TWICE inside one row: a mobile block
# (md:hidden) that folds size/context/modality into a single bullet line,
# then a desktop grid (hidden md:flex) with one fact per cell. Both carry
# the same data, and the row body below spans both — so a regex written
# against only one layout silently loses fields on the other.
_ROW_OPEN = r'<div class="group px-4 py-3">'
_TAG_ROW_RE = re.compile(
    rf'{_ROW_OPEN}(?P<body>.*?)(?={_ROW_OPEN}|\Z)',
    re.S,
)
_TAG_LINK_RE = re.compile(r'href="/library/(?P<ref>[^"/?#:]+:[^"/?#]+)"')
# Mobile line: "<digest> • 4.7GB • 32K context window  • Text, Image input •"
_SIZE_INLINE_RE = re.compile(r'•\s*([\d.]+)\s*(B|KB|MB|GB|TB)\s*•')
_CONTEXT_INLINE_RE = re.compile(r'([\d.]+[KMB])\s+context window')
# Desktop grid cell: <p class="col-span-2 ...">4.7GB</p>
_SIZE_P_RE = re.compile(
    r'<p[^>]*class="col-span-2[^"]*"[^>]*>\s*([\d.]+)\s*(B|KB|MB|GB|TB)\s*</p>')
_CONTEXT_P_RE = re.compile(
    r'<p[^>]*class="col-span-2[^"]*"[^>]*>\s*([\d.]+[KMB])\s*</p>')
_MODALITY_RE = re.compile(r'(Text\s*,\s*Image|Image|Text)\s+input')
_DIGEST_RE = re.compile(r'<span class="font-mono[^"]*">\s*([0-9a-f]{6,})\s*</span>')
# Age text ("2 days ago", "11 months ago") trails the modality in a bullet
# line. It is rendered inside <span> on the mobile block and <div> on the
# desktop one, so the terminator cannot name a tag.
_AGE_RE = re.compile(
    r'•\s*((?:\d+\s*)?(?:second|minute|hour|day|week|month|year)s?\s+ago|just now|yesterday|today)'
)

_UNIT = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
         "K": 10**3, "M": 10**6}


def _parse_size(num: str, unit: str) -> int:
    return int(float(num) * _UNIT.get(unit, 1))


def parse_tags(html_text: str) -> list[dict]:
    """Extract tag rows from /library/<name>/tags.

    Returns [{ref, tag, size_bytes, context, modality, digest, updated}]
    sorted by ascending size — small quantized tags first is what a user
    with a 16 GB card actually wants to click.
    """
    out: list[dict] = []
    seen: set[str] = set()
    # Strip tags once so the inline (mobile) patterns can match across the
    # markup that separates the bullets.
    for row in _TAG_ROW_RE.finditer(html_text):
        body = row.group("body")
        link = _TAG_LINK_RE.search(body)
        if not link:
            continue
        ref = html.unescape(link.group("ref")).strip()
        if not is_valid_spec(ref) or ref in seen:
            continue
        seen.add(ref)

        flat = _collapse(_strip_tags(body))

        # Desktop grid cell first (unambiguous), then the mobile bullet line.
        size = 0
        s = _SIZE_P_RE.search(body) or _SIZE_INLINE_RE.search(flat)
        if s:
            size = _parse_size(s.group(1), s.group(2))

        context = ""
        c = _CONTEXT_P_RE.search(body) or _CONTEXT_INLINE_RE.search(flat)
        if c:
            context = _collapse(_strip_tags(c.group(1)))

        modality = ""
        mm = _MODALITY_RE.search(flat)
        if mm:
            modality = "vision" if "Image" in mm.group(1) else "text"

        digest = ""
        dg = _DIGEST_RE.search(body)
        if dg:
            digest = dg.group(1)

        updated = ""
        ag = _AGE_RE.search(body)
        if ag:
            updated = _collapse(_strip_tags(ag.group(1)))

        out.append({
            "ref": ref,
            "tag": ref.split(":", 1)[1],
            "size_bytes": size,
            "context": context,
            "modality": modality,
            "digest": digest,
            "updated": updated,
        })
    out.sort(key=lambda t: (t["size_bytes"] or 0, t["ref"]))
    return out


def list_tags(model: str, limit: int = 60) -> list[dict]:
    """List available tags for one model. `model` must be a bare model name."""
    name = model.strip()
    if "/" in name or ":" in name or not is_valid_spec(name):
        raise ValueError(f"invalid model name: {model}")
    with _client() as client:
        resp = client.get(f"{SITE}/library/{name}/tags")
        resp.raise_for_status()
    tags = parse_tags(resp.text)
    if not tags:
        logger.warning(f"no tags parsed from ollama.com/library/{name}/tags")
    return tags[:limit]


# --- already installed -----------------------------------------------------

def installed_refs() -> list[str]:
    """Model refs present in the local Ollama store (from manifests)."""
    from lmloader.infra.ollama_lib import ollama_root, _iter_manifests, _manifest_name

    root = ollama_root()
    if root is None:
        return []
    manifests_dir = root / "manifests"
    if not manifests_dir.is_dir():
        return []
    refs: list[str] = []
    for mp in _iter_manifests(root):
        try:
            refs.append(_manifest_name(mp, manifests_dir))
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(refs))