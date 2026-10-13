"""Repetition-loop detector for streamed model output.

llama-server's sampling defaults leave the repetition channels off
(``repeat_penalty=1.0``, ``dry_multiplier=0.0``), so a model can fall into a
degenerate cycle — typically visible as the same short phrase repeating dozens
of times inside the reasoning block. Left alone it streams until max_tokens and
burns the GPU for nothing.

This module is a pure function of the text seen so far, so it can be unit
tested without a model, a socket or a Qt event loop. The GUI calls it while
streaming and stops the request once it trips.

Detection is deliberately conservative — a false positive truncates a perfectly
good answer, so the bar is "the same short fragment, many times over", not
"the same word appears again".
"""

from __future__ import annotations

# Fragment length in characters, sized to roughly ONE markdown table row / one
# full sentence. This is the key to telling a degenerate loop apart from a dense
# table: in a table the repeated parts are short column values while the whole
# row differs (serial number / spec / figures change), so a ~48-char span is
# NOT repeated; in a degenerate loop the SAME whole sentence repeats verbatim,
# so the long span repeats every cycle. (16-char spans false-tripped on tables:
# a column value such as " | dense busbar | 2000" recurred 40 times.)
FRAGMENT_LEN = 48

# How many times the same whole-row/sentence fragment must appear before we call
# it a loop. Coherent output — including tables, lists, code and refrains — does
# not repeat a 48-char span eight times; a degenerate loop repeats it dozens.
REPEAT_THRESHOLD = 8

# Only inspect the tail of the stream. The detector runs on every token, so the
# scan window is capped to keep it O(window) rather than O(whole answer) — a
# 32k-token answer would otherwise be rescanned in full every single token.
WINDOW_CHARS = 4000

# Never trip before the model has produced at least this much text: early output
# is short and legitimately repetitive (restating the question, "好的", ...).
MIN_SCAN_CHARS = 400


def _top_fragment_count(text: str) -> tuple[str, int]:
    """Return (fragment, occurrences) for the most repeated FRAGMENT_LEN span."""
    from collections import Counter

    if len(text) < FRAGMENT_LEN:
        return "", 0
    counts = Counter(text[i:i + FRAGMENT_LEN] for i in range(len(text) - FRAGMENT_LEN + 1))
    fragment, n = counts.most_common(1)[0]
    return fragment, n


def detect_repetition_loop(text: str) -> dict | None:
    """Return a diagnostic dict when *text* looks like a repetition loop.

    Returns ``None`` for healthy output, otherwise a dict with the offending
    fragment, its occurrence count and how many characters were scanned — enough
    for the UI to tell the user what happened without dumping the whole answer.
    """
    if len(text) < MIN_SCAN_CHARS:
        return None

    window = text[-WINDOW_CHARS:]
    fragment, count = _top_fragment_count(window)
    if count < REPEAT_THRESHOLD:
        return None

    return {
        "reason": "repetition_loop",
        "fragment": fragment,
        "occurrences": count,
        "scanned_chars": len(window),
    }
