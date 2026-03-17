"""
extract_figure_captions.py
──────────────────────────
Extracts figure captions from "Hands-On Large Language Models" (O'Reilly PDF)
and maps each extracted image (5.png … 289.png) to its caption.

Strategy
────────
Raw-text regex approaches fail because "Figure X-Y." appears both as:
  • A real caption  → starts its own layout line, rendered in ITALIC font
  • An inline ref   → embedded mid-paragraph, rendered in ROMAN font

Fix: use pdfplumber's word-level data to:
  1. Group words into lines by y-coordinate.
  2. Match "Figure X-Y." only when it STARTS a line.
  3. Accept the line only when its first word is in an italic font.
  4. Capture the DOMINANT FONT(S) of that opening caption line.
  5. Collect wrapped continuation lines only while the majority of their
     words share those same caption fonts — stops the moment the font
     switches to body text.
     Using font identity (not a generic italic-ratio) is robust because:
       • a caption's wrapped lines always use the exact same italic font
       • the switch from caption to body text is an abrupt, complete
         font change, not a gradual one
       • numbers / proper nouns in a caption still use the caption font
         (that was the failure mode of the previous ratio approach)
  6. Deduplicate by figure ID (first occurrence in reading order wins).
"""

import re
import json
from collections import defaultdict, Counter
import pdfplumber  # pip install pdfplumber

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH          = r"C:\Users\hello\Documents\Personal\hands-on LLMs.pdf"
START_INDEX       = 5
TOTAL_FIGURES     = 284
OUTPUT_JSON       = "figure_map.json"
Y_TOLERANCE       = 2    # px — tolerance for grouping words into lines
MAX_CAPTION_LINES = 8    # hard ceiling on wrapped continuation lines
# ──────────────────────────────────────────────────────────────────────────────

CAPTION_START = re.compile(r'^Figure\s+(\d+[-\u2013]\d+)\.\s+(.*)', re.UNICODE)


# ── Font helpers ──────────────────────────────────────────────────────────────

def base_font(fontname: str) -> str:
    """
    Strip the random PDF embedding prefix (e.g. 'ABCDEF+') so that
    'ABCDEF+ItalicMT' and 'XYZUVW+ItalicMT' both normalise to 'ItalicMT'.
    This lets us match fonts across pages even when the prefix changes.
    """
    return fontname.split("+")[-1]


def is_italic(fontname: str) -> bool:
    lower = base_font(fontname).lower()
    return any(tag in lower for tag in ("italic", "oblique", "slant", "it-"))


def dominant_fonts(words: list) -> set:
    """
    Return the set of base font names that together account for the
    majority (>= 50%) of words on this line.

    For a pure caption line every word uses the same italic font, so this
    returns a singleton set.  The '>= 50%' threshold gracefully handles
    the rare case where a caption mixes two closely related italic variants
    (e.g. ItalicMT and BoldItalicMT for an emphasised term).
    """
    if not words:
        return set()
    counts = Counter(base_font(w["fontname"]) for w in words)
    total  = len(words)
    return {font for font, n in counts.items() if n / total >= 0.5}


def line_matches_fonts(words: list, caption_fonts: set, threshold: float = 0.6) -> bool:
    """
    Return True if at least `threshold` fraction of words on this line
    use one of the fonts in `caption_fonts`.

    threshold=0.6 means a line must be at least 60% caption-font words.
    This tolerates the occasional inline code token or page-number artifact
    while still blocking body-text lines (which are 0% caption-font).
    """
    if not words:
        return False
    matching = sum(1 for w in words if base_font(w["fontname"]) in caption_fonts)
    return (matching / len(words)) >= threshold


# ── Line helpers ──────────────────────────────────────────────────────────────

def normalise_fig_id(raw: str) -> str:
    return raw.replace("\u2013", "-").replace("\u2014", "-")


def page_lines(page) -> list:
    """Group page words into horizontal lines sorted top-to-bottom."""
    words = page.extract_words(extra_attrs=["fontname", "size", "top"])
    if not words:
        return []
    buckets = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / Y_TOLERANCE) * Y_TOLERANCE
        buckets[y_key].append(w)
    return [
        sorted(ws, key=lambda w: w["x0"])
        for _, ws in sorted(buckets.items())
    ]


def line_text(words: list) -> str:
    return " ".join(w["text"] for w in words)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_captions(pdf_path: str) -> dict:
    """
    Returns {normalised_fig_id: full_caption_text}, deduplicated.
    Example: {"1-1": "A peek into the history of Language AI."}
    """
    seen = {}

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            if page_num % 50 == 0:
                print(f"  … scanning page {page_num}/{total}")

            lines = page_lines(page)

            for i, words in enumerate(lines):
                txt = line_text(words)
                m = CAPTION_START.match(txt)
                if not m:
                    continue

                # ── Gate 1: opening line must begin with an italic word ───
                if not is_italic(words[0]["fontname"]):
                    continue

                fig_id  = normalise_fig_id(m.group(1))
                caption = m.group(2).strip()

                # ── Gate 2: deduplicate (first occurrence = real caption) ─
                if fig_id in seen:
                    continue

                # ── Capture the caption's font signature ──────────────────
                # Use the dominant fonts of the opening line as the reference.
                # All genuine continuation lines will share these same fonts.
                caption_fonts = dominant_fonts(words)

                # ── Collect wrapped continuation lines ────────────────────
                j = i + 1
                while j < len(lines) and j <= i + MAX_CAPTION_LINES:
                    next_words = lines[j]
                    next_txt   = line_text(next_words)

                    # Hard stops
                    if not next_txt.strip():
                        break
                    if CAPTION_START.match(next_txt):
                        break

                    # Font-identity check: does this line speak the same
                    # font as the caption? If not, we've crossed into body text.
                    if not line_matches_fonts(next_words, caption_fonts):
                        break

                    caption += " " + next_txt.strip()
                    j += 1

                seen[fig_id] = caption

    return seen


# ── Sorting / mapping ─────────────────────────────────────────────────────────

def sort_key(fig_id: str) -> tuple:
    chapter, num = fig_id.split("-", 1)
    return (int(chapter), int(num))


def build_map(captions: dict, start_index: int) -> dict:
    """Map sorted figure IDs to sequential filenames (5.png, 6.png, …)."""
    sorted_ids = sorted(captions.keys(), key=sort_key)
    return {
        f"{start_index + i}.png": f"Figure {fig_id}. {captions[fig_id]}"
        for i, fig_id in enumerate(sorted_ids)
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose(captions: dict, expected: int) -> None:
    found = len(captions)
    print(f"\n{'─'*55}")
    print(f"  Found    : {found} unique figure IDs")
    print(f"  Expected : {expected}")

    if found == expected:
        print("  ✓ Count matches perfectly.")
    elif found < expected:
        all_ids = sorted(captions.keys(), key=sort_key)
        prev    = None
        gaps    = []
        for fid in all_ids:
            ch, n = sort_key(fid)
            if prev and (ch, n) != (prev[0], prev[1] + 1) and ch == prev[0]:
                gaps.append(f"    gap after Figure {prev[0]}-{prev[1]}")
            prev = (ch, n)
        print(f"  ✗ Missing {expected - found} caption(s). Gaps detected:")
        for g in gaps[:10]:
            print(g)
        print("\n  Tip: flip FALLBACK_ROMAN = True to include roman-font opening lines.")
    else:
        print(f"  ✗ {found - expected} extra — duplicates slipped through.")

    print(f"{'─'*55}\n")


# ── Debug helper ──────────────────────────────────────────────────────────────

def debug_caption(pdf_path: str, target_fig_id: str) -> None:
    """
    Print word-by-word font information for the lines around a specific
    figure caption.  Use this when a caption is truncated or bleeding to
    understand exactly what pdfplumber sees.

    Usage:
        debug_caption("hands_on_llms.pdf", "1-1")
    """
    target = normalise_fig_id(target_fig_id)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            lines = page_lines(page)
            for i, words in enumerate(lines):
                txt = line_text(words)
                m = CAPTION_START.match(txt)
                if not m or normalise_fig_id(m.group(1)) != target:
                    continue

                print(f"\nFound Figure {target} on page {page_num}, line {i}")
                caption_fonts = dominant_fonts(words)
                print(f"Caption fonts: {caption_fonts}\n")

                # Print opening line
                print(f"  [LINE {i}] {txt}")
                for w in words:
                    print(f"    '{w['text']:30s}' font={base_font(w['fontname'])}")

                # Print up to MAX_CAPTION_LINES+2 following lines with decision
                for j in range(i + 1, min(i + MAX_CAPTION_LINES + 3, len(lines))):
                    nw  = lines[j]
                    nt  = line_text(nw)
                    ok  = line_matches_fonts(nw, caption_fonts)
                    tag = "INCLUDE" if ok else "STOP   "
                    print(f"\n  [LINE {j}] [{tag}] {nt}")
                    for w in nw:
                        matched = base_font(w["fontname"]) in caption_fonts
                        print(f"    {'✓' if matched else '✗'} '{w['text']:30s}' font={base_font(w['fontname'])}")
                return

    print(f"Figure {target_fig_id} not found in {pdf_path}.")


# ── Fallback ──────────────────────────────────────────────────────────────────

FALLBACK_ROMAN = True  # flip to True if count is too low

def extract_captions_fallback(pdf_path: str) -> dict:
    """
    Identical to extract_captions() but skips the italic gate on the
    opening line.  Font-identity check on continuation lines still applies.
    Use only when the main extractor misses figures.
    """
    seen = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page_lines(page)
            for i, words in enumerate(lines):
                txt = line_text(words)
                m = CAPTION_START.match(txt)
                if not m:
                    continue
                fig_id = normalise_fig_id(m.group(1))
                if fig_id in seen:
                    continue
                caption       = m.group(2).strip()
                caption_fonts = dominant_fonts(words)
                j = i + 1
                while j < len(lines) and j <= i + MAX_CAPTION_LINES:
                    nw = lines[j]
                    nt = line_text(nw)
                    if not nt.strip() or CAPTION_START.match(nt):
                        break
                    if not line_matches_fonts(nw, caption_fonts):
                        break
                    caption += " " + nt.strip()
                    j += 1
                seen[fig_id] = caption
    return seen


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Optional: debug a specific caption
    #   python extract_figure_captions.py debug 1-1
    if len(sys.argv) == 3 and sys.argv[1] == "debug":
        debug_caption(PDF_PATH, sys.argv[2])
        sys.exit(0)

    print(f"Scanning '{PDF_PATH}' for figure captions …\n")

    captions = (
        extract_captions_fallback(PDF_PATH)
        if FALLBACK_ROMAN
        else extract_captions(PDF_PATH)
    )

    diagnose(captions, TOTAL_FIGURES)

    figure_map = build_map(captions, START_INDEX)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(figure_map, f, indent=2, ensure_ascii=False)

    print(f"Saved → {OUTPUT_JSON}\n")
    print("First 8 entries:")
    for k, v in list(figure_map.items())[:8]:
        print(f"  {k:>8}  {v[:90]}")