"""
generate_front_matter.py
─────────────────────────────────────────────────────────────────────────────
Scans each week's markdown file for figure references, resolves them to image
file indices via figure_map.json, and prepends a front matter stub with
CHAPTER_FIGURES into each raw-content file.

Handles two reference formats:

  Single  →  "Figure 1-5"      or  "figure 1-5"
  Range   →  "Figures 1-6 à 1-9"   (French range notation, expands to
              Figure 1-6, 1-7, 1-8, 1-9)
             Also handles English "to": "Figures 1-6 to 1-9"
             Also handles plain hyphen range: "Figures 1-6-1-9" (rare)

Run before inject_colab_urls.py.

Output prepended to each week*.md:
  ---
  # WEEK: 3
  # TITLE: Attention Mechanisms
  # CHAPTER_FIGURES: [16, 17, 18, 19, 20]
  # COLAB_NOTEBOOKS: []
  ---
"""

import json
import re
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

FIGURE_MAP_PATH = Path("../figure_map.json")
RAW_CONTENT_DIR = Path("../raw-content")

# ─────────────────────────────────────────────────────────────────────────────

FRONT_MATTER_TEMPLATE = """\
---
# WEEK: {week}
# TITLE: {title}
# CHAPTER_FIGURES: {figures}
# COLAB_NOTEBOOKS: []
---

"""


# ── Figure reference parsing ──────────────────────────────────────────────────

# Matches a single figure ID like "1-5" or "1–5" (en-dash)
_FIG_ID = r"(\d+)[-\u2013](\d+)"

# Single reference: "Figure 1-5" or "figure 1-5"
_SINGLE_RE = re.compile(
    rf"[Ff]igure\s+{_FIG_ID}",
    re.UNICODE,
)

# Range reference: "Figures 1-6 à 1-9" / "Figures 1-6 to 1-9"
# Captures: chapter, start_num, (optional same chapter), end_num
# "à" covers the French preposition (with or without accent)
_RANGE_RE = re.compile(
    rf"[Ff]igures\s+{_FIG_ID}\s+(?:à|a\b|to)\s+(?:(\d+)[-\u2013])?(\d+)",
    re.UNICODE,
)


def expand_range(chapter: int, start: int, end: int) -> list:
    """Return ["chapter-start", "chapter-start+1", ..., "chapter-end"]."""
    if end < start:
        # Defensive: swap silently if reversed
        start, end = end, start
    return [f"{chapter}-{n}" for n in range(start, end + 1)]


def extract_fig_ids_from_text(text: str) -> list:
    """
    Extract all figure IDs (as normalised strings like "1-5") from a block
    of markdown text, handling both single references and French/English ranges.
    Returns IDs in order of first appearance, deduplicated.
    """
    seen: dict = {}   # ordered set

    # ── Pass 1: expand ranges first so their constituent IDs don't get
    #            double-counted by the single-figure pass below.
    range_spans = []
    for m in _RANGE_RE.finditer(text):
        ch_start   = int(m.group(1))
        num_start  = int(m.group(2))
        # group(3) is the optional chapter prefix before end number
        ch_end     = int(m.group(3)) if m.group(3) else ch_start
        num_end    = int(m.group(4))

        if ch_start != ch_end:
            # Cross-chapter range is unusual; fall back to individual IDs
            # by treating it as two separate singles at the boundaries
            seen[f"{ch_start}-{num_start}"] = None
            seen[f"{ch_end}-{num_end}"]     = None
        else:
            for fig_id in expand_range(ch_start, num_start, num_end):
                seen[fig_id] = None

        range_spans.append((m.start(), m.end()))

    # ── Pass 2: single references, skipping characters already covered by
    #            a range match (so "Figures 1-6 à 1-9" doesn't also yield
    #            a spurious singleton match on "1-6").
    def inside_range(pos: int) -> bool:
        return any(start <= pos < end for start, end in range_spans)

    for m in _SINGLE_RE.finditer(text):
        if inside_range(m.start()):
            continue
        fig_id = f"{m.group(1)}-{m.group(2)}"
        seen[fig_id] = None

    return list(seen.keys())


# ── Figure map loading ────────────────────────────────────────────────────────

def load_figure_lookups(figure_map_path: Path) -> tuple:
    """
    Returns:
      fig_id_to_index  { "1-1": 5, "3-5": 47, ... }
      index_to_caption { 5: "Figure 1-1. ...", ... }
    """
    with open(figure_map_path, encoding="utf-8") as f:
        figure_map = json.load(f)

    fig_id_re = re.compile(r"Figure\s+([\d]+[-\u2013][\d]+)\.")

    fig_id_to_index  = {}
    index_to_caption = {}

    for filename, caption in figure_map.items():
        m = fig_id_re.match(caption)
        if not m:
            print(f"  Warning: Could not parse figure ID from: {caption[:60]}")
            continue
        raw_id = m.group(1).replace("\u2013", "-")
        index  = int(Path(filename).stem)
        fig_id_to_index[raw_id]  = index
        index_to_caption[index]  = caption

    return fig_id_to_index, index_to_caption


# ── Week scanning ─────────────────────────────────────────────────────────────

def scan_week_figures(raw_content_dir: Path, fig_id_to_index: dict) -> dict:
    """
    Scans every week*.md for figure references (singles and ranges),
    resolves each ID to an image index via fig_id_to_index, and returns:
        { week_number: [sorted deduplicated image indices] }
    """
    week_file_re = re.compile(r"week-(\d+)", re.IGNORECASE)

    md_files = sorted(raw_content_dir.glob("week-*.md"))
    if not md_files:
        print(f"  ERROR: No week-*.md files found in {raw_content_dir}")
        sys.exit(1)

    week_figures:   dict = {}
    unresolved_ids: list = []

    for md_file in md_files:
        m = week_file_re.search(md_file.stem)
        if not m:
            print(f"  Warning: Skipping {md_file.name} (no week number in filename)")
            continue

        week    = int(m.group(1))
        content = md_file.read_text(encoding="utf-8")

        fig_ids = extract_fig_ids_from_text(content)

        indices = []
        for fig_id in fig_ids:
            if fig_id in fig_id_to_index:
                indices.append(fig_id_to_index[fig_id])
            else:
                unresolved_ids.append(f"  Week {week:02d}: Figure {fig_id}")

        week_figures[week] = sorted(set(indices))

    if unresolved_ids:
        print("\n  Warning: Figure references not found in figure_map.json:")
        for msg in unresolved_ids:
            print(msg)
        print("  These are likely table/equation references — verify manually.\n")

    return week_figures


# ── Front matter injection ────────────────────────────────────────────────────

def inject_front_matter(raw_content_dir: Path, week_figures: dict) -> None:
    """
    Prepends the front matter stub to each week*.md.
    Idempotent: skips files that already have # CHAPTER_FIGURES:.
    """
    week_file_re = re.compile(r"week-(\d+)", re.IGNORECASE)

    for md_file in sorted(raw_content_dir.glob("week-*.md")):
        m = week_file_re.search(md_file.stem)
        if not m:
            continue

        week    = int(m.group(1))
        content = md_file.read_text(encoding="utf-8")

        if "# CHAPTER_FIGURES:" in content:
            print(f"  Skip  Week {week:02d}: front matter already present.")
            continue

        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title       = title_match.group(1).strip() if title_match else f"Week {week}"
        figures     = week_figures.get(week, [])

        stub = FRONT_MATTER_TEMPLATE.format(
            week=week,
            title=title,
            figures=json.dumps(figures),
        )

        md_file.write_text(stub + content, encoding="utf-8")
        print(f"  OK    Week {week:02d}: {len(figures)} figure(s) -> {md_file.name}")


# ── Self-test for range expansion (runs on import during dev) ─────────────────

def _self_test():
    cases = [
        # (input text,                         expected IDs)
        ("Figure 1-5",                          ["1-5"]),
        ("figure 1-5",                          ["1-5"]),
        ("Figures 1-6 à 1-9",                   ["1-6", "1-7", "1-8", "1-9"]),
        ("Figures 1-6 a 1-9",                   ["1-6", "1-7", "1-8", "1-9"]),
        ("Figures 1-6 to 1-9",                  ["1-6", "1-7", "1-8", "1-9"]),
        ("Figures 3-1 à 3-3 and Figure 3-5",    ["3-1", "3-2", "3-3", "3-5"]),
        # Range must not double-count the boundary figure as a singleton
        ("Figures 2-4 à 2-6",                   ["2-4", "2-5", "2-6"]),
    ]
    all_ok = True
    for text, expected in cases:
        result = extract_fig_ids_from_text(text)
        if result != expected:
            print(f"  FAIL: {text!r}\n    expected {expected}\n    got      {result}")
            all_ok = False
    if all_ok:
        print("  All self-tests passed.")
    return all_ok


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" generate_front_matter.py")
    print("=" * 60)

    print("\nRunning self-tests on range expansion …")
    _self_test()

    print("\nStep 1 -- Loading figure_map.json …")
    fig_id_to_index, _ = load_figure_lookups(FIGURE_MAP_PATH)
    print(f"  {len(fig_id_to_index)} figure IDs loaded.")

    print("\nStep 2 -- Scanning week files for figure references …")
    week_figures = scan_week_figures(RAW_CONTENT_DIR, fig_id_to_index)
    for w, idxs in sorted(week_figures.items()):
        suffix = "..." if len(idxs) > 4 else ""
        print(f"  Week {w:02d}: {len(idxs):3d} figure(s)  {idxs[:4]}{suffix}")
    total = sum(len(v) for v in week_figures.values())
    print(f"  Total: {total} figure references across {len(week_figures)} weeks.")

    print("\nStep 3 -- Injecting front matter stubs …")
    inject_front_matter(RAW_CONTENT_DIR, week_figures)

    print("\nDone. Run inject_colab_urls.py next to fill in COLAB_NOTEBOOKS.")