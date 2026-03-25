"""
deploy_week_figures.py — Per-week Figure Deployer

Copies all figures referenced in a week's raw content into the Hugo
static/images/weekNN/ directory and writes sidecar .txt caption files
with course-sequential numbering.

Uses the same mapping logic as figure_mapping.py:
  • Figures are discovered by scanning the prose for "Figure X-Y" references
    (in order of first appearance), rather than relying on a CHAPTER_FIGURES
    header.
  • The course label is assigned sequentially: the 1st unique figure becomes
    Figure W-1, the 2nd becomes Figure W-2, etc.

Usage:
    python scripts/deploy_week_figures.py <week_number>

Example:
    python scripts/deploy_week_figures.py 6

Output:
    Copies <idx>.png and writes <idx>.txt for every figure into
    llm-course/static/images/weekNN/
"""

import os
import re
import json
import shutil
import sys

# ── reuse the core functions from figure_mapping.py ──────────────────────────
# We import directly so the logic stays in one place.
sys.path.insert(0, os.path.dirname(__file__))
from figure_mapping import (
    load_figure_map,
    build_mapping_table,
)


def build_figures_path_index(figures_dir: str) -> dict[str, str]:
    """Walk source-assets/figures/ and return { "68.png": "/abs/path/68.png" }."""
    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(figures_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                index[fname] = os.path.join(root, fname)
    return index


def deploy_week(week_num: int) -> None:
    # ── resolve paths ─────────────────────────────────────────────────────────
    # scripts/ lives inside llm-course/, so base_dir is the repo root
    # (one level above llm-course/).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts/ lives inside llm-course/; base_dir == llm-course/
    # (matches figure_mapping.py's own base_dir convention)
    base_dir = os.path.dirname(script_dir)

    raw_path = os.path.join(
        base_dir, "source-assets", "raw-content", f"week-{week_num}.md"
    )
    figures_dir = os.path.join(base_dir, "source-assets", "figures")
    week_str = f"week{week_num:02d}"
    dest_dir = os.path.join(base_dir, "static", "images", week_str)

    # ── sanity checks ─────────────────────────────────────────────────────────
    if not os.path.exists(raw_path):
        print(f"ERROR: Raw content file not found: {raw_path}")
        sys.exit(1)

    # ── load inputs ───────────────────────────────────────────────────────────
    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read()

    figure_map = load_figure_map(base_dir)
    table = build_mapping_table(week_num, content, figure_map)

    if not table:
        print(f"No figure references found in week-{week_num}.md — nothing to deploy.")
        return

    figures_index = build_figures_path_index(figures_dir)

    # ── deploy ────────────────────────────────────────────────────────────────
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    skipped = 0

    print(f"\nDeploying figures for Week {week_num:02d} → {dest_dir}\n")
    print(f"  {'#':<4} {'Image':<12} {'Course Label':<14} {'Status'}")
    print(f"  {'-'*4} {'-'*12} {'-'*14} {'-'*30}")

    for row in table:
        img_file = row["image_file"]
        course_label = row["course_label"]   # e.g. "Figure 6-3"
        counter = row["counter"]

        # ── copy PNG ──────────────────────────────────────────────────────────
        src_png = figures_index.get(img_file)
        dest_png = os.path.join(dest_dir, img_file)

        if src_png and os.path.exists(src_png):
            shutil.copy2(src_png, dest_png)
            status = "copied"
            copied += 1
        else:
            status = "MISSING SOURCE"
            skipped += 1

        # ── write sidecar .txt with updated caption ───────────────────────────
        # Replace the book "Figure X-Y." prefix with the course label.
        raw_caption = figure_map.get(img_file, "")
        updated_caption = re.sub(
            r"^Figure\s+\d+-\d+\.",
            f"{course_label}.",
            raw_caption,
        )
        txt_name = img_file.replace(".png", ".txt")
        dest_txt = os.path.join(dest_dir, txt_name)
        with open(dest_txt, "w", encoding="utf-8") as f:
            f.write(updated_caption)

        print(f"  {counter:<4} {img_file:<12} {course_label:<14} {status}")

    print(f"\n  ✓ {copied} figure(s) copied, {skipped} missing.")
    if skipped:
        print(
            "  ⚠ Missing figures were not copied but their .txt sidecar files "
            "were still written."
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/deploy_week_figures.py <week_number>")
        print("Example: python scripts/deploy_week_figures.py 6")
        sys.exit(1)

    week_num = int(sys.argv[1])
    deploy_week(week_num)


if __name__ == "__main__":
    main()
