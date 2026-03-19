import os, re

# Chronological assignment: image_file -> course_figure_label
mapping = {
    "68.png": "3-1",
    "69.png": "3-2",
    "70.png": "3-3",
    "71.png": "3-4",
    "72.png": "3-5",
    "73.png": "3-6",
    "74.png": "3-7",
    "79.png": "3-8",
    "78.png": "3-9",
    "85.png": "3-10",
    "86.png": "3-11",
    "84.png": "3-12",
    "82.png": "3-13",
    "83.png": "3-14",
    "66.png": "3-15",
    "80.png": "3-16",
    "75.png": "3-17",
    "76.png": "3-18",
    "57.png": "3-19",
    "58.png": "3-20",
    "59.png": "3-21",
    "63.png": "3-22",
    # Non-referenced but deployed — reset to original caption label
    "25.png": "1-21",
    "61.png": "3-8",
    "62.png": "3-9",
}

images_dir = r"c:\Projects\LLM-Course\llm-course\static\images\week03"

for fname, new_num in mapping.items():
    txt_path = os.path.join(images_dir, fname.replace(".png", ".txt"))
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = re.sub(r"^Figure \d+-\d+\.", f"Figure {new_num}.", content, count=1)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {fname} -> Figure {new_num}.")
    else:
        print(f"MISSING: {txt_path}")
