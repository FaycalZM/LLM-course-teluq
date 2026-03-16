import os
import re
import json
import shutil

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_content_dir = os.path.join(base_dir, 'source-assets', 'raw-content')
    figures_dir = os.path.join(base_dir, 'source-assets', 'figures')
    figure_map_path = os.path.join(base_dir, 'source-assets', 'figure_map.json')
    static_images_dir = os.path.join(base_dir, 'llm-course', 'static', 'images')

    with open(figure_map_path, 'r', encoding='utf-8') as f:
        figure_map = json.load(f)


    figures_map_paths = {}
    for root, dirs, files in os.walk(figures_dir):
        for file in files:
            if file.endswith('.png'):
                figures_map_paths[file] = os.path.join(root, file)

    summary: list[str] = []

    for filename in sorted(os.listdir(raw_content_dir)):
        if filename.startswith('week-') and filename.endswith('.md'):
            # Extract week number
            week_num_match = re.search(r'week-(\d+)\.md', filename)
            if not week_num_match:
                continue
            week_num = week_num_match.group(1)
            week_str = f"week{int(week_num):02d}"

            filepath = os.path.join(raw_content_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find CHAPTER_FIGURES
            match = re.search(r'# CHAPTER_FIGURES:\s*\[(.*?)\]', content)
            if match:
                indices_str = match.group(1)
                if not indices_str.strip():
                    indices = []
                else:
                    indices = [idx.strip() for idx in indices_str.split(',')]
                
                week_dest_dir = os.path.join(static_images_dir, week_str)
                os.makedirs(week_dest_dir, exist_ok=True)
                
                copied_count: int = 0
                for idx in indices:
                    png_filename = f"{idx}.png"
                    src_png = figures_map_paths.get(png_filename)
                    dest_png = os.path.join(week_dest_dir, png_filename)
                    
                    if src_png and os.path.exists(src_png):
                        shutil.copy2(src_png, dest_png)
                        copied_count = copied_count + 1  # type: ignore
                    else:
                        print(f"Warning: Figure {png_filename} not found.")
                    
                    caption = figure_map.get(png_filename, "")
                    dest_txt = os.path.join(week_dest_dir, f"{idx}.txt")
                    with open(dest_txt, 'w', encoding='utf-8') as txt_f:
                        txt_f.write(caption)
                
                summary.append(f"Week {int(week_num):02d}: {copied_count} figures copied")
            
    print(" | ".join(summary))

if __name__ == "__main__":
    main()
