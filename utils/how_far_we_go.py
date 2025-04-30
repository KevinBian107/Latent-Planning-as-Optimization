import os
import sys
from collections import defaultdict


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())


def count_code_lines(root_dir, include_exts=None):
    folder_counts = defaultdict(lambda: defaultdict(int))
    type_counts = defaultdict(int)
    total_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            if include_exts and ext not in include_exts:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line for line in f if line.strip()]
                    count = len(lines)
            except (UnicodeDecodeError, FileNotFoundError):
                continue

            rel_dir = os.path.relpath(dirpath, root_dir)
            folder_counts[rel_dir][ext] += count
            type_counts[ext] += count
            total_count += count

    return folder_counts, type_counts, total_count


def print_side_by_side(folder_counts, type_counts, total_count):

    folder_table = []
    for folder, types in sorted(folder_counts.items()):
        folder_total = sum(types.values())
        folder_table.append((folder, folder_total))


    type_table = []
    for ext, count in sorted(type_counts.items(), key=lambda x: (-x[1], x[0])):
        type_table.append((ext or "no extension", count))


    max_rows = max(len(folder_table), len(type_table))
    folder_table += [("", "")] * (max_rows - len(folder_table))
    type_table += [("", "")] * (max_rows - len(type_table))


    left_title = "Folder".ljust(40) + "Lines"
    right_title = "File Type".ljust(20) + "Lines"
    print(f"{left_title} | {right_title}")
    print("=" * 40 + " | " + "=" * 20)


    for (folder, f_lines), (ext, e_lines) in zip(folder_table, type_table):
        folder_part = f"{folder:<40} {f_lines}".ljust(45)
        type_part = f"{ext:<20} {e_lines}"
        print(f"{folder_part} | {type_part}")

    
    print("\n" + "=" * 70)
    print(f"Total lines in project: {total_count}")
    if total_count < 1000:
        print("🐣 Still a baby project. Keep building!")
    elif total_count < 2500:
        print("🌱 Growing strong! Let's go!")
    elif total_count < 5000:
        print("🚀 Nice! You're crafting a real codebase!")
    elif total_count < 10000:
        print("🏔️ Climbing to the top... impressive!")
    else:
        print("🌌 You have created a universe of code. Respect!")
    print("=" * 70)

if __name__ == "__main__":
    root_dir = os.getcwd()  
    include_exts = {".py", ".cpp", ".c", ".h", ".js", ".ts", ".html", ".css", ".java", ".rs"}  
    folder_counts, type_counts, total_count = count_code_lines(root_dir, include_exts)
    print_side_by_side(folder_counts, type_counts, total_count)

