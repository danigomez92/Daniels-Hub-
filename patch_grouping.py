#!/usr/bin/env python3
"""Fix the over-eager duplicate grouping in beatcut.py, in place."""
import re
import sys
from pathlib import Path

p = Path("beatcut.py")
s = p.read_text(encoding="utf-8")

if "by_bytes" in s:
    sys.exit("Already patched — nothing to do.")

start = s.find("    # Merge families:")
end = s.find("    save_cache(cache_path, cache)")
if start == -1 or end == -1 or end < start:
    sys.exit("Could not find the grouping block. Download the full file instead.")

NEW = '''    # Merge duplicates on two STRONG signals only: normalised filename, or
    # identical byte size AND duration. Duration+resolution alone collapses
    # unrelated AI-generated clips that happen to share a length.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_name, by_bytes = {}, {}
    for src in sources:
        node = str(src.path.resolve())
        find(node)
        stem = src.path.stem.lower()
        stem = re.sub(r"[*]+", " ", stem)
        stem = re.sub(r"\\s*\\((?:\\d+)\\)\\s*$", " ", stem)
        stem = re.sub(r"\\b(copy|duplicate|dup|kopie)\\b", " ", stem)
        stem = re.sub(r"\\s+", " ", stem).strip() or src.path.stem.lower()
        if stem in by_name:
            union(node, by_name[stem])
        else:
            by_name[stem] = node
        try:
            shape = (src.path.stat().st_size, round(src.duration, 2))
        except OSError:
            continue
        if shape in by_bytes:
            union(node, by_bytes[shape])
        else:
            by_bytes[shape] = node

    for src in sources:
        src.family = find(str(src.path.resolve()))

    groups = {}
    for src in sources:
        groups.setdefault(src.family, []).append(src.path.name)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if dupes:
        print(f"  grouped {sum(len(v) for v in dupes.values())} duplicate files into "
              f"{len(dupes)} set(s); each set shares one cooldown:")
        for names in list(dupes.values())[:8]:
            print(f"    {', '.join(n[:44] for n in names)}")

'''

s = s[:start] + NEW + s[end:]
if "\nimport re\n" not in s:
    s = s.replace("\nimport random\n", "\nimport random\nimport re\n", 1)

p.write_text(s, encoding="utf-8")
print("Patched. Verifying...")
import py_compile
py_compile.compile("beatcut.py", doraise=True)
print("Syntax OK — re-run your render.")