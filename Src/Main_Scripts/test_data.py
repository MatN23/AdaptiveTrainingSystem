# Copyright (c) 2025 MatN23. All rights reserved.
import json
import glob
from collections import defaultdict
from pathlib import Path

counts = defaultdict(int)

# Handle .jsonl files
for filepath in glob.glob("datasets/**/*.jsonl", recursive=True):
    source = Path(filepath).stem
    with open(filepath) as f:
        for line in f:
            try:
                d = json.loads(line)
                text = d.get("text", "") or d.get("content", "") or d.get("messages", "")
                if isinstance(text, list):
                    text = " ".join([m.get("content", "") for m in text])
                counts[source] += len(text.split())
            except:
                continue

# Handle .txt files
for filepath in glob.glob("datasets/**/*.txt", recursive=True):
    source = Path(filepath).stem
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        text = f.read()
        counts[source] += len(text.split())

for source, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{source}: ~{count/1e6:.1f}M words (~{count/750/1e6:.1f}M tokens)")

print(f"\nTotal: ~{sum(counts.values())/1e6:.1f}M words (~{sum(counts.values())/750:.0f}K tokens)")