#!/usr/bin/env python3
"""Pre-deployment checklist for Render."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
REQUIRED_FILES = [
    "render.yaml",
    "requirements.txt",
    "README.md",
    ".env.example",
    "app/main.py",
    "app/config.py",
    "data/corpus/chunks.jsonl",
    "data/glossary.json",
    "data/topics.json",
    "data/regions.json",
]
REQUIRED_DIRS = [
    "app",
    "app/api",
    "app/agents",
    "app/engine",
    "app/rag",
    "data/corpus",
    "data/curriculum",
    "data/personas",
]


def main():
    print("🔍 Checking Render deployment structure...\n")
    
    missing = []
    for path in REQUIRED_FILES:
        full = ROOT / path
        if not full.exists():
            missing.append(f"  ❌ Missing file: {path}")
            print(f"  ❌ Missing: {path}")
        else:
            size = full.stat().st_size
            print(f"  ✅ {path} ({size:,} bytes)")
    
    for path in REQUIRED_DIRS:
        full = ROOT / path
        if not full.is_dir():
            missing.append(f"  ❌ Missing directory: {path}")
            print(f"  ❌ Missing: {path}/")
        else:
            items = list(full.iterdir())
            print(f"  ✅ {path}/ ({len(items)} items)")
    
    # Check data files
    print("\n📊 Data files:")
    chunks_file = ROOT / "data/corpus/chunks.jsonl"
    if chunks_file.exists():
        chunks = sum(1 for _ in chunks_file.open(encoding="utf-8"))
        print(f"  ✅ Corpus: {chunks:,} chunks")
    
    curriculum = list((ROOT / "data/curriculum").glob("*.json"))
    print(f"  ✅ Curriculum: {len(curriculum)} personas")
    
    seeds = list((ROOT / "data/personas").glob("*.seed.json"))
    print(f"  ✅ Seeds: {len(seeds)} personas")
    
    # Check requirements.txt
    print("\n📦 Dependencies:")
    reqs = (ROOT / "requirements.txt").read_text()
    active_reqs = [line for line in reqs.split('\n') if line.strip() and not line.strip().startswith('#')]
    active_text = '\n'.join(active_reqs)
    
    if "transformers" in active_text or "torch" in active_text:
        print("  ⚠️  WARNING: Heavy dependencies detected (transformers/torch)")
        print("     These won't fit in free tier 512MB RAM")
        missing.append("  ⚠️  Remove transformers/torch from requirements.txt")
    else:
        print("  ✅ Lightweight dependencies (no transformers/torch)")
    
    if "chromadb" in active_text:
        print("  ⚠️  WARNING: ChromaDB enabled but data/chroma/ not included")
        print("     Consider commenting out chromadb in requirements.txt")
    
    # Summary
    print("\n" + "="*60)
    if missing:
        print("❌ Deployment check FAILED\n")
        print("Issues found:")
        for m in missing:
            print(m)
        sys.exit(1)
    else:
        print("✅ Deployment check PASSED")
        print("\nReady to deploy! Next steps:")
        print("  1. git init && git add . && git commit -m 'Deploy to Render'")
        print("  2. Create GitHub repo: shastrartha-backend")
        print("  3. git remote add origin <your-repo-url>")
        print("  4. git push -u origin main")
        print("  5. Connect to Render and deploy")
        print("\nSee DEPLOY.md for detailed instructions.")


if __name__ == "__main__":
    main()
