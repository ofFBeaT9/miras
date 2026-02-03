import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api import server_fixed as sf

# patterns to consider (same as ingest)
PATTERNS = ['.md', '.markdown', '.txt', '.rst']

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def find_files(root: Path, patterns):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip venv and .git and miras-api folder
        parts = Path(dirpath).parts
        if any(p in ('.venv', 'venv', '.git', '__pycache__') for p in parts):
            continue
        # also skip the miras-api memory store and tools output
        if 'miras-api' in parts and parts[-1] == 'miras-api':
            continue
        for fn in filenames:
            if any(fn.lower().endswith(p) for p in patterns):
                files.append(Path(dirpath) / fn)
    return files

with TestClient(sf.app) as client:
    # use registry from server_fixed (should be initialized by lifespan)
    registry = sf.registry
    if registry is None:
        print('Registry not initialized')
        raise SystemExit(1)
    try:
        mem = registry.get_project('my-saas-app')
    except KeyError:
        print('Project my-saas-app not found')
        raise SystemExit(1)

    # collect ingested file paths from memories
    ingested = set()
    for m in mem.memories:
        c = getattr(m, 'content', '')
        if c.startswith('File:'):
            firstline = c.splitlines()[0]
            rel = firstline[len('File: '):].strip()
            ingested.add(rel.replace('\\','/'))

    # find all candidate files
    files = find_files(ROOT, PATTERNS)
    files_rel = [str(p.relative_to(ROOT)).replace('\\','/') for p in files]

    missing = []
    for f in files_rel:
        if f not in ingested:
            missing.append(f)

    print(f'Total files found: {len(files_rel)}')
    print(f'Ingested memories with File: prefix: {len(ingested)}')
    print(f'Missing count: {len(missing)}')
    if missing:
        print('\nSome missing files (first 200):')
        for m in missing[:200]:
            print(m)
    else:
        print('\nAll files appear ingested.')
