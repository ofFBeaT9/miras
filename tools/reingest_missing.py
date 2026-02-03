"""Re-ingest missing files found by comparison script.

This script finds files not recorded with a `File: <relpath>` prefix
in project `my-saas-app` and POSTs them in batches to the in-process
/memorize/bulk endpoint using TestClient.
"""
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

PATTERNS = ['.md', '.markdown', '.txt', '.rst']
ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'my-saas-app'
BATCH = 50


def find_files(root: Path, patterns):
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).parts
        if any(p in ('.venv', 'venv', '.git', '__pycache__') for p in parts):
            continue
        for fn in filenames:
            if any(fn.lower().endswith(p) for p in patterns):
                yield Path(dirpath) / fn


def read_file(path: Path, max_chars=120000):
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        try:
            text = path.read_text(encoding='latin-1')
        except Exception:
            return None, 'binary_or_unreadable'
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, 'truncated' if truncated else 'ok'


def make_item(path: Path, root: Path):
    rel = str(path.relative_to(root)).replace('\\','/')
    text, status = read_file(path)
    if text is None:
        return None
    content = f"File: {rel}\n\n" + text
    tags = [path.suffix.lstrip('.'), 'doc', 'reingest']
    if status == 'truncated':
        tags.append('truncated')
    return {
        'content': content,
        'content_type': 'document',
        'tags': tags,
        'source_agent': 'ingest',
        'force': True,
    }


def batch_iter(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def main():
    with TestClient(app) as client:
        # ensure project exists
        r = client.get(f'/projects/{PROJECT}')
        if r.status_code != 200:
            print('Project not available:', r.status_code, r.text)
            return

        # build set of already-ingested file paths
        registry = app.router.dependencies if False else None
        # access registry via server module
        from api import server_fixed as sf
        mem = sf.registry.get_project(PROJECT)
        ingested = set()
        for m in mem.memories:
            c = getattr(m, 'content', '')
            if c.startswith('File:'):
                firstline = c.splitlines()[0]
                rel = firstline[len('File: '):].strip()
                ingested.add(rel.replace('\\','/'))

        # find files
        files = list(find_files(ROOT, PATTERNS))
        files_rel = [str(p.relative_to(ROOT)).replace('\\','/') for p in files]

        # filter missing
        missing_paths = [p for p, r in zip(files, files_rel) if r not in ingested]
        print(f'Found {len(files)} candidate files, missing {len(missing_paths)}')
        if not missing_paths:
            print('Nothing to re-ingest.')
            return

        items = []
        for p in missing_paths:
            it = make_item(p, ROOT)
            if it:
                items.append(it)
        print(f'Prepared {len(items)} items to ingest')

        stored_total = 0
        skipped_total = 0
        for i, b in enumerate(batch_iter(items, BATCH), start=1):
            payload = {'items': b}
            r = client.post(f'/projects/{PROJECT}/memorize/bulk', json=payload)
            if r.status_code != 200:
                print(f'Batch {i} server error: {r.status_code} {r.text}')
                continue
            resp = r.json()
            stored_total += resp.get('stored', 0)
            skipped_total += resp.get('skipped', 0)
            print(f'Batch {i}: total={resp.get("total")} stored={resp.get("stored")} skipped={resp.get("skipped")}')

        print('\nRe-ingest complete:')
        print(f'  items_prepared: {len(items)}')
        print(f'  stored_total: {stored_total}')
        print(f'  skipped_total: {skipped_total}')


if __name__ == '__main__':
    main()
