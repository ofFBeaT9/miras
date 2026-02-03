"""Ingest documentation files into MIRAS project via /memorize/bulk

Usage:
  python tools/ingest_docs.py --project my-saas-app --root "C:\Tritone SoC" --batch 40

Defaults:
  project: my-saas-app
  base_url: http://localhost:8100
  root: two levels up from this script (workspace root)

This script finds .md, .markdown, .txt, .rst files, reads them,
and posts them in batches to the MIRAS /memorize/bulk endpoint.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

DEFAULT_PATTERNS = [".md", ".markdown", ".txt", ".rst"]


def find_files(root: Path, patterns):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtualenvs and .git
        if any(part in (".venv", "venv", ".git", "__pycache__") for part in Path(dirpath).parts):
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
    rel = str(path.relative_to(root))
    text, status = read_file(path)
    if text is None:
        return None
    content = f"File: {rel}\n\n" + text
    tags = [path.suffix.lstrip('.'), 'doc', 'ingest']
    if status == 'truncated':
        tags.append('truncated')
    return {
        'content': content,
        'content_type': 'document',
        'tags': tags,
        'source_agent': 'ingest',
        'force': True,
    }


def batch(iterable, n=40):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='my-saas-app')
    parser.add_argument('--base-url', default='http://localhost:8100')
    parser.add_argument('--root', default=None)
    parser.add_argument('--batch', type=int, default=40)
    parser.add_argument('--patterns', nargs='*', default=DEFAULT_PATTERNS)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_root = Path(script_dir).parent.parent.resolve()
    root = Path(args.root).resolve() if args.root else default_root

    print(f"Scanning {root} for patterns: {args.patterns}")
    files = list(find_files(root, args.patterns))
    print(f"Found {len(files)} files")
    items = []
    for p in files:
        it = make_item(p, root)
        if it:
            items.append(it)

    url = f"{args.base_url.rstrip('/')}/projects/{args.project}/memorize/bulk"
    stored_total = 0
    skipped_total = 0
    failed = 0
    for i, b in enumerate(batch(items, args.batch), start=1):
        payload = {'items': b}
        try:
            r = requests.post(url, json=payload, timeout=60)
        except Exception as e:
            print(f"Batch {i} failed to POST: {e}")
            failed += len(b)
            continue
        if r.status_code != 200:
            print(f"Batch {i} server error: {r.status_code} {r.text}")
            failed += len(b)
            continue
        resp = r.json()
        stored_total += resp.get('stored', 0)
        skipped_total += resp.get('skipped', 0)
        print(f"Batch {i}: total={resp.get('total')} stored={resp.get('stored')} skipped={resp.get('skipped')}")

    print('\nIngestion complete:')
    print(f'  files_found: {len(files)}')
    print(f'  items_attempted: {len(items)}')
    print(f'  stored_total: {stored_total}')
    print(f'  skipped_total: {skipped_total}')
    print(f'  failed_count: {failed}')


if __name__ == '__main__':
    main()
