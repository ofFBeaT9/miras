"""Ingest docs using FastAPI TestClient (in-process) to avoid network issues."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

DEFAULT_PATTERNS = [".md", ".markdown", ".txt", ".rst"]


def find_files(root: Path, patterns):
    for dirpath, dirnames, filenames in os.walk(root):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='my-saas-app')
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

    stored_total = 0
    skipped_total = 0
    failed = 0

    with TestClient(app) as client:
        for i, b in enumerate(batch(items, args.batch), start=1):
            payload = {'items': b}
            r = client.post(f"/projects/{args.project}/memorize/bulk", json=payload)
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
