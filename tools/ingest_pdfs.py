import argparse
import os
from pathlib import Path
from typing import List

MAX_CHARS = 120000
BATCH_SIZE = 50


def find_pdfs(root: Path) -> List[Path]:
    ignores = {".venv", ".git", "node_modules"}
    pdfs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip ignored dirs
        parts = set(Path(dirpath).parts)
        if parts & ignores:
            continue
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                pdfs.append(Path(dirpath) / fn)
    return sorted(pdfs)


def extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf is required: pip install pypdf") from e

    try:
        reader = PdfReader(str(path))
        texts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                texts.append(t)
        return "\n\n".join(texts)
    except Exception as e:
        return f"[PDF-EXTRACTION-ERROR] {e}\n"


def chunk_list(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def build_item(path: Path, repo_root: Path) -> dict:
    rel = path.relative_to(repo_root).as_posix()
    txt = extract_text_from_pdf(path)
    if len(txt) > MAX_CHARS:
        txt = txt[:MAX_CHARS]
    content = f"File: {rel}\n\n{txt}"
    return {"content": content, "metadata": {"path": rel, "source": "pdf"}}


def ingest_items_via_testclient(items, project_id: str):
    try:
        from fastapi.testclient import TestClient
    except Exception:
        raise RuntimeError("fastapi[testclient] is required in the environment")

    # prefer fixed server if present
    try:
        from api import server_fixed as server_mod
    except Exception:
        try:
            from api import server as server_mod
        except Exception as e:
            raise RuntimeError("Could not import api.server_fixed or api.server") from e
    # Try in-process TestClient first; if server registry isn't ready, fall back to HTTP
    stored = 0
    skipped = 0
    try:
        client = TestClient(server_mod.app)
        for batch in chunk_list(items, BATCH_SIZE):
            r = client.post(f"/projects/{project_id}/memorize/bulk", json={"items": batch})
            if r.status_code != 200:
                print("ERROR (TestClient): bulk ingest returned", r.status_code, r.text)
                raise RuntimeError("TestClient ingest failed")
            j = r.json()
            stored += j.get("stored", 0) or j.get("stored_total", 0) or len(batch)
        return stored, skipped
    except Exception:
        # Fall back to HTTP against localhost:8100
        try:
            import requests
        except Exception:
            raise RuntimeError("requests is required for HTTP fallback: pip install requests")
        url_base = "http://127.0.0.1:8100"
        for batch in chunk_list(items, BATCH_SIZE):
            try:
                r = requests.post(f"{url_base}/projects/{project_id}/memorize/bulk", json={"items": batch}, timeout=60)
            except Exception as e:
                print("ERROR (HTTP): request failed ->", e)
                continue
            if r.status_code != 200:
                print("ERROR (HTTP): bulk ingest returned", r.status_code, r.text)
                continue
            j = r.json()
            stored += j.get("stored", 0) or j.get("stored_total", 0) or len(batch)
        return stored, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="my-saas-app", help="Project id to ingest into")
    p.add_argument("--root", default=None, help="Repository root (defaults to two parents up)")
    args = p.parse_args()

    repo_root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    print("Scanning for PDFs under", repo_root)
    pdfs = find_pdfs(repo_root)
    print(f"Found {len(pdfs)} PDF files")
    if not pdfs:
        return

    items = []
    for p in pdfs:
        try:
            it = build_item(p, repo_root)
        except Exception as e:
            print("Failed to build item for", p, "->", e)
            continue
        items.append(it)

    print(f"Prepared {len(items)} items; ingesting in batches of {BATCH_SIZE}...")
    stored, skipped = ingest_items_via_testclient(items, args.project)
    print("Ingestion complete. Stored:", stored, "Skipped:", skipped)


if __name__ == "__main__":
    main()
