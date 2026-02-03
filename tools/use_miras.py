import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

with TestClient(app) as client:
    project_id = "my-saas-app"

    # 1) Memorize a fact
    mem_payload = {
        "content": "Use PostgreSQL for primary datastore with Prisma ORM",
        "content_type": "decision",
        "tags": ["database","backend"],
        "source_agent": "architect",
        "force": True
    }
    r1 = client.post(f"/projects/{project_id}/memorize", json=mem_payload)
    print("MEMORIZE ->", r1.status_code, json.dumps(r1.json(), indent=2))

    # 2) Recall by query
    recall_payload = {"query": "PostgreSQL", "top_k": 5}
    r2 = client.post(f"/projects/{project_id}/recall", json=recall_payload)
    print("RECALL ->", r2.status_code, json.dumps(r2.json(), indent=2))

    # 3) Get context window
    ctx_payload = {"query": "datastore", "max_tokens": 800}
    r3 = client.post(f"/projects/{project_id}/context", json=ctx_payload)
    print("CONTEXT ->", r3.status_code)
    print(r3.json().get("context","(no context)"))

    # 4) Chat (will use fallback if Anthropic not configured)
    chat_payload = {
        "message": "How should we set up the database migrations?",
        "system_prompt": "You are a helpful assistant aware of the project's persistent decisions.",
        "model": "claude-sonnet-4-5-20250929",
        "source_agent": "dev",
        "memorize_response": False
    }
    r4 = client.post(f"/projects/{project_id}/chat", json=chat_payload)
    print("CHAT ->", r4.status_code, json.dumps(r4.json(), indent=2))
