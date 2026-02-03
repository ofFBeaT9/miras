import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

queries = [
    'Progressive Technique Flow',
    'Discussion Orchestration and Multi-Agent Conversation',
    'brainstorming',
    'party-mode'
]

with TestClient(app) as client:
    project = 'my-saas-app'
    for q in queries:
        r = client.post(f'/projects/{project}/recall', json={'query': q, 'top_k': 5})
        print('\n=== Query:', q, '===')
        if r.status_code != 200:
            print('ERROR', r.status_code, r.text)
            continue
        j = r.json()
        print('count:', j.get('count'))
        for m in j.get('memories', []):
            print('id:', m.get('id'))
            print('content snippet:', m.get('content','')[:200].replace('\n','\\n'))
            print('tags:', m.get('tags'))
