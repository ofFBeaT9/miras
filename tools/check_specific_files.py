import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

filenames = [
    'MANUFACTURING_READINESS_CHECKLIST.md',
    'MODULE_DEPENDENCY_MAP.md',
    'OPENROAD_SYNTHESIS_SUMMARY_1.5GHZ.md',
    'OPENROAD_COMPLETE_FLOW_EXECUTION_REPORT.txt',
    'README.md',
    'tritone_ieee_camera_ready_FINAL_CORRECTED.tex',
    'Tritone article draft v1.1.pdf'
]

with TestClient(app) as client:
    project = 'my-saas-app'
    for fn in filenames:
        q = fn
        r = client.post(f'/projects/{project}/recall', json={'query': q, 'top_k': 5})
        if r.status_code != 200:
            print(fn, 'ERROR', r.status_code, r.text)
            continue
        j = r.json()
        print('\n===', fn, '===')
        print('count:', j.get('count'))
        for m in j.get('memories', []):
            content = m.get('content','')
            print('--- memory id:', m.get('id'))
            print(content[:400].replace('\n','\n'))
            print('...')
