import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.server_fixed import app

with TestClient(app) as client:
	resp = client.post("/projects", json={"project_id":"my-saas-app","name":"My SaaS Application","preset":"titans_default"})
	print(resp.status_code)
	print(resp.json())
