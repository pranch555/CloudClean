"""GET /api/system: server health for the Settings page."""
from fastapi.testclient import TestClient

from cloudclean.web.server import create_app


def test_system_status(tmp_path):
    with TestClient(create_app(tmp_path / "ws")) as client:
        r = client.get("/api/system")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["host"] and body["cpu_count"] >= 1
        assert body["disk"]["free_gb"] > 0 and isinstance(body["disk"]["low"], bool)
        assert isinstance(body["gpus"], list)
