"""Smoke test for the health endpoint. Requires a reachable Postgres/Redis
(e.g. run via `docker-compose up` first, or point DATABASE_URL/REDIS_URL at
local instances) — this is an integration test, not a unit test.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint_responds():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()
