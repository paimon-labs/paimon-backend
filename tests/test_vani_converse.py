import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.narada.orchestrator import TurnResult


@pytest.fixture
def client():
    return TestClient(app)


def test_converse_round_trip(client, tmp_path):
    fake_audio_out = tmp_path / "reply.mp3"
    fake_audio_out.write_bytes(b"fake-mp3-bytes")

    fake_turn = TurnResult(
        turn_id="turn-123",
        response="It's sunny today.",
        task_type="factual_qa",
        skill_used=None,
        provider_order=["nvidia-build", "groq", "local-ollama"],
    )

    with (
        patch("app.modules.vani.router.stt.listen", new=AsyncMock(return_value="what's the weather")),
        patch("app.modules.vani.router.handle_turn", new=AsyncMock(return_value=fake_turn)),
        patch("app.modules.vani.router.tts.speak", new=AsyncMock(return_value=fake_audio_out)),
        patch("app.modules.vani.router.record_event", new=AsyncMock()),
    ):
        response = client.post(
            "/vani/converse",
            files={"audio": ("test.ogg", io.BytesIO(b"fake-audio-bytes"), "audio/ogg")},
            data={"conversation_id": "conv-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-123"
    assert body["transcript"] == "what's the weather"
    assert body["response_text"] == "It's sunny today."
    assert body["audio_mime"] == "audio/mpeg"
    assert len(body["audio_base64"]) > 0


def test_converse_stt_failure_returns_502(client):
    with (
        patch("app.modules.vani.router.stt.listen", new=AsyncMock(side_effect=RuntimeError("stt down"))),
        patch("app.modules.vani.router.record_event", new=AsyncMock()),
    ):
        response = client.post(
            "/vani/converse",
            files={"audio": ("test.ogg", io.BytesIO(b"fake-audio-bytes"), "audio/ogg")},
        )

    assert response.status_code == 502
