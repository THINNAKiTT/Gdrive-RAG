"""
Fixtures scoped ONLY to tests/unit/.

This is where the autouse environment-faking fixture lives, kept
separate from tests/conftest.py specifically so it cannot leak into
tests/integration/, which needs the real .env values (real
GCP_CREDENTIALS_PATH, real GOOGLE_DRIVE_FOLDER_ID, etc).
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """
    Every unit test gets a clean, fake environment so nothing
    accidentally reaches a real Google Drive account, a real Ollama
    server, or the developer's real ./chroma_db and ./logs
    directories -- even if a test forgets to mock something.
    """
    monkeypatch.setenv("GCP_CREDENTIALS_PATH", str(tmp_path / "fake_creds.json"))
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "fake_folder_id")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield
