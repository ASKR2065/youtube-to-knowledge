import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extract_triplets import parse_triplets
from mistral_client import MistralClient, load_dotenv
from summarize import DEPTH_INSTRUCTIONS, DEPTHS, summarize_transcript


def test_parse_triplets_clean_array():
    raw = '[{"subject": "A", "predicate": "founded", "object": "B"}]'
    assert parse_triplets(raw) == [{"subject": "A", "predicate": "founded", "object": "B"}]


def test_parse_triplets_with_markdown_fences():
    raw = '```json\n[{"subject": "A", "predicate": "uses", "object": "C"}]\n```'
    assert parse_triplets(raw) == [{"subject": "A", "predicate": "uses", "object": "C"}]


def test_parse_triplets_skips_invalid_entries():
    raw = json.dumps(
        [
            {"subject": "A", "predicate": "founded", "object": "B"},
            {"subject": "", "predicate": "x", "object": "y"},
            "not-an-object",
            {"subject": "C", "predicate": "built"},
        ]
    )
    triplets = parse_triplets(raw)
    assert triplets == [{"subject": "A", "predicate": "founded", "object": "B"}]


def test_parse_triplets_rejects_non_array():
    with pytest.raises(ValueError):
        parse_triplets('{"subject": "A"}')


def test_depth_instructions_cover_all_depths():
    assert set(DEPTH_INSTRUCTIONS) == set(DEPTHS) == {"light", "standard", "deep"}


def test_summarize_rejects_unknown_depth():
    class FakeClient:
        def chat(self, *a, **kw):
            return "ok"

    with pytest.raises(ValueError):
        summarize_transcript(FakeClient(), "text", "t", "c", "en", "ultra")


def test_summarize_builds_prompt_with_transcript():
    captured = {}

    class FakeClient:
        def chat(self, prompt, system=None, temperature=0.3):
            captured["prompt"] = prompt
            captured["system"] = system
            return "# Summary"

    out = summarize_transcript(FakeClient(), "TRANSCRIPT BODY", "My title", "My channel", "fr", "light")
    assert out == "# Summary"
    assert "TRANSCRIPT BODY" in captured["prompt"]
    assert "My title" in captured["prompt"]
    assert "fr" in captured["prompt"]
    assert "Key takeaways" in captured["prompt"]


def test_missing_api_key_exits(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        MistralClient()


def test_client_reads_env_api_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = MistralClient(api_key="explicit-key", model="test-model")
    assert client.api_key == "explicit-key"
    assert client.model == "test-model"


def test_load_dotenv_sets_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MISTRAL_API_KEY=from-file\n# comment\nMISTRAL_MODEL=mistral-small-latest\n", encoding="utf-8")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    load_dotenv(str(env_file))
    assert os.environ.get("MISTRAL_API_KEY") == "from-file"
    assert os.environ.get("MISTRAL_MODEL") == "mistral-small-latest"


def test_load_dotenv_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    load_dotenv(str(tmp_path / "does-not-exist.env"))
    assert os.environ.get("MISTRAL_API_KEY") is None


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MISTRAL_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MISTRAL_API_KEY", "already-set")
    load_dotenv(str(env_file))
    assert os.environ.get("MISTRAL_API_KEY") == "already-set"


