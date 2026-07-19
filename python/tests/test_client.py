from unittest.mock import MagicMock, patch

import pytest

from polygate import chat
from polygate.exceptions import MissingAPIKeyError, UnsupportedProviderError


def test_unsupported_provider_raises():
    with pytest.raises(UnsupportedProviderError):
        chat(provider="not-a-real-provider", model="x", messages=[], api_key="key")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        chat(provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


@patch("polygate.providers.openai.requests.post")
def test_openai_chat_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    mock_post.return_value = mock_response

    result = chat(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
    )
    assert result.content == "Hello!"
    assert result.provider == "openai"
    assert result.usage.total_tokens == 7


@patch("polygate.providers.anthropic.requests.post")
def test_anthropic_chat_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "Hi there!"}],
        "usage": {"input_tokens": 4, "output_tokens": 3},
    }
    mock_post.return_value = mock_response

    result = chat(
        provider="claude",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
    )
    assert result.content == "Hi there!"
    assert result.provider == "anthropic"
    assert result.usage.total_tokens == 7


@patch("polygate.providers.gemini.requests.post")
def test_gemini_chat_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Bonjour!"}]}}],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }
    mock_post.return_value = mock_response

    result = chat(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
    )
    assert result.content == "Bonjour!"
    assert result.provider == "gemini"


@patch("polygate.providers.moonshot.requests.post")
def test_moonshot_chat_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "kimi-k2-0711-preview",
        "choices": [{"message": {"role": "assistant", "content": "Ni hao!"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }
    mock_post.return_value = mock_response

    result = chat(
        provider="kimi",
        model="kimi-k2-0711-preview",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
    )
    assert result.content == "Ni hao!"
    assert result.provider == "moonshot"


@patch("polygate.providers.openai.requests.post")
def test_provider_api_error_raised(mock_post):
    from polygate.exceptions import ProviderAPIError

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "invalid api key"
    mock_post.return_value = mock_response

    with pytest.raises(ProviderAPIError):
        chat(
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="bad-key",
        )
