"""The provider registry and the shared OpenAI-compatible adapter.

Most of the market speaks OpenAI's chat-completions format, so one adapter backs
twenty providers. That is only safe if the differences that *do* exist — the
URL, the environment variable, the reported name — are actually per-provider,
which is what these pin.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polygate import chat  # noqa: E402
from polygate.exceptions import UnsupportedProviderError  # noqa: E402
from polygate.providers import (  # noqa: E402
    CANONICAL_PROVIDERS,
    SUPPORTED_PROVIDERS,
    get_provider,
)
from polygate.providers._openai_compatible import _resolve_url, parse  # noqa: E402

#: Provider → (module name, host fragment, env var). One row per adapter that
#: shares the OpenAI wire format.
COMPATIBLE = [
    ("openai",     "openai",     "api.openai.com",     "OPENAI_API_KEY"),
    ("groq",       "groq",       "api.groq.com",       "GROQ_API_KEY"),
    ("together",   "together",   "api.together.xyz",   "TOGETHER_API_KEY"),
    ("fireworks",  "fireworks",  "api.fireworks.ai",   "FIREWORKS_API_KEY"),
    ("mistral",    "mistral",    "api.mistral.ai",     "MISTRAL_API_KEY"),
    ("xai",        "xai",        "api.x.ai",           "XAI_API_KEY"),
    ("perplexity", "perplexity", "api.perplexity.ai",  "PERPLEXITY_API_KEY"),
    ("cerebras",   "cerebras",   "api.cerebras.ai",    "CEREBRAS_API_KEY"),
    ("deepseek",   "deepseek",   "api.deepseek.com",   "DEEPSEEK_API_KEY"),
    ("moonshot",   "moonshot",   "api.moonshot.ai",    "MOONSHOT_API_KEY"),
    ("zai",        "zai",        "api.z.ai",           "ZAI_API_KEY"),
    # Gateways: same wire format, someone else's models behind it.
    ("openrouter", "openrouter", "openrouter.ai",      "OPENROUTER_API_KEY"),
    ("baseten",    "baseten",    "inference.baseten.co", "BASETEN_API_KEY"),
    ("deepinfra",  "deepinfra",  "api.deepinfra.com",  "DEEPINFRA_API_KEY"),
    ("sambanova",  "sambanova",  "api.sambanova.ai",   "SAMBANOVA_API_KEY"),
    ("nebius",     "nebius",     "api.studio.nebius.com", "NEBIUS_API_KEY"),
    ("novita",     "novita",     "api.novita.ai",      "NOVITA_API_KEY"),
    ("hyperbolic", "hyperbolic", "api.hyperbolic.xyz", "HYPERBOLIC_API_KEY"),
    ("vercel",     "vercel",     "ai-gateway.vercel.sh", "AI_GATEWAY_API_KEY"),
]


# ── Registry ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,module,_host,_env", COMPATIBLE)
def test_each_provider_resolves(name, module, _host, _env):
    assert get_provider(name).__name__.rsplit(".", 1)[-1] == module


@pytest.mark.parametrize(
    "alias,module",
    [
        ("claude", "anthropic"), ("gpt", "openai"), ("google", "gemini"),
        ("kimi", "moonshot"), ("grok", "xai"), ("sonar", "perplexity"),
        ("pplx", "perplexity"), ("togetherai", "together"),
        ("ollama", "custom"), ("local", "custom"), ("vllm", "custom"),
        ("glm", "zai"), ("z.ai", "zai"), ("zhipu", "zai"), ("bigmodel", "zai"),
        ("aws", "bedrock"), ("azure_openai", "azure"), ("vertexai", "vertex"),
        ("ai-gateway", "vercel"), ("workers-ai", "cloudflare"),
    ],
)
def test_popular_names_resolve_too(alias, module):
    """People reach for "grok" long before "xai". Being told that is an unknown
    provider is a bad first minute with a library whose job is hiding this."""
    assert get_provider(alias).__name__.rsplit(".", 1)[-1] == module


def test_provider_names_are_case_and_space_insensitive():
    assert get_provider("  OpenAI ").__name__.endswith("openai")


def test_an_unknown_provider_lists_the_known_ones():
    with pytest.raises(UnsupportedProviderError, match="Supported"):
        get_provider("nope")


def test_none_is_rejected_rather_than_crashing():
    with pytest.raises(UnsupportedProviderError):
        get_provider(None)  # type: ignore[arg-type]


def test_canonical_list_has_no_aliases():
    """It backs docs and UIs, which should not list "gpt" next to "openai"."""
    for name in CANONICAL_PROVIDERS:
        assert name in SUPPORTED_PROVIDERS
    assert len(CANONICAL_PROVIDERS) < len(SUPPORTED_PROVIDERS)
    assert len(set(CANONICAL_PROVIDERS)) == len(CANONICAL_PROVIDERS)


# ── Each provider hits its own endpoint ──────────────────────────────────────

def _ok_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    return response


@pytest.mark.parametrize("name,_module,host,env", COMPATIBLE)
@patch("polygate.providers._openai_compatible.requests.post")
def test_provider_posts_to_its_own_host(mock_post, name, _module, host, env, monkeypatch):
    """The shared adapter must not collapse eleven providers onto one URL."""
    monkeypatch.setenv(env, "test-key")
    mock_post.return_value = _ok_response()

    result = chat(provider=name, model="m", messages=[{"role": "user", "content": "x"}])

    assert host in mock_post.call_args[0][0]
    assert result.provider == name


@pytest.mark.parametrize("name,_module,_host,env", COMPATIBLE)
@patch("polygate.providers._openai_compatible.requests.post")
def test_provider_reads_its_own_env_var(mock_post, name, _module, _host, env, monkeypatch):
    """A shared implementation sharing one env var would send every provider the
    same key."""
    for _n, _m, _h, other in COMPATIBLE:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(env, "sentinel-key")
    mock_post.return_value = _ok_response()

    chat(provider=name, model="m", messages=[{"role": "user", "content": "x"}])

    assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer sentinel-key"


@patch("polygate.providers._openai_compatible.requests.post")
def test_base_url_overrides_the_default_host(mock_post, monkeypatch):
    """A regional endpoint or a proxy shouldn't need a new provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    mock_post.return_value = _ok_response()

    chat(
        provider="openai", model="m", messages=[{"role": "user", "content": "x"}],
        base_url="https://proxy.internal/v1",
    )
    assert mock_post.call_args[0][0] == "https://proxy.internal/v1/chat/completions"


# ── Custom / local endpoints ─────────────────────────────────────────────────

@patch("polygate.providers._openai_compatible.requests.post")
def test_custom_provider_needs_no_api_key(mock_post, monkeypatch):
    """A laptop running Ollama authenticates nothing. Demanding a key would fail
    the most common case for a reason that doesn't exist."""
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    mock_post.return_value = _ok_response()

    result = chat(
        provider="ollama", model="llama3.1:8b",
        messages=[{"role": "user", "content": "x"}],
        base_url="http://localhost:11434/v1",
    )
    assert result.provider == "custom"
    assert mock_post.call_args[0][0] == "http://localhost:11434/v1/chat/completions"


def test_custom_without_an_endpoint_says_so(monkeypatch):
    """Defaulting to localhost would send the caller's prompt somewhere they
    never chose."""
    monkeypatch.delenv("POLYGATE_BASE_URL", raising=False)
    with pytest.raises(UnsupportedProviderError, match="needs an endpoint"):
        chat(provider="custom", model="m", messages=[{"role": "user", "content": "x"}])


@patch("polygate.providers._openai_compatible.requests.post")
def test_custom_endpoint_can_come_from_the_environment(mock_post, monkeypatch):
    monkeypatch.setenv("POLYGATE_BASE_URL", "http://gpu-box:8000/v1")
    mock_post.return_value = _ok_response()

    chat(provider="vllm", model="m", messages=[{"role": "user", "content": "x"}])
    assert mock_post.call_args[0][0].startswith("http://gpu-box:8000/v1")


# ── URL normalisation ────────────────────────────────────────────────────────

def test_a_bare_root_gains_the_completions_path():
    assert _resolve_url("http://x/v1") == "http://x/v1/chat/completions"


def test_a_full_url_is_left_alone():
    """People pass both forms, and posting to the wrong path yields a 404 that
    reads like a broken provider rather than a mistyped setting."""
    full = "http://x/v1/chat/completions"
    assert _resolve_url(full) == full


def test_a_trailing_slash_does_not_double_up():
    assert _resolve_url("http://x/v1/") == "http://x/v1/chat/completions"


# ── Response parsing ─────────────────────────────────────────────────────────

def test_missing_usage_is_zeros_not_an_error():
    """"OpenAI-compatible" is a claim, not a standard: several providers omit
    usage on some routes. A response that arrived must not raise."""
    result = parse(
        {"choices": [{"message": {"content": "hi"}}]},
        provider="groq", fallback_model="m",
    )
    assert result.usage.total_tokens == 0
    assert result.content == "hi"


def test_reasoning_content_is_used_when_content_is_empty():
    """DeepSeek's reasoning models put the answer there."""
    result = parse(
        {"choices": [{"message": {"content": "", "reasoning_content": "the answer"}}]},
        provider="deepseek", fallback_model="m",
    )
    assert result.content == "the answer"


def test_an_empty_choices_list_does_not_crash():
    result = parse({"choices": []}, provider="groq", fallback_model="fallback")
    assert result.content == ""
    assert result.model == "fallback"


def test_the_model_falls_back_when_the_provider_omits_it():
    result = parse(
        {"choices": [{"message": {"content": "x"}}]},
        provider="groq", fallback_model="llama-3.3-70b",
    )
    assert result.model == "llama-3.3-70b"


def test_role_defaults_to_assistant():
    result = parse(
        {"choices": [{"message": {"content": "x"}}]},
        provider="groq", fallback_model="m",
    )
    assert result.role == "assistant"


def test_raw_carries_the_untouched_body():
    """Perplexity returns citations no other provider does; normalising them
    away would lose the only reason to use it."""
    body = {"choices": [{"message": {"content": "x"}}], "citations": ["https://a"]}
    assert parse(body, provider="perplexity", fallback_model="m").raw["citations"] == ["https://a"]


# ── Z.AI ─────────────────────────────────────────────────────────────────────

def test_zai_keeps_its_unusual_path():
    """Z.AI serves the OpenAI format from /api/paas/v4, not /v1. Getting this
    wrong yields a 404 that reads like a dead provider rather than a bad path."""
    from polygate.providers import zai
    assert zai.API_URL == "https://api.z.ai/api/paas/v4/chat/completions"


def test_the_brand_spelling_with_a_dot_resolves():
    """"z.ai" is how the name is written everywhere it appears. A user typing it
    the way the vendor does must not be told the provider is unknown."""
    assert get_provider("z.ai").__name__.endswith("zai")
    assert get_provider(" Z.AI ").__name__.endswith("zai")


@patch("polygate.providers._openai_compatible.requests.post")
def test_the_mainland_host_is_reachable_without_a_second_provider(mock_post, monkeypatch):
    """Zhipu's mainland platform is the same API on a different host, and keys
    do not cross between them. base_url covers it, so there is no near-duplicate
    provider to keep in step."""
    monkeypatch.setenv("ZAI_API_KEY", "k")
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"choices": [{"message": {"content": "hi"}}]},
    )
    from polygate.providers import zai
    chat(provider="zai", base_url=zai.CHINA_BASE_URL,
         model="glm-4.6", messages=[{"role": "user", "content": "hi"}])
    assert mock_post.call_args[0][0] == (
        "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    )
