from ..exceptions import UnsupportedProviderError
from . import (
    anthropic,
    azure,
    baseten,
    bedrock,
    cerebras,
    cloudflare,
    custom,
    databricks,
    deepinfra,
    deepseek,
    fireworks,
    gemini,
    groq,
    hyperbolic,
    mistral,
    moonshot,
    nebius,
    novita,
    openai,
    openrouter,
    perplexity,
    sambanova,
    together,
    vercel,
    vertex,
    xai,
    zai,
)

# Multiple aliases map to the same adapter so callers can use whichever
# name feels natural (e.g. "claude" or "anthropic", "kimi" or "moonshot").
#
# Aliases are also where a model family's popular name lives: people reach for
# "grok" far more often than "xai", and being told that is an unknown provider
# is a bad first minute with a library whose whole job is to hide this.
_REGISTRY = {
    "anthropic": anthropic,
    "claude": anthropic,
    "openai": openai,
    "gpt": openai,
    "gemini": gemini,
    "google": gemini,
    "moonshot": moonshot,
    "kimi": moonshot,
    "groq": groq,
    "together": together,
    "togetherai": together,
    "fireworks": fireworks,
    "mistral": mistral,
    "xai": xai,
    "grok": xai,
    "perplexity": perplexity,
    "pplx": perplexity,
    "sonar": perplexity,
    "cerebras": cerebras,
    "deepseek": deepseek,
    "zai": zai,
    # The dot is how the brand is written, and .lower().strip() below means a
    # caller who types it that way is not punished for it.
    "z.ai": zai,
    "glm": zai,
    # Zhipu is the company behind GLM and the name the mainland platform ships
    # under. Same wire format, different host — see zai.CHINA_BASE_URL.
    "zhipu": zai,
    "bigmodel": zai,
    # ── Clouds ────────────────────────────────────────────────────────────
    # Not token-in-a-header like everything above: Bedrock signs every request,
    # Azure uses its own header and puts the deployment in the path, and Vertex
    # takes a short-lived OAuth token.
    "bedrock": bedrock,
    "aws": bedrock,
    "azure": azure,
    "azure_openai": azure,
    "vertex": vertex,
    "vertexai": vertex,
    "databricks": databricks,
    "cloudflare": cloudflare,
    "workers-ai": cloudflare,

    # ── Gateways and aggregators ──────────────────────────────────────────
    # These route to other people's models, so a model id here is usually
    # "vendor/model" rather than a bare name.
    "openrouter": openrouter,
    "vercel": vercel,
    "ai-gateway": vercel,
    "baseten": baseten,
    "deepinfra": deepinfra,
    "sambanova": sambanova,
    "nebius": nebius,
    "novita": novita,
    "hyperbolic": hyperbolic,

    # Anything you host yourself that speaks the OpenAI format.
    "custom": custom,
    "ollama": custom,
    "local": custom,
    "vllm": custom,
}

SUPPORTED_PROVIDERS = sorted(_REGISTRY.keys())

#: The canonical name per adapter, for docs and UIs that shouldn't list every
#: alias. Ordered as a person would expect to see them.
CANONICAL_PROVIDERS = [
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "groq",
    "together",
    "fireworks",
    "perplexity",
    "xai",
    "cerebras",
    "deepseek",
    "moonshot",
    "zai",
    # Clouds.
    "bedrock",
    "azure",
    "vertex",
    "databricks",
    "cloudflare",
    # Gateways.
    "openrouter",
    "vercel",
    "baseten",
    "deepinfra",
    "sambanova",
    "nebius",
    "novita",
    "hyperbolic",
    "custom",
]


def get_provider(name: str):
    key = (name or "").lower().strip()
    if key not in _REGISTRY:
        raise UnsupportedProviderError(
            f"Unknown provider '{name}'. Supported: {SUPPORTED_PROVIDERS}"
        )
    return _REGISTRY[key]
