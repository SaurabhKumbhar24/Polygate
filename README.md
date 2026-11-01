# polygate

One function. Any LLM provider.

[![PyPI](https://img.shields.io/pypi/v/polygate?label=pypi)](https://pypi.org/project/polygate/)
[![npm](https://img.shields.io/npm/v/polygate?label=npm)](https://www.npmjs.com/package/polygate)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

`polygate` is a tiny, dependency-light wrapper that gives you a single
consistent interface for calling 26 LLM providers, clouds and gateways — and anything you host
yourself — instead of learning a different SDK and a different response shape
for each.

```python
from polygate import chat

response = chat(
    provider="anthropic",       # or "openai", "gemini", "groq", "mistral", …
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Say hi in one word."}],
    api_key="sk-...",           # or set ANTHROPIC_API_KEY in the environment
)

print(response.content)   # "Hi!"
print(response.usage)     # Usage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
```

```typescript
import { chat } from "polygate";

const response = await chat({
  provider: "openai",
  model: "gpt-4o",
  messages: [{ role: "user", content: "Say hi in one word." }],
  apiKey: "sk-...", // or set OPENAI_API_KEY in the environment
});

console.log(response.content);
```

This repo isn't trying to out-feature mature gateways like LiteLLM,
Portkey, or Helicone — it's a small, boring, easy-to-read wrapper built
to be genuinely simple to depend on and easy to maintain. Contributions
are welcome, but the scope is intentionally kept narrow: normalize the
request/response shape across providers, nothing more.

## Why this exists

Most unified-LLM libraries are either full frameworks (with retries,
caching, cost tracking, routing, and a dozen other opinions baked in)
or tightly coupled to one ecosystem (e.g. LangChain). `polygate` is
neither — it's the smallest possible layer that lets you swap
providers by changing one string, with each provider adapter being
readable in under 100 lines.

## Packages

| Language | Package | Install |
|---|---|---|
| Python | `polygate` | `pip install polygate` |
| TypeScript / JavaScript | `polygate` | `npm install polygate` |

Both packages are published at `0.1.0`. To run from source instead, see
[CONTRIBUTING.md](./CONTRIBUTING.md).

## Supported providers

| Provider | Aliases | Env var fallback |
|---|---|---|
| OpenAI | `openai`, `gpt` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic`, `claude` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini`, `google` | `GEMINI_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Together AI | `together`, `togetherai` | `TOGETHER_API_KEY` |
| Fireworks | `fireworks` | `FIREWORKS_API_KEY` |
| Perplexity (Sonar) | `perplexity`, `sonar`, `pplx` | `PERPLEXITY_API_KEY` |
| xAI (Grok) | `xai`, `grok` | `XAI_API_KEY` |
| Cerebras | `cerebras` | `CEREBRAS_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Moonshot (Kimi K2) | `moonshot`, `kimi` | `MOONSHOT_API_KEY` |
| Z.AI (GLM) | `zai`, `z.ai`, `glm`, `zhipu`, `bigmodel` | `ZAI_API_KEY` |

**Clouds** — these need an endpoint or region as well as a credential, because
the URL contains your own resource. See [Clouds](#clouds).

| Cloud | Aliases | Credential |
|---|---|---|
| Amazon Bedrock | `bedrock`, `aws` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | `azure`, `azure_openai` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Google Vertex AI | `vertex`, `vertexai` | `GOOGLE_ACCESS_TOKEN` + `VERTEX_BASE_URL` |
| Databricks | `databricks` | `DATABRICKS_TOKEN` + `DATABRICKS_HOST` |
| Cloudflare Workers AI | `cloudflare`, `workers-ai` | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |

**Gateways** — one key, many vendors. Model ids are usually `vendor/model`.

| Gateway | Aliases | Env var fallback |
|---|---|---|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Vercel AI Gateway | `vercel`, `ai-gateway` | `AI_GATEWAY_API_KEY` |
| Baseten | `baseten` | `BASETEN_API_KEY` |
| DeepInfra | `deepinfra` | `DEEPINFRA_API_KEY` |
| SambaNova | `sambanova` | `SAMBANOVA_API_KEY` |
| Nebius AI Studio | `nebius` | `NEBIUS_API_KEY` |
| Novita AI | `novita` | `NOVITA_API_KEY` |
| Hyperbolic | `hyperbolic` | `HYPERBOLIC_API_KEY` |

| Anything else | Aliases | Env var fallback |
|---|---|---|
| Your own endpoint | `custom`, `ollama`, `local`, `vllm` | `POLYGATE_BASE_URL` |

### Z.AI and the mainland endpoint

Z.AI serves GLM from two platforms with **non-interchangeable keys**:
`api.z.ai` internationally (the default) and `open.bigmodel.cn` in mainland
China. They are the same API on a different host, so there is no second
provider — pass the other host as `base_url`:

```python
polygate.chat(
    provider="zai",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4.6",
    messages=[{"role": "user", "content": "hi"}],
)
```

### Clouds

Amazon Bedrock goes through the **Converse** API, so one message shape works
across every model AWS hosts. Credentials come from the standard AWS variables:

```python
polygate.chat(
    provider="bedrock",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "hi"}],
    region="us-west-2",   # or $AWS_REGION, default us-east-1
)
```

Requests are signed with SigV4, implemented here in ~60 lines of stdlib rather
than by depending on boto3 — someone installing polygate to call OpenAI should
not pay for an AWS SDK. The signer is diffed against botocore in the test suite,
and the TypeScript one is pinned to the same outputs, so neither can drift.

Temporary credentials work by appending the session token:
`ACCESS_KEY:SECRET:SESSION_TOKEN`. A list of those rotates exactly like a list
of OpenAI keys.

Azure needs its resource host, and the `model` is your **deployment name**, not
a vendor model id:

```python
polygate.chat(
    provider="azure",
    base_url="https://my-resource.openai.azure.com",   # or $AZURE_OPENAI_ENDPOINT
    model="my-gpt4o-deployment",
    messages=[{"role": "user", "content": "hi"}],
)
```

Vertex authenticates with a short-lived OAuth access token rather than an API
key — `gcloud auth print-access-token` — which expires in about an hour.

### Your own endpoint

Ollama, vLLM, LM Studio, llama.cpp's server, TGI, a corporate gateway — all of
them speak the OpenAI chat-completions format, so none needs its own adapter.
Point `base_url` at it:

```python
chat(
    provider="ollama",
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
    messages=[{"role": "user", "content": "hello"}],
)
```

No API key is required for this provider — a local runtime usually has none, and
demanding one would fail the common case for a reason that doesn't exist. Set
`POLYGATE_BASE_URL` to skip passing `base_url` every time.

`base_url` also works on every other provider, so a regional endpoint or a proxy
needs no new code.

## The unified message format

Regardless of provider, `messages` is always a list of:

```json
{ "role": "system" | "user" | "assistant", "content": "..." }
```

Each adapter translates this into whatever shape the underlying
provider actually expects (e.g. Anthropic's separate `system` field,
Gemini's `contents`/`model` role naming).

## The unified response

Every call returns the same shape, regardless of provider:

- `content` — the assistant's reply text
- `role` — always `"assistant"`
- `model` — the model that was actually used
- `provider` — which provider served the request
- `usage` — `{ prompt_tokens, completion_tokens, total_tokens }`
- `raw` — the untouched, original provider response, for anything not
  normalized above

## Extra parameters

Anything you pass beyond `provider`, `model`, `messages`, `api_key`,
`temperature`, and `max_tokens` is forwarded straight through to the
provider's request payload — so provider-specific options (tools,
`top_p`, stop sequences, etc.) still work without `polygate` needing to
know about them ahead of time.

## Project layout

```
polygate/
├── sdk/
│   ├── python/       # pip package
│   ├── typescript/   # npm package
│   ├── README.md
│   ├── CONTRIBUTING.md
│   └── LICENSE
└── website/          # polygate.getfluiq.com (Next.js, static export)
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Adding a new provider is
usually just: one new adapter file + one line in the registry.

## Who builds this

`polygate` is maintained by the people behind
[Fluiq](https://getfluiq.com), which handles the layer above this one:
tracing, evals, and output guardrails for agents running in production.

Keeping all of that *out* of `polygate` is deliberate. This stays a
wrapper you can depend on without buying into anything else, and it has
no dependency on Fluiq in either direction.

If you're curious what an output guardrail actually catches, there's a
free demo with no account required:
[what an LLM still leaks after it refuses](https://getfluiq.com/response-gate-demo).
It runs five real prompt-injection and data-exfiltration attacks against
Claude Haiku 4.5 — which refused all five, and still put customer PII in
two of the refusals.

## License

MIT — see [LICENSE](./LICENSE).
