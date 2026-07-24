# polygate

One function. Any LLM provider.

`polygate` is a tiny, dependency-light wrapper that gives you a single
consistent interface for calling Anthropic (Claude), OpenAI, Google
Gemini, and Moonshot (Kimi K2), instead of learning four different
SDKs and four different response shapes.

```python
from polygate import chat

response = chat(
    provider="anthropic",       # or "openai", "gemini", "moonshot"
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
Portkey, or Helicone. It's a small, boring, easy-to-read wrapper built
to be genuinely simple to depend on and easy to maintain. Contributions
are welcome, but the scope is intentionally kept narrow: normalize the
request/response shape across providers, plus the two reliability
primitives that are impractical to bolt on from outside: key rotation
and backoff retries, both opt-in and off by default. Caching and routing
are still yours to build on top.

## Key rotation and retries

Both are optional. A single key with no `retry` behaves exactly as it did
before these existed: one request, one response, errors raised straight
through.

```ts
import { chat } from "polygate";

const response = await chat({
  provider: "anthropic",
  model: "claude-sonnet-4-6",
  messages: [{ role: "user", content: "Say hi." }],
  apiKey: ["sk-1", "sk-2", "sk-3"],   // string | string[] | KeyPool
  retry: { maxAttempts: 4 },
});
```

A rate limit is per key, not per host, so a 429 rotates to another key
*before* it sleeps. Backing off while a key sits idle wastes the whole
window. A key the provider rejects (401/403) is evicted for the rest of
the process instead of burning an attempt on every later request.
Deterministic failures (400, 404, 422) are never retried.

The provider environment variables also accept a comma-separated list, so
rotation can be turned on without a code change.

## Why this exists

Most unified-LLM libraries are either full frameworks (with retries,
caching, cost tracking, routing, and a dozen other opinions baked in)
or tightly coupled to one ecosystem (e.g. LangChain). `polygate` is
neither. It's the smallest possible layer that lets you swap
providers by changing one string, with each provider adapter being
readable in under 100 lines.

## Packages

| Language | Package | Install |
|---|---|---|
| Python | `polygate` | `pip install polygate` |
| TypeScript / JavaScript | `polygate` | `npm install polygate` |

Both packages are published at `0.1.0`. To run from source instead, see
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Supported providers

| Provider | Aliases | Env var fallback |
|---|---|---|
| Anthropic (Claude) | `anthropic`, `claude` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai`, `gpt` | `OPENAI_API_KEY` |
| Google Gemini | `gemini`, `google` | `GEMINI_API_KEY` |
| Moonshot (Kimi K2) | `moonshot`, `kimi` | `MOONSHOT_API_KEY` |

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

- `content`: the assistant's reply text
- `role`: always `"assistant"`
- `model`: the model that was actually used
- `provider`: which provider served the request
- `usage`: `{ prompt_tokens, completion_tokens, total_tokens }`
- `raw`: the untouched, original provider response, for anything not
  normalized above

## Extra parameters

Anything you pass beyond `provider`, `model`, `messages`, `api_key`,
`temperature`, and `max_tokens` is forwarded straight through to the
provider's request payload, so provider-specific options (tools,
`top_p`, stop sequences, etc.) still work without `polygate` needing to
know about them ahead of time.

## Project layout

```
polygate/
├── python/       # pip package
├── typescript/   # npm package
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Adding a new provider is
usually just: one new adapter file + one line in the registry.

## License

MIT. See [LICENSE](./LICENSE).
