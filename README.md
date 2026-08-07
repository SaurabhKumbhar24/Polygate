# polygate

One function. Any LLM provider.

[![PyPI](https://img.shields.io/pypi/v/polygate?label=pypi)](https://pypi.org/project/polygate/)
[![npm](https://img.shields.io/npm/v/polygate?label=npm)](https://www.npmjs.com/package/polygate)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

`polygate` is a tiny, dependency-light wrapper that gives you a single
consistent interface for calling Anthropic (Claude), OpenAI, Google
Gemini, and Moonshot (Kimi K2) — instead of learning four different
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
Portkey, or Helicone — it's a small, boring, easy-to-read wrapper built
to be genuinely simple to depend on and easy to maintain. Contributions
are welcome, but the scope is intentionally kept narrow: normalize the
request/response shape across providers, nothing more (no built-in
retries, caching, or routing logic — bring your own if you need them,
or build them on top).

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
