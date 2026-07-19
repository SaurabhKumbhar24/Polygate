# Contributing

Thanks for considering a contribution. The scope of this project is
intentionally narrow — a thin, consistent wrapper around LLM provider
APIs — so the bar for "does this belong here" is: *does it help
normalize calling different providers through one interface?* Features
like retries, caching, cost tracking, and routing are explicitly
out of scope; wrap `polygate` with your own middleware for those.

## Running from source

### Python

```bash
cd python
pip install -e ".[dev]"
pytest
```

### TypeScript

```bash
cd typescript
npm install
npm test
```

## Adding a new provider

Adding a provider should only require:

1. **Python**: one new file in `python/polygate/providers/<name>.py` with
   a `chat(*, model, messages, api_key, temperature, max_tokens, **kwargs)`
   function that returns a `ChatResponse`. Register it (and any
   aliases) in `python/polygate/providers/__init__.py`.
2. **TypeScript**: one new file in `typescript/src/providers/<name>.ts`
   exporting an async `chat(options: ProviderChatOptions)` function
   that returns a `ChatResponse`. Register it in
   `typescript/src/providers/index.ts`.
3. Tests for the new adapter (mock the HTTP call — don't hit real
   provider APIs in the test suite).
4. A row in the provider table in the root `README.md`.

Look at `anthropic.py`/`anthropic.ts` and `gemini.py`/`gemini.ts` for
two adapters with meaningfully different request shapes — most new
providers will look like one or the other.

## Guidelines

- Keep each adapter readable in under ~100 lines. If a provider needs
  much more than that (e.g. complex auth, multi-step requests), open
  an issue to discuss the approach first.
- No adapter should depend on the provider's official SDK — adapters
  talk to the HTTP API directly, so the whole project stays
  dependency-light (Python: only `requests`; TypeScript: none, using
  native `fetch`).
- Don't break the existing normalized `ChatResponse`/`Usage` shape.
  If a provider returns something that doesn't fit, put it in `.raw`
  instead of forcing a new top-level field onto everyone else.

## Reporting issues

Open a GitHub issue with a clear description and, if it's a bug, the
smallest reproduction you can manage (provider, model, and the
messages you sent — redact your API key).
