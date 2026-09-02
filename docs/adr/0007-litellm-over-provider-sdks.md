# 0007 — LiteLLM rather than provider SDKs

- **Status:** accepted
- **Date:** 2026-09-01

`evalstand` reaches every model through LiteLLM instead of calling provider SDKs
(`openai`, `anthropic`, `google-genai`, …) directly, or defining its own provider
interface with an adapter per vendor.

## Context

An eval tool is provider-agnostic by nature. A user comparing two models across
vendors is a normal case, not an edge one, and the tool must report tokens,
latency, and cost identically for all of them — otherwise the numbers a run
produces are not comparable, which defeats the point.

Three options were available:

**Provider SDKs directly.** Each has its own client, response shape, error
hierarchy, streaming protocol, and token-accounting quirks. Supporting three
vendors means writing and maintaining three of everything, and every new
provider is new work in `llm.py`, `cache.py`, and the tracing layer.

**Our own provider interface with per-vendor adapters.** This is the SDK option
with an abstraction layer added: the same per-vendor work, plus an interface to
maintain. It is what LiteLLM already is, built worse and by fewer people.

**LiteLLM.** One `completion()` signature, one response shape, one error
hierarchy, across 130 providers and 3517 models with pricing data — figures
measured against the pinned version, not claimed.

## Decision

LiteLLM, wrapped in a thin `llm.py` rather than used directly at call sites.

The wrapper matters as much as the choice. Everything `evalstand` needs beyond a
raw call — caching, tracing, retry, cost normalisation, cassette replay — lives
in that one module, so the rest of the codebase never imports `litellm`. The
tracing layer in particular depends on every model call passing through a single
function: automatic trace capture cannot work if a task calls a provider SDK
directly.

Cost accounting is the decisive advantage. `litellm.completion_cost` and
`cost_per_token` maintain per-model pricing that would otherwise be a table we
hand-update and quietly get wrong. Parity item 8 asks for cost per call and per
run; without a maintained pricing source that item is a promise we cannot keep
honestly.

## Consequences

**What it costs us, measured rather than assumed:**

- `import litellm` takes ~3 seconds. Phase 2's pytest plugin pays that on every
  collection, so it likely wants a deferred import.
- It pulls 14 direct requirements — `boto3`, `aiohttp`, `openai`, and others —
  contributing to 102 packages in the environment. Heavy for a tool whose own
  code is small.
- Its response shape is ours by inheritance. When LiteLLM changes how it reports
  usage, our token counts change with it.

**What we accept as risk:**

- Pricing accuracy is delegated. If LiteLLM's table is stale for a model, our
  reported cost is stale too, and we would not know. This is why an unpriced
  call reports `None` rather than `0.0`: at least the gap is visible.
- A provider quirk LiteLLM smooths over is a quirk we cannot see. The cassettes
  from task 1.6 are partial insurance — they pin real payload shapes so a change
  in what LiteLLM returns fails a test rather than silently altering results.

**What stays open:** nothing in the design prevents a second backend later. The
wrapper boundary is the seam; `llm.py` is the only module that would change.
