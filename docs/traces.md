# Traces

A Trace records one model call made inside a Task: its input, output, model,
duration, tokens, and cost. Traces **nest** — a call made inside another call is
its child, so a Result carries a tree rather than a list.

That tree is what makes a bad score actionable. A case that scored 0.2 tells you
something is wrong; a trace tree showing that the retrieval step returned nothing
tells you *what*.

## Automatic capture

Every call made through `evalstand.llm` during a Task is recorded. Nothing to
turn on:

```python
from evalstand import evaluate, llm


async def answer(question: str) -> str:
    reply = await llm.acall("gpt-4o-mini", [{"role": "user", "content": question}])
    return reply.text
```

The Result for each case now carries one Trace with the model, both token
counts, the duration, and the cost.

## Naming your own spans

`trace()` wraps any operation so it appears in the tree:

```python
from evalstand import trace


async def answer(question: str) -> str:
    with trace("retrieve"):
        context = await search(question)

    with trace("generate"):
        reply = await llm.acall(
            "gpt-4o-mini",
            [{"role": "user", "content": f"{context}\n\n{question}"}],
        )

    return reply.text
```

Anything that happens inside a `trace()` block becomes its child, including
model calls. The tree above has two roots; a model call inside `generate`
appears beneath it.

Extra fields are attached to the node:

```python
with trace("retrieve", output=f"{len(hits)} documents"):
    ...
```

Outside a running case `trace()` does nothing rather than raising, so a function
you are experimenting with at a REPL still works.

## Reading the tree

`evalstand show <run-id>` prints it:

```
traces
├── retrieve  12ms  -
│   └── embed  gpt-4o-mini  84ms  8 in / 0 out  $0.0000
└── generate  gpt-4o-mini  1502ms  312 in / 47 out  $0.0004
```

A `-` in the cost column means the call was never priced — not that it was free.
`--full` prints the prompts and completions themselves; they are withheld by
default because one 4000-token prompt fills a screen and buries the tree it
belongs to, and because a trace input can hold a customer record nobody meant to
display on a shared terminal.

In the live view, `enter` on any row opens the same tree for that case.

## How nesting works, and when it degrades

A `ContextVar` holds the currently open node. Entering a scope sets it and keeps
the token; leaving resets it in a `finally`. That single rule survives both of
the things that break naive approaches:

- **`asyncio.gather`** — contextvars are copied per task, so concurrent siblings
  share a parent without coordinating.
- **an exception mid-call** — the `finally` reset stops a raising call from
  corrupting its siblings' parentage.

**When a parent cannot be determined confidently, the node attaches to the
Result root.** A visible orphan is honest; a wrongly-parented node is a
plausible-looking lie, and a tree you cannot trust is worse than a flat list you
can. See [ADR 0006](adr/0006-trace-parenting.md).

The shape is validated before it is stored. A cycle has no root and would hang a
renderer; a dangling parent would silently orphan a node. Both are rejected
where the data is built rather than where it is drawn.

## Cost

Each node carries its own cost, and a Result's total is the sum of its nodes.
When a model is not in LiteLLM's pricing table the cost is `None` — recorded as
unknown, never as zero.

That distinction runs through every report: a run with unpriced calls shows its
total as a **lower bound**, because the alternative is understating a bill.

```
> 3 model calls could not be priced, so the cost above is a lower bound.
```
