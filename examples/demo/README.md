# The demo recording

The GIF at the top of the README is recorded from this directory.

## Why it is offline

`capitals_eval.py` answers from a dict and sleeps briefly. It calls no provider,
so re-recording costs nothing, needs no API key, works in CI, and does not drift
as models change.

That is not a shortcut: what the GIF demonstrates is the **interface** — rows
landing one at a time, a trace tree, a re-run triggered by an edit. None of that
is made more truthful by spending money on it. The example that measures a real
model is [`../pdf_extraction/`](../pdf_extraction/), and its numbers are
reproducible for a different reason: a corpus generated from a fixed seed with
ground truth written at generation time.

The eval gets one case deliberately wrong (`peru` → `Cusco`). A demo where
everything passes shows nothing about what the tool is for, and the mistake is
what gives `f` and the watch-mode re-run something to do.

## Recording it

Needs [VHS](https://github.com/charmbracelet/vhs):

```
cd examples/demo
vhs demo.tape
```

That writes `docs/demo.gif`. Check afterwards:

- **under 5 MB** and **under 30 seconds** — the plan's limits
- the table visibly **fills row by row**; if it appears complete in one frame,
  the recording missed the point and the sleeps in `answer()` need lengthening
- the trace tree is legible at the recorded font size
- the watch-mode re-run happens **inside** the live view, not after a restart

`demo.tape` edits `capitals_eval.py` to trigger the re-run, and resets it with
`git checkout` at the start. Re-record from a clean tree, or the demo starts
from the fixed answer and never shows a failure.
