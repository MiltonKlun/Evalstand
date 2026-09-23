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

Needs [VHS](https://github.com/charmbracelet/vhs), plus `ttyd` and `ffmpeg` on
the PATH — installing VHS with winget does not install either, and VHS does not
say they are missing:

```
cd examples/demo
vhs demo.tape
```

That writes `docs/demo.gif`, and `evalstand` must be on the PATH it inherits.

### What the recording edits, and what it does not

The few seconds `evalstand watch` takes to start are **cut**: importing LiteLLM
alone is over three seconds, even for this eval, which never calls a model. The
tape waits for the live view off-camera, then shows it. Everything after that is
the tool running — rows landing, the trace tree, the filter, and the re-run
triggered by an edit — and the tape waits on what is on screen rather than on
fixed sleeps, so it cannot record before the live view exists or stop before
the re-run finishes.

### On Windows

The tape uses PowerShell. `bash` cannot be made to mean Git Bash here: process
creation searches System32 before PATH, so the name resolves to the WSL launcher
whatever PATH says, and the first attempt recorded a WSL error screen.

VHS 0.12 on Windows also exits 0 without writing a GIF when its ffmpeg step
fails. Its frame capture works, so record the frames and encode them directly:

```
# In a copy of demo.tape, change the Output line to:  Output vhsframes/
vhs framestape.tape
ffmpeg -framerate 50 -i vhsframes/frame-text-%05d.png \
       -framerate 50 -i vhsframes/frame-cursor-%05d.png \
       -filter_complex "[0][1]overlay,trim=end=16.4,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=2,fps=20,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
       -loop 0 ../../docs/demo.gif
```

`trim` ends on the finished run rather than the shell prompt `q` returns to, and
`tpad` holds that last real frame for two seconds so each loop ends on the
result. Find the right trim point by looking at the frames, not by guessing.

### Checking it

Afterwards:

- **under 5 MB** and **under 30 seconds** — the plan's limits
- the table visibly **fills row by row**; if it appears complete in one frame,
  the recording missed the point and the sleeps in `answer()` need lengthening
- the trace tree is legible at the recorded font size
- the watch-mode re-run happens **inside** the live view, not after a restart
- the re-run **changes the result**: peru fails before it and passes after. A
  re-run that leaves it failing ran stale code — the first recording did
  exactly that, and it was a real bug in watch mode, not in the tape
- the latency column shows **numbers**, not `-` — another real bug the frames
  exposed: the runner never recorded latency at all

Look at the frames. Every defect above was invisible in the tool's exit code and
obvious in a single frame.

`demo.tape` edits `capitals_eval.py` to trigger the re-run, and resets it with
`git checkout` at the start. Re-record from a clean tree, or the demo starts
from the fixed answer and never shows a failure.
