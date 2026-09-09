# The three-minute demo (task 7.7)

A shot list, not a storyboard. Whoever records this needs to know what to type,
what the viewer should notice, and what to do when something does not appear —
so each beat states its **point** and its **failure mode**.

Everything here runs against `examples/demo/`, which is **offline**: the task
answers from a dict and sleeps. Nothing costs money, nothing needs a key, and
the recording is reproducible next month when a model has changed underneath it.
The one beat that needs a real model is noted, and has an offline substitute.

**Target: under 3 minutes.** The timings below sum to 2:35, leaving room to
breathe. If it runs long, cut beat 6 before shortening beat 2 — the live table
is the thing people cannot picture from prose.

---

## Setup, before recording

```bash
cd examples/demo
git checkout capitals_eval.py     # the deliberate wrong answer must be present
rm -rf .evalstand                 # start with no history
clear
```

Terminal at **120×32** or wider. Below that the summary table wraps and the
trace tree becomes unreadable — the two things the video exists to show.

---

## Beat 1 — What an eval is (0:00–0:25)

**Show:** `capitals_eval.py` on screen, scrolled to the `evaluate()` call.

**Say:** "An eval is three things. Cases are the inputs. The task is your
function. Scorers judge what it returned. That's the whole API."

**Point:** a viewer should believe they could write this in two minutes.

**Do not** narrate the imports or the dataclass. The `evaluate()` call is the
API; everything else is Python they already know.

---

## Beat 2 — The live view (0:25–1:05)

**Type:** `evalstand watch`

**Show:** rows appearing one at a time. **Wait for it.** The instinct is to talk
over the pause, but the pause *is* the feature — a run that filled in instantly
would be indistinguishable from a static report, which is what every other tool
shows.

**Say:** "Rows land as each case finishes. Six cases, one is wrong."

**Point:** incremental results. This is the beat that justifies the TUI.

**Failure mode:** if the table appears complete in one frame, the sleeps in
`answer()` are too short for the recording frame rate. Lengthen them in
`capitals_eval.py` rather than re-taking.

---

## Beat 3 — A case in detail, and its trace tree (1:05–1:35)

**Do:** arrow down to `peru`, press `enter`.

**Show:** the full output, the per-scorer breakdown, and the trace tree with its
model, duration and cost.

**Say:** "Here's the case that failed. Expected Lima, got Cusco. And here's what
the task actually did — every call, nested, with what it cost."

**Point:** the trace tree is the *beyond-parity* feature. The reference
implementation's traces are a flat list; a viewer who has used one should notice
the nesting.

**Then:** press `escape`.

---

## Beat 4 — Watch mode (1:35–2:05)

**Do:** in a second pane, fix the wrong answer:

```bash
sed -i 's/"peru": "Cusco"/"peru": "Lima"/' capitals_eval.py
```

**Show:** the eval re-running by itself, and `peru` turning green.

**Say:** "Editing the file re-runs the eval. No restart, no re-typing."

**Point:** the feedback loop. This is why the tool exists.

**Failure mode:** if nothing happens within two seconds, the watcher never
started — check that `watch` was run without `--once`.

---

## Beat 5 — History and comparison (2:05–2:25)

**Do:** press `q`, then:

```bash
evalstand history
evalstand compare <old-run> <new-run>
```

**Show:** the two runs, and the delta on `exact` moving from 0.83 to 1.00.

**Say:** "Every run is recorded locally. Compare shows what changed — a
difference, not a verdict. There's no significance testing here, so it won't
tell you a change is real."

**Point:** the honesty. Say this line as written; it is the project's position,
not a disclaimer to rush past.

---

## Beat 6 — The CI gate (2:25–2:35)

**Do:** re-break the answer, then:

```bash
sed -i 's/"peru": "Lima"/"peru": "Cusco"/' capitals_eval.py
evalstand run --threshold 0.9 --fail-on-error ; echo "exit $?"
```

**Show:** the threshold failure line, and **`exit 1`**.

**Say:** "In CI, `--threshold` fails the build when quality drops. Exit 1 means
below the bar; exit 2 means something didn't run — a different problem, and a
different person to wake up."

**Point:** it works in a pipeline, and the exit codes distinguish a worse model
from a broken one.

**Substitute:** the plan says "see the threshold gate fail in CI". Showing a
real GitHub Actions run means a push, a wait, and a repository the viewer cannot
see. The local exit code is the same mechanism and verifiable on screen — cut to
a still of a red check only if one already exists.

---

## Recording it

`asciinema` keeps the terminal as text, which stays legible at any size and
diffs in review:

```bash
asciinema rec demo.cast --cols 120 --rows 32
# ... perform the beats ...
agg demo.cast docs/demo.gif --font-size 15
```

Or `vhs`, driving it from a tape (`examples/demo/demo.tape` covers beats 2–4
non-interactively).

## Before publishing

- **Under 3 minutes**, and under 5 MB if it becomes the README GIF
- The table **visibly fills** row by row — if not, the recording missed beat 2
- The trace tree is legible at the recorded font size
- `sed` edits do not appear as typing in the final cut; either use a second pane
  or hide them
- `git checkout capitals_eval.py` afterwards, so the repo keeps its deliberate
  failure for the next recording
