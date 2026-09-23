# Where this project stands

Updated 2026-09-23. `evalstand` 1.0.1 is on PyPI and the showcase baseline is
recorded. The Jev evaluation lives in `JEV.md` (short version: not as a scorer;
yes as an extraction method, which is what produced the baseline).

## The one-line answer

Every checkbox in `PLAN.md` is done except 7.7, the 3-minute narrated video,
which needs a person to narrate it. Nothing is half-finished.

## Verified state

| | |
| --- | --- |
| Release | **`evalstand` 1.0.1 on PyPI**, tag `v1.0.1`, installed from PyPI into a clean venv and run end to end |
| Tests | 1631 passing, 8 deselected (the `live` marker) |
| Mutants | 287 in `scripts/mutate.py`, **zero stale anchors** (a suite test enforces it) |
| Gate | `ruff check` · `ruff format --check` · `mypy` · `mkdocs build --strict` clean over the whole tree |
| CI | green on Python 3.11, 3.12, 3.13 |
| Docs | live at <https://miltonklun.github.io/Evalstand/> |

## What remains

1. **7.7 — the 3-minute video.** Write an eval, run it in watch mode, edit the
   prompt, watch the re-run, open a trace tree, break something and see the
   threshold gate fail in CI. It needs a narrator. The GIF
   (6.8) is done; `examples/demo/README.md` records how it was made, including
   the Windows workarounds.

2. **Optional: the generative baseline.** `extraction_eval.py` (which also
   scores line items) has never been run against a real model. It needs an
   `OPENAI_API_KEY`, and its numbers belong in a second document beside the
   Jev one, not overwriting it — the two methods answer different questions.


## The intermittent collection failure — a watch item

The full suite once failed intermittently with `collected 0 items` from a
nested `evalstand run`. It has not reproduced in many consecutive full runs
since `c0b0e14`.

**The `entry_points()` theory is disproved** — tested directly on 2026-09-21:
five nested `pytest.main()` runs in one process kept `entry_points(pytest11)`
at 3 and all collected. That poisoning is real for *pytester* sessions (which
is what ADR 0008 documents), not for `pytest.main` from a plain process.

If it reappears, capture the untruncated collection error first with
`pytest --tb=long -rA` and read the `errors` section. The failing test names
shift between runs and are not the signal.

## Conventions worth knowing before changing anything

- **Never run two `uv` commands against this repo at once**, and never run tests
  or edit `src/` while `scripts/mutate.py` is running — it rewrites source files
  and restores them in a `finally`.
- **Verify CI checks over the whole tree**, not a subset.
- **Prove a fix with a mutant that reintroduces the defect** — and confirm the
  mutant actually *applied* before believing the result. A mutation that fails
  to match is indistinguishable from a test that works.
- **Look at the frames.** Recording the demo found four real bugs that a green
  suite and an exit code of 0 both hid. When checking anything visual, inspect
  the output itself.
- **Money is formatted only through `format_usd`.** A test fails on any other
  `${...:.Nf}` in the package.
- `TYPESAFE_API_KEY` lives in `.env`, which is gitignored and **not loaded by
  anything** — export it into the environment of the process that needs it.
- Plugin tests use `runpytest_subprocess`, never the inline runner.
- No time estimates or deadline framing anywhere in this project.
