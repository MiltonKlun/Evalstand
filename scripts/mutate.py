#!/usr/bin/env python
"""Mutation testing: break the code on purpose and check the suite notices.

A surviving mutant means no test asserts on that behaviour — the line is covered
but not verified, which is the gap coverage percentages cannot show.

    python scripts/mutate.py            # run every mutant
    python scripts/mutate.py --list     # just show what would run

**Bytecode is purged around every mutant.** Restoring the source file is not
enough: Python caches compiled modules, and during the post-Phase-2 audit a
mutated `.pyc` outlived its restored source and corrupted later measurements —
`UNKNOWN` read as "0" while the file on disk said "-". Any harness that edits
source in place has to clear `__pycache__` or its results cannot be trusted.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A few mutants need a helper defined inside the module they mutate (to make a
# real except-clause unreachable, say). It is injected at a known anchor so the
# mutant body can refer to it.
PRELUDE = "class _NeverRaised(Exception):\n    pass\n\n\n"
PRELUDE_ANCHOR = 'EVAL_FILE_SUFFIX = "_eval.py"'

# (file, description, original, replacement)
#
# The anchors are verbatim copies of source lines, so they cannot be wrapped to
# satisfy the line limit: an anchor that no longer matches silently reports
# ANCHOR NOT FOUND instead of testing anything.
# ruff: noqa: E501
MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "src/evalstand/models.py",
        "errored scores count towards the mean",
        "return self.error is None",
        "return True",
    ),
    (
        "src/evalstand/models.py",
        "an empty mean returns 0.0 instead of None",
        "return sum(values) / len(values) if values else None",
        "return sum(values) / len(values) if values else 0.0",
    ),
    (
        "src/evalstand/models.py",
        "Score.value upper bound removed",
        "value: Annotated[float, Field(ge=0.0, le=1.0)] | None = None",
        "value: float | None = None",
    ),
    (
        "src/evalstand/models.py",
        "content_hash ignores expected",
        '{"input": self.input, "expected": self.expected}',
        '{"input": self.input}',
    ),
    (
        "src/evalstand/models.py",
        "a cancelled batch is comparable",
        "return self.status is BatchStatus.COMPLETED",
        "return self.status is not BatchStatus.RUNNING",
    ),
    (
        "src/evalstand/models.py",
        "trace cycles accepted",
        'raise ValueError(f"traces form a cycle through {current!r}")',
        "break",
    ),
    (
        "src/evalstand/llm.py",
        "an unpriced call reports 0.0",
        'logger.debug("could not price completion: %s", exc)\n        return None',
        'logger.debug("could not price completion: %s", exc)\n        return 0.0',
    ),
    (
        "src/evalstand/llm.py",
        "401 becomes retryable",
        "RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})",
        "RETRYABLE_STATUS_CODES = frozenset({401, 408, 409, 429, 500, 502, 503, 504, 529})",
    ),
    (
        "src/evalstand/llm.py",
        "the cache key ignores temperature",
        'KEY_PARAMETERS = ("temperature", "top_p", "max_tokens", "seed", "tools", '
        '"response_format")',
        'KEY_PARAMETERS = ("top_p", "max_tokens", "seed", "tools", "response_format")',
    ),
    (
        "src/evalstand/cache.py",
        "bypass still reads",
        "if bypass:\n            return None\n",
        "if False:\n            return None\n",
    ),
    (
        "src/evalstand/cache.py",
        "bypass still writes",
        "if bypass:\n            return\n",
        "if False:\n            return\n",
    ),
    (
        "src/evalstand/cassettes.py",
        "an api_key is recorded into a committed cassette",
        '"params": {name: params[name] for name in KEY_PARAMETERS if name in params},',
        '"params": dict(params),',
    ),
    (
        "src/evalstand/api.py",
        "auto-numbering overwrites a supplied id",
        'if isinstance(item, dict) and not item.get("id"):',
        "if isinstance(item, dict):",
    ),
    (
        "src/evalstand/api.py",
        "duplicate eval names silently merge",
        "if not same_known_file:",
        "if False:",
    ),
    (
        "src/evalstand/api.py",
        "duplicate case ids allowed",
        "if duplicates:",
        "if False:",
    ),
    (
        "src/evalstand/plugin.py",
        "repeats collapse to a single item",
        "for repeat_index in range(declared.repeat):",
        "for repeat_index in range(1):",
    ),
    (
        "src/evalstand/plugin.py",
        "an unmeasured case passes",
        "if self.scores and not any(s.counts_towards_mean for s in self.scores):",
        "if False:",
    ),
    (
        "src/evalstand/reporting/console.py",
        "the unknown placeholder becomes a number",
        'UNKNOWN = "-"',
        'UNKNOWN = "0"',
    ),
    (
        "src/evalstand/reporting/console.py",
        "unjudged results counted in the pass denominator",
        "judged = [r for r in run.results if any(s.passed is not None for s in r.scores)]",
        "judged = list(run.results)",
    ),
    (
        "src/evalstand/reporting/console.py",
        "an unpriced run reports as free",
        "if not priced:\n        return UNKNOWN",
        "if not priced:\n        return '$0.0000'",
    ),
    (
        "src/evalstand/runner.py",
        "repeats no longer bypass the cache",
        "bypass = config.bypass_cache or declared.repeat > 1",
        "bypass = config.bypass_cache",
    ),
    (
        "src/evalstand/runner.py",
        "the concurrency semaphore is removed",
        "async with semaphore:",
        "if True:",
    ),
    (
        "src/evalstand/runner.py",
        "a failed case is scored anyway",
        "if error is not None",
        "if False",
    ),
    (
        "src/evalstand/tracing.py",
        "the span contextvar token is never reset",
        "        current_span.reset(token)",
        "        pass  # current_span.reset(token)",
    ),
    (
        "src/evalstand/tracing.py",
        "unpriced calls count as free in the case total",
        "        if self.unpriced_call_count:\n            return None",
        "        if False:\n            return None",
    ),
    (
        "src/evalstand/models.py",
        "cache_hit_rate reports 0.0 when nothing was called",
        "        if not self.model_calls:\n            return None",
        "        if not self.model_calls:\n            return 0.0",
    ),
    # --- Phase 3 wiring: the runner reached through the plugin ---
    # --- selection: the money-losing bug ---
    (
        "src/evalstand/plugin.py",
        "deselected cases are executed anyway (-k costs full price)",
        "            units.add((item.case.id, item.repeat_index))",
        "            units.update((c.id, r) for c in item.declared.load_cases()\n"
        "                         for r in range(item.declared.repeat))",
    ),
    (
        "src/evalstand/runner.py",
        "run_eval ignores the selection it was given",
        "    if only is not None:\n        units = [unit for unit in units if (unit[1].id, unit[2]) in only]",
        "    if False:\n        units = [unit for unit in units if (unit[1].id, unit[2]) in only]",
    ),
    # --- the results must reach the items ---
    (
        "src/evalstand/plugin.py",
        "results are never attached to their items",
        "            item.result = by_eval.get(item.declared.name, {}).get((item.case.id, item.repeat_index))",
        "            item.result = None",
    ),
    (
        "src/evalstand/plugin.py",
        "results are matched by case only, ignoring repeat_index",
        "        run.name: {(r.case_id, r.repeat_index): r for r in run.results} for run in runs",
        "        run.name: {(r.case_id, 0): r for r in run.results} for run in runs",
    ),
    (
        "src/evalstand/plugin.py",
        "a case the runner never executed reports as a pass",
        '            raise EvalCaseUnmeasuredError(f"case {self.case.id!r} was never executed by the runner")',
        "            return",
    ),
    # --- the summary must carry the runner's real numbers ---
    (
        "src/evalstand/plugin.py",
        "the summary drops the runner's Runs (no traces, tokens or cost)",
        '    return list(getattr(config, "_evalstand_runs", []))',
        "    return []",
    ),
    (
        "src/evalstand/plugin.py",
        "runs are never stored for the summary",
        "    session.config._evalstand_runs = runs  # type: ignore[attr-defined]",
        "    pass",
    ),
    # --- the flags ---
    (
        "src/evalstand/plugin.py",
        "--no-cache is ignored",
        '    kwargs: dict[str, Any] = {"bypass_cache": bool(config.getoption("--no-cache", default=False))}',
        '    kwargs: dict[str, Any] = {"bypass_cache": False}',
    ),
    (
        "src/evalstand/plugin.py",
        "--concurrency is ignored",
        '        kwargs["concurrency"] = concurrency',
        "        pass",
    ),
    (
        "src/evalstand/plugin.py",
        "--timeout is ignored",
        '        kwargs["timeout_seconds"] = timeout',
        "        pass",
    ),
    (
        "src/evalstand/plugin.py",
        "an invalid flag value crashes instead of being a usage error",
        "    except ValueError as exc:\n        raise pytest.UsageError(str(exc)) from exc",
        "    except _NeverRaised as exc:\n        raise pytest.UsageError(str(exc)) from exc",
    ),
    # --- a task error must still fail its item ---
    (
        "src/evalstand/plugin.py",
        "a task that raised reports as a pass",
        "        if self.result.error is not None:\n            raise EvalTaskError(self.result.error)",
        "        if False:\n            raise EvalTaskError(self.result.error)",
    ),
    # --- the frames ---
    (
        "src/evalstand/runner.py",
        "the user's stack frames are never captured",
        "                frames = _user_frames(exc)",
        "                frames = []",
    ),
    (
        "src/evalstand/runner.py",
        "internal machinery frames leak into the user's report",
        "        if _is_library_frame(frame.filename):\n            continue",
        "        if False:\n            continue",
    ),
    (
        "src/evalstand/plugin.py",
        "captured frames are never rendered",
        "            if self.result is not None:\n                lines += self.result.error_frames",
        "            if False:\n                lines += self.result.error_frames",
    ),
    # --- --collect-only must not spend money ---
    (
        "src/evalstand/plugin.py",
        "--collect-only executes the evals anyway",
        "    if session.config.option.collectonly:\n        return None",
        "    if False:\n        return None",
    ),
    # --- a plain test session must stay untouched ---
    (
        "src/evalstand/plugin.py",
        "a session with no evals still builds a RunConfig and runs",
        "    if not wanted:\n        # A plain test session must be untouched by a plugin it never asked for.\n        return None",
        "    if False:\n        return None",
    ),
    # --- evals must not run concurrently with each other ---
    (
        "src/evalstand/plugin.py",
        "evals run concurrently, multiplying the concurrency the user asked for",
        "    return [await run_eval(declared, config, only=units) for declared, units in wanted.values()]",
        "    import asyncio as _a\n"
        "    return list(await _a.gather(*(run_eval(d, config, only=u) for d, u in wanted.values())))",
    ),
]


def purge_bytecode() -> None:
    """Delete every cached module under src/.

    Restoring a mutated file leaves its compiled bytecode behind, and Python
    will happily keep using it. This is the difference between a trustworthy
    result and a confidently wrong one.
    """
    for cache in ROOT.joinpath("src").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def run_suite() -> bool:
    """True when the suite failed, which is what killing a mutant looks like."""
    proc = subprocess.run(
        ["uv", "run", "pytest", "tests", "-x", "-q", "--no-cov", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode != 0


def run_mutant(rel: str, description: str, old: str, new: str) -> tuple[bool, str]:
    target = ROOT / rel
    original = target.read_text(encoding="utf-8")

    if old not in original:
        return False, "ANCHOR NOT FOUND (the mutant is stale, not the code)"

    try:
        mutated = original.replace(old, new, 1)
        if "_NeverRaised" in new:
            mutated = mutated.replace(PRELUDE_ANCHOR, PRELUDE + PRELUDE_ANCHOR, 1)
        target.write_text(mutated, encoding="utf-8")
        purge_bytecode()
        return (True, "killed") if run_suite() else (False, "SURVIVED")
    except subprocess.TimeoutExpired:
        # A hang kills the mutant, but by wedging CI rather than going red.
        return True, "killed (by hanging — worth investigating)"
    finally:
        target.write_text(original, encoding="utf-8")
        purge_bytecode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the mutants and exit")
    args = parser.parse_args()

    if args.list:
        for rel, description, _, _ in MUTANTS:
            print(f"  {rel:38} {description}")
        return 0

    survivors: list[tuple[str, str]] = []
    for rel, description, old, new in MUTANTS:
        killed, note = run_mutant(rel, description, old, new)
        print(f"  {'OK ' if killed else 'GAP'}  {description}  [{note}]", flush=True)
        if not killed:
            survivors.append((description, note))

    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} killed")
    if survivors:
        print("\nSURVIVORS — behaviour no test asserts on:")
        for description, note in survivors:
            print(f"  - {description}  [{note}]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
