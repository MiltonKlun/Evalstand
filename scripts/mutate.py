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

# (file, description, original, replacement)
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
        target.write_text(original.replace(old, new, 1), encoding="utf-8")
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
