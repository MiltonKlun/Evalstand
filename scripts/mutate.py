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


def _line_append_full() -> str:
    """The `--full` prompt line, built from codepoints.

    Written this way because the anchor contains a literal backslash-n: a
    source edit that turns it into a real newline makes the anchor silently
    stop matching, which reports ANCHOR MISS rather than testing anything.
    """
    quote, backslash = chr(34), chr(92)
    return (
        "            line.append(f"
        + quote
        + backslash
        + "n{label}: {value!r}"
        + quote
        + ", style="
        + quote
        + "dim"
        + quote
        + ")"
    )


def _never_raised(*names: str) -> str:
    """Exception classes nothing throws, for making an except-clause dead."""
    return "".join(f"class {name}(Exception):\n    pass\n\n\n" for name in names)


# Modules whose mutants need a differently named helper, or a different anchor
# to inject it at. Anything not listed here uses PRELUDE / PRELUDE_ANCHOR.
EXTRA_PRELUDES = {
    "src/evalstand/provenance.py": (
        _never_raised("_NeverRaised", "_NeverRaisedP"),
        "_TIMEOUT_SECONDS = 10",
    ),
    "src/evalstand/storage.py": (
        _never_raised("_NeverRaisedS"),
        'logger = logging.getLogger("evalstand.storage")',
    ),
    "src/evalstand/recording.py": (
        _never_raised("_NeverRaisedR"),
        'logger = logging.getLogger("evalstand.recording")',
    ),
}

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
        "    runs: list[Run] = []\n    for declared, units in wanted.values():",
        "    import asyncio as _a\n"
        "    return list(await _a.gather(*(run_eval(d, config, only=u) for d, u in wanted.values())))\n"
        "    runs: list[Run] = []\n    for declared, units in wanted.values():",
    ),
    # --- Phase 4: the scorer library ---
    # Was SURVIVED as an equivalent mutant; this version is real.
    (
        "src/evalstand/scorers/base.py",
        "a defaulted parameter counts as one the runner supplies",
        "        if parameter.default is inspect.Parameter.empty:\n            positional += 1",
        "        if True:\n            positional += 1",
    ),
    # Was SURVIVED for real: endpoints-only tests could not see it.
    (
        "src/evalstand/scorers/fuzzy.py",
        "ratio is clamped instead of rescaled from percent",
        '    return Score(scorer_name="ratio", value=fuzz.ratio(left, right) / 100.0)',
        '    return Score(scorer_name="ratio", value=min(fuzz.ratio(left, right), 1.0))',
    ),
    (
        "src/evalstand/scorers/fuzzy.py",
        "levenshtein collapses to a binary check",
        "    similarity = Levenshtein.normalized_similarity(left, right)",
        "    similarity = 1.0 if left == right else 0.0",
    ),
    # The defect found in this audit: over-eager punctuation folding.
    (
        "src/evalstand/scorers/text.py",
        "punctuation is stripped everywhere, so -5 matches 5",
        '    tokens = (_EDGE_PUNCTUATION.sub("", token) for token in folded.split())',
        "    import re as _re\n"
        '    tokens = (_re.sub(r"[^\\w\\s]", "", token) for token in folded.split())',
    ),
    (
        "src/evalstand/scorers/text.py",
        "the trimmed set is widened to swallow signs and currency",
        '        "\\"\'`.,;:!?()[]{}<>",',
        '        "\\"\'`.,;:!?()[]{}<>+-$%=",',
    ),
    (
        "src/evalstand/scorers/text.py",
        "punctuation is not folded at all",
        '    tokens = (_EDGE_PUNCTUATION.sub("", token) for token in folded.split())',
        "    tokens = (token for token in folded.split())",
    ),
    (
        "src/evalstand/scorers/text.py",
        "tokens are joined without a separator",
        '    return " ".join(token for token in tokens if token)',
        '    return "".join(token for token in tokens if token)',
    ),
    (
        "src/evalstand/scorers/text.py",
        "empty tokens are kept, doubling separators",
        '    return " ".join(token for token in tokens if token)',
        '    return " ".join(tokens)',
    ),
    # --- Phase 4.4: the numeric scorer ---
    (
        "src/evalstand/scorers/numeric.py",
        "an ambiguous output takes the first number instead of refusing",
        "    if len(matches) != 1:",
        "    if not matches:",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "ambiguity is judged on distinct values, so 5 + 5 reads as 5",
        "    if len(matches) != 1:",
        "    if len(set(matches)) != 1:",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "nan and infinity are accepted as measurements",
        "    return number if math.isfinite(number) else None",
        "    return number",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "a bool is scored as the number one",
        "    if value is None or isinstance(value, bool):\n        return None",
        "    if value is None:\n        return None",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "the thousands separator is left in, so 1,000 fails to parse",
        '    return _finite(float(matches[0].replace(",", "")))',
        "    return _finite(float(matches[0]))",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "the default tolerance silently allows 1%",
        "    rel_tol: float = 0.0,",
        "    rel_tol: float = 0.01,",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "a negative tolerance is accepted",
        "    if rel_tol < 0 or abs_tol < 0:",
        "    if False:",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "abs_tol is ignored, reintroducing the zero trap",
        "        within = math.isclose(actual, target, rel_tol=rel_tol, abs_tol=abs_tol)",
        "        within = math.isclose(actual, target, rel_tol=rel_tol)",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "rel_tol is ignored",
        "        within = math.isclose(actual, target, rel_tol=rel_tol, abs_tol=abs_tol)",
        "        within = math.isclose(actual, target, abs_tol=abs_tol)",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "magnitudes are compared, so -5 matches 5",
        "        within = math.isclose(actual, target, rel_tol=rel_tol, abs_tol=abs_tol)",
        "        within = math.isclose(abs(actual), abs(target), rel_tol=rel_tol, abs_tol=abs_tol)",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "an unreadable output is reported as a pass",
        "        actual = parse_number(output)\n        if actual is None:",
        "        actual = parse_number(output)\n        if False:",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "an unreadable expected value is reported as a pass",
        "        target = parse_number(expected)\n        if target is None:",
        "        target = parse_number(expected)\n        if False:",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "the scorer stops claiming a verdict",
        "            value=1.0 if within else 0.0,\n            passed=within,",
        "            value=1.0 if within else 0.0,",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "digits inside a token are read as numbers",
        '    (?<![\\w.])          # not mid-token: the "5" in "a5" is not a number',
        '    (?:)          # not mid-token: the "5" in "a5" is not a number',
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "a number ending a sentence fails to parse (the shipped regression)",
        "    (?!\\w)              # not followed by more of a token",
        "    (?![\\w.])           # the original buggy lookahead",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "a dotted sequence like 1.2.3 parses as a number",
        "    (?!\\.\\d)            # nor by more number",
        "    (?:)                # no guard against more number",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "the sign is dropped when parsing",
        "    [+-]?               # sign, which is part of the value",
        "    (?:)               # sign, which is part of the value",
    ),
    (
        "src/evalstand/scorers/numeric.py",
        "the parsed number is not reported in metadata",
        '                "parsed_output": actual,',
        '                "parsed_output": None,',
    ),
    # --- Phase 4.5: the structured scorer ---
    (
        "src/evalstand/scorers/json_field.py",
        "the denominator counts volunteered fields, penalising verbosity",
        "        value = len(matched) / len(fields)",
        "        value = len(matched) / max(len(set(fields) | set(given)), 1)",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a missing field is reported as wrong",
        "            if path not in given:\n                missing.append(path)",
        "            if False:\n                missing.append(path)",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "nested dicts are compared whole rather than flattened",
        '        if isinstance(value, Mapping) and value:\n            flat.update(flatten(value, f"{path}."))',
        '        if False:\n            flat.update(flatten(value, f"{path}."))',
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "an empty nested dict silently drops the requirement",
        "        if isinstance(value, Mapping) and value:",
        "        if isinstance(value, Mapping):",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "an empty expected object scores 1.0 instead of erroring",
        "        if not fields:",
        "        if False:",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a non-object expected value is treated as a task failure",
        "        target = parse_object(expected)\n        if target is None:",
        "        target = parse_object(expected)\n        if False:",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a fenced JSON block is not unwrapped",
        "    fenced = _FENCE.match(text)\n    if fenced:",
        "    fenced = _FENCE.match(text)\n    if False:",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a JSON array is accepted as an object",
        "    return parsed if isinstance(parsed, dict) else None",
        "    return parsed",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a null matches an empty string",
        "        return output_value is None and expected_value is None",
        "        return normalise(output_value) == normalise(expected_value)",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "numbers are compared as text, so 1843 misses 1843.0",
        "    left, right = _as_number(output_value), _as_number(expected_value)",
        "    left, right = None, None",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a number inside prose makes two answers equal",
        "def _as_number(value: Any) -> float | None:",
        "def _as_number(value: Any) -> float | None:\n"
        "    from evalstand.scorers.numeric import parse_number\n"
        "    return parse_number(value)\n"
        "\n"
        "def _unused(value: Any) -> float | None:",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a bool matches the number one",
        "    if isinstance(value, bool):\n        return None",
        "    if False:\n        return None",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "lists ignore order",
        "        return len(output_value) == len(expected_value) and all(\n            _matches(a, b) for a, b in zip(output_value, expected_value, strict=True)\n        )",
        "        return sorted(map(str, output_value)) == sorted(map(str, expected_value))",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "require_all is ignored, so nothing ever claims a verdict",
        "            passed=(not wrong and not missing) if require_all else None,",
        "            passed=None,",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "a partial score claims a verdict it was never given",
        "            passed=(not wrong and not missing) if require_all else None,",
        "            passed=(not wrong and not missing),",
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "the failing field names are not reported",
        '                "wrong": sorted(wrong),',
        '                "wrong": [],',
    ),
    (
        "src/evalstand/scorers/json_field.py",
        "an unreadable output reports no missing fields",
        '                    "missing": sorted(fields),',
        '                    "missing": [],',
    ),
    # --- Phase 4.6: LLM judges ---
    (
        "src/evalstand/scorers/llm.py",
        "an unreadable judgement is laundered into a zero",
        "            return Score.from_error(",
        "            return Score(scorer_name=name, value=0.0)\n        if False:\n            return Score.from_error(",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "an ambiguous reply takes the first choice named",
        "    if len(named) == 1:\n        return named.pop()",
        "    if named:\n        return sorted(named)[0]",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the unvalidated caveat is dropped from the score",
        '                "unvalidated": True,',
        '                "unvalidated": False,',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the judge's own words are not kept",
        '                "judge_reply": response.text.strip(),',
        '                "judge_reply": "",',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the judging model is not recorded",
        '                "judge_model": response.model,',
        '                "judge_model": None,',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the caller's choice mapping is ignored, every choice scores 1.0",
        "            value=choices[choice],",
        "            value=1.0,",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "a judge with no choices is accepted",
        "    if not choices:",
        "    if False:",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "a choice outside the score range is accepted",
        "        if not 0.0 <= value <= 1.0:",
        "        if False:",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the rubric is never sent to the judge",
        '    sections = [rubric, "", f"[Question]\\n{case.input}", "", f"[Submission]\\n{output}"]',
        '    sections = ["", f"[Question]\\n{case.input}", "", f"[Submission]\\n{output}"]',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the submission is never sent to the judge",
        'f"[Submission]\\n{output}"]',
        '"[Submission]"]',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "include_expected=False still leaks the reference answer",
        "    if include_expected:",
        "    if True:",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the available choices are not named in the prompt",
        '        f"Reply with a single letter, one of: {labels}. Give no other text.",',
        '        "Reply with a single letter.",',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "a label inside a word counts as a choice",
        '        if re.search(\n            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE\n        )',
        "        if re.search(re.escape(label), text, flags=re.IGNORECASE)",
    ),
    (
        "src/evalstand/scorers/llm.py",
        "choice matching becomes case sensitive, so 'Yes.' is unreadable",
        '            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE',
        '            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", text',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "factuality treats a superset answer as a disagreement",
        '    "B": 1.0,',
        '    "B": 0.0,',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "factuality treats a disagreement as consistent",
        '    "D": 0.0,',
        '    "D": 1.0,',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "factuality reports under the generic judge name",
        '        name="factuality",',
        '        name="judge",',
    ),
    (
        "src/evalstand/scorers/llm.py",
        "the judge calls the provider directly, escaping trace and cost capture",
        '        response = await acall(model, [{"role": "user", "content": prompt}], **params)',
        "        import litellm as _l\n"
        "        from evalstand.llm import LLMResponse as _R\n"
        '        _raw = await _l.acompletion(model=model, messages=[{"role": "user", "content": prompt}])\n'
        "        response = _R(text=_raw.choices[0].message.content, model=model, latency_ms=1)",
    ),
    # --- Pre-Phase-5 audit: report honesty ---
    (
        "src/evalstand/models.py",
        "F2: a Score may pass with a value of 0.0 (the shipped behaviour)",
        "        if self.passed is True and self.value == 0.0:",
        "        if False:",
    ),
    (
        "src/evalstand/models.py",
        "F2b: a Score may fail with a value of 1.0",
        "        if self.passed is False and self.value == 1.0:",
        "        if False:",
    ),
    (
        "src/evalstand/models.py",
        "F3: a Run may hold the same execution twice",
        "        if duplicates:\n            named =",
        "        if False:\n            named =",
    ),
    (
        "src/evalstand/reporting/console.py",
        "F1a: a case scoring 0.0 is invisible in the report",
        "            zeroed = [s for s in result.scores if s.passed is None and s.value == 0.0]",
        "            zeroed = []",
    ),
    (
        "src/evalstand/reporting/console.py",
        "F1a-inverse: every partial score floods the table",
        "            zeroed = [s for s in result.scores if s.passed is None and s.value == 0.0]",
        "            zeroed = [s for s in result.scores if s.passed is None]",
    ),
    (
        "src/evalstand/plugin.py",
        "F1b: --threshold is ignored, so a garbage run exits zero",
        "        session.exitstatus = EXIT_BELOW_THRESHOLD",
        "        pass",
    ),
    (
        "src/evalstand/plugin.py",
        "F1b-inverse: the threshold fails runs that are above it",
        "        if run.mean_score is not None and run.mean_score < threshold",
        "        if run.mean_score is not None",
    ),
    (
        "src/evalstand/plugin.py",
        "an unmeasured run is reported as below the threshold",
        "        if run.mean_score is not None and run.mean_score < threshold",
        "        if (run.mean_score or 0.0) < threshold",
    ),
    (
        "src/evalstand/plugin.py",
        "the threshold breach is never explained to the user",
        "        config._evalstand_breaches = breaches  # type: ignore[attr-defined]",
        "        pass",
    ),
    (
        "src/evalstand/runner.py",
        "F4: the stored output aliases a task's mutable object",
        "            output=_snapshot(output),",
        "            output=output,",
    ),
    (
        "src/evalstand/runner.py",
        "F4b: an uncopyable output kills the case instead of falling back",
        '    except Exception:\n        logger.debug("could not copy a task output; storing its repr", exc_info=True)\n        return repr(output)',
        "    except _NeverRaised:\n        return repr(output)",
    ),
    # --- Phase 5.1: the run store ---
    # --- the bug that really happened ---
    (
        "src/evalstand/storage.py",
        "content_hash is stored uncalled, so the column holds a method repr",
        "                case.content_hash(),",
        "                str(case.content_hash),",
    ),
    # --- migration safety ---
    (
        "src/evalstand/storage.py",
        "a newer database is opened best-effort instead of refused",
        "        if current > LATEST_VERSION:",
        "        if False:",
    ),
    (
        "src/evalstand/storage.py",
        "migrations run through executescript, so a failure cannot roll back",
        "            for statement in statements:\n                self.connection.execute(statement)",
        "            self.connection.executescript(';\\n'.join(statements))",
    ),
    (
        "src/evalstand/storage.py",
        "an already-applied migration is re-run on every open",
        "            if version <= current:\n                continue",
        "            if False:\n                continue",
    ),
    # --- the pragmas, each per-connection and each easy to omit ---
    (
        "src/evalstand/storage.py",
        "foreign keys are left off, so cascades are decoration",
        '            self.connection.execute("PRAGMA foreign_keys = ON")',
        "            pass",
    ),
    # --- write atomicity ---
    (
        "src/evalstand/storage.py",
        "a failed save leaves partial rows behind",
        '                self.connection.execute("ROLLBACK")\n                logger.exception("failed to save run',
        '                self.connection.execute("COMMIT")\n                logger.exception("failed to save run',
    ),
    # --- what is stored must be what was measured ---
    (
        "src/evalstand/storage.py",
        "an absent verdict is stored as a failure",
        "                None if score.passed is None else int(score.passed),",
        "                int(bool(score.passed)),",
    ),
    (
        "src/evalstand/storage.py",
        "an unknown git state is recorded as clean",
        "                    None if batch.git_dirty is None else int(batch.git_dirty),",
        "                    int(bool(batch.git_dirty)),",
    ),
    (
        "src/evalstand/storage.py",
        "a structured output loses its structure",
        "                _json(result.output),",
        "                None,",
    ),
    (
        "src/evalstand/storage.py",
        "a cyclic output kills the whole save",
        "        return json.dumps(value, default=str, sort_keys=True)\n    except Exception:\n        pass",
        "        return json.dumps(value, default=str, sort_keys=True)\n    except _NeverRaised:\n        pass",
    ),
    (
        "src/evalstand/storage.py",
        "a repr that raises escapes the json fallback",
        "        return json.dumps(repr(value))\n    except Exception:",
        "        return json.dumps(repr(value))\n    except _NeverRaised:",
    ),
    (
        "src/evalstand/storage.py",
        "a str() that raises kills the save",
        "        return str(value)\n    except Exception:",
        "        return str(value)\n    except _NeverRaised:",
    ),
    # --- schema constraints ---
    (
        "src/evalstand/migrations/__init__.py",
        "duplicate executions are allowed by the schema",
        "        UNIQUE (run_id, case_id, repeat_index)",
        "        UNIQUE (run_id, case_id, repeat_index, id)",
    ),
    (
        "src/evalstand/migrations/__init__.py",
        "results do not cascade when their run is deleted",
        "        run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,\n        case_id       TEXT NOT NULL,\n        repeat_index",
        "        run_id        TEXT NOT NULL,\n        case_id       TEXT NOT NULL,\n        repeat_index",
    ),
    (
        "src/evalstand/migrations/__init__.py",
        "scores.value_float is NOT NULL, so an errored score cannot be stored",
        "        value_float   REAL,",
        "        value_float   REAL NOT NULL DEFAULT 0.0,",
    ),
    # --- Phase 5.2: provenance and recording ---
    # --- the bug that made history hold one entry forever ---
    (
        "src/evalstand/migrations/__init__.py",
        "results are keyed by id alone, so a second run of an eval is dropped",
        "        PRIMARY KEY (run_id, id),",
        "        PRIMARY KEY (id),",
    ),
    (
        "src/evalstand/migrations/__init__.py",
        "scores are not scoped by run, moving the collision one level down",
        '        FOREIGN KEY (run_id, result_id) REFERENCES results(run_id, id) ON DELETE CASCADE\n    )\n    """,\n    # `parent_id` is what makes',
        '        FOREIGN KEY (result_id) REFERENCES results(id) ON DELETE CASCADE\n    )\n    """,\n    # `parent_id` is what makes',
    ),
    # --- the "HEAD" trap ---
    (
        "src/evalstand/provenance.py",
        "anything git prints is stored as a sha",
        "    if not _looks_like_a_sha(sha):",
        "    if False:",
    ),
    # --- absent is not false ---
    (
        "src/evalstand/provenance.py",
        "an unknown tree state is reported as clean",
        "        return GitState(sha=sha, dirty=None)",
        "        return GitState(sha=sha, dirty=False)",
    ),
    (
        "src/evalstand/provenance.py",
        "untracked files make the tree dirty",
        '    status = _git("status", "--porcelain", "--untracked-files=no", cwd=directory)',
        '    status = _git("status", "--porcelain", cwd=directory)',
    ),
    (
        "src/evalstand/provenance.py",
        "a missing git crashes the run",
        "    except (OSError, subprocess.SubprocessError):",
        "    except _NeverRaised:",
    ),
    # --- the dirty gate ---
    (
        "src/evalstand/provenance.py",
        "a dirty tree is persisted without --allow-dirty",
        "    if state.dirty and not allow_dirty:",
        "    if False:",
    ),
    (
        "src/evalstand/provenance.py",
        "an unknown tree state is refused like a dirty one",
        "    if state.dirty and not allow_dirty:",
        "    if not state.dirty and not allow_dirty:",
    ),
    # --- the task hash ---
    (
        "src/evalstand/provenance.py",
        "unreadable task source crashes instead of returning None",
        "    except (OSError, TypeError):",
        "    except _NeverRaisedP:",
    ),
    (
        "src/evalstand/provenance.py",
        "indentation is not normalised, so moving a task reads as changing it",
        "    normalised = inspect.cleandoc(source)",
        "    normalised = source",
    ),
    # --- recording ---
    (
        "src/evalstand/recording.py",
        "a storage failure takes down the run",
        "        try:\n            action()\n        except Exception:",
        "        try:\n            action()\n        except _NeverRaisedR:",
    ),
    (
        "src/evalstand/recording.py",
        "the storage warning repeats for every run",
        "            if not self._failed:",
        "            if True:",
    ),
    (
        "src/evalstand/recording.py",
        "a too-new database is silently downgraded to a warning",
        "    except DatabaseTooNewError:\n        # Deliberately not downgraded",
        "    except _NeverRaisedR:\n        # Deliberately not downgraded",
    ),
    (
        "src/evalstand/recording.py",
        "the task source hash is never recorded",
        '        run = run.model_copy(update={"task_source_hash": task_source_hash(task) if task else None})',
        '        run = run.model_copy(update={"task_source_hash": None})',
    ),
    (
        "src/evalstand/recording.py",
        "an interrupted batch is marked completed",
        '                "status": BatchStatus.CANCELLED if cancelled else BatchStatus.COMPLETED,',
        '                "status": BatchStatus.COMPLETED,',
    ),
    (
        "src/evalstand/recording.py",
        "the recorder closes a store it does not own",
        "        if self._owns_store:\n            self.store.close()",
        "        self.store.close()",
    ),
    # --- Phase 5.3: reading history back ---
    (
        "src/evalstand/reporting/console.py",
        "a run that made no calls is reported as free",
        "    if not priced:\n        # Either no calls were made",
        "    if False:\n        # Either no calls were made",
    ),
    (
        "src/evalstand/reporting/console.py",
        "a partly-priced run is shown as an exact total",
        '    return formatted if run.cost_is_complete else f"{formatted}+"',
        "    return formatted",
    ),
    (
        "src/evalstand/reporting/console.py",
        "every cost is marked as a lower bound",
        '    return formatted if run.cost_is_complete else f"{formatted}+"',
        '    return f"{formatted}+"',
    ),
    (
        "src/evalstand/reporting/console.py",
        "a dirty tree is not marked in the commit column",
        '    return f"{short}*" if batch.git_dirty else short',
        "    return short",
    ),
    (
        "src/evalstand/reporting/console.py",
        "every commit is marked dirty",
        '    return f"{short}*" if batch.git_dirty else short',
        '    return f"{short}*"',
    ),
    (
        "src/evalstand/storage.py",
        "cancelled batches are listed alongside completed ones",
        "            \"WHERE b.status != 'cancelled'\",",
        '            "WHERE 1=1",',
    ),
    (
        "src/evalstand/storage.py",
        "history is ordered oldest first",
        '        query.append("ORDER BY r.started_at DESC, r.id DESC LIMIT ?")',
        '        query.append("ORDER BY r.started_at ASC, r.id ASC LIMIT ?")',
    ),
    (
        "src/evalstand/storage.py",
        "the name filter is ignored, so every eval is listed",
        '            query.append("AND r.name = ?")',
        '            query.append("AND ? IS NOT NULL")',
    ),
    (
        "src/evalstand/storage.py",
        "an absent verdict is read back as a failure",
        '                passed=None if score_row["passed"] is None else bool(score_row["passed"]),',
        '                passed=bool(score_row["passed"]),',
    ),
    (
        "src/evalstand/storage.py",
        "a trace loses its parent, flattening the tree",
        '            parent_id=row["parent_id"],',
        "            parent_id=None,",
    ),
    (
        "src/evalstand/storage.py",
        "an unpriced trace is read back as free",
        '            output_tokens=tokens.get("output"),',
        '            output_tokens=tokens.get("output"),\n            cost_usd=row["cost_usd"] or 0.0,  # mutant\n            #',
    ),
    (
        "src/evalstand/storage.py",
        "a structured output is read back as text",
        '            output=_unjson(row["output_json"], fallback=row["output_text"]),',
        '            output=row["output_text"],',
    ),
    (
        "src/evalstand/storage.py",
        "error frames are dropped on read",
        '            error_frames=_unjson(row["error_frames"]) or [],',
        "            error_frames=[],",
    ),
    (
        "src/evalstand/storage.py",
        "an unreadable timestamp takes down the whole run",
        '    except ValueError:\n        logger.debug("could not parse a stored timestamp',
        '    except _NeverRaisedS:\n        logger.debug("could not parse a stored timestamp',
    ),
    (
        "src/evalstand/storage.py",
        "git_dirty None is read back as clean",
        '            git_dirty=None if row["git_dirty"] is None else bool(row["git_dirty"]),',
        '            git_dirty=bool(row["git_dirty"]),',
    ),
    # --- Phase 5.4: the run-detail view ---
    (
        "src/evalstand/reporting/console.py",
        "the tree walk becomes recursive and overflows on deep chains",
        "    pending: list[tuple[Tree, Trace]] = [\n        (tree, trace) for trace in reversed(children.get(None, []))\n    ]\n    while pending:\n        parent, trace = pending.pop()\n        node = parent.add(_trace_label(trace, full=full))\n        pending.extend((node, child) for child in reversed(children.get(trace.id, [])))",
        "    def _walk(parent, trace):\n        node = parent.add(_trace_label(trace, full=full))\n        for child in children.get(trace.id, []):\n            _walk(node, child)\n\n    for trace in children.get(None, []):\n        _walk(tree, trace)",
    ),
    (
        "src/evalstand/reporting/console.py",
        "only the first root is rendered, dropping the rest of a forest",
        "        (tree, trace) for trace in reversed(children.get(None, []))",
        "        (tree, trace) for trace in reversed(children.get(None, [])[:1])",
    ),
    (
        "src/evalstand/reporting/console.py",
        "siblings come out in reverse order",
        "        pending.extend((node, child) for child in reversed(children.get(trace.id, [])))",
        "        pending.extend((node, child) for child in children.get(trace.id, []))",
    ),
    (
        "src/evalstand/reporting/console.py",
        "children are never attached, flattening the tree",
        "        node = parent.add(_trace_label(trace, full=full))",
        "        node = tree.add(_trace_label(trace, full=full))",
    ),
    (
        "src/evalstand/reporting/console.py",
        "an unpriced call is shown as free",
        '    line.append(f"  {UNKNOWN if trace.cost_usd is None else f\'${trace.cost_usd:.4f}\'}", style="dim")',
        '    line.append(f"  ${trace.cost_usd or 0.0:.4f}", style="dim")',
    ),
    (
        "src/evalstand/reporting/console.py",
        "prompts are printed by default",
        "    if not full:\n        # A size rather than the content.",
        "    if False:\n        # A size rather than the content.",
    ),
    (
        "src/evalstand/reporting/console.py",
        "--full still withholds the prompt",
        _line_append_full(),
        "            pass",
    ),
    (
        "src/evalstand/reporting/console.py",
        "a score with no verdict is shown as a pass",
        '    if score.passed is None:\n        # No verdict was given, and the report must not invent one.\n        return f"{score.scorer_name}: {value}"',
        '    if False:\n        return f"{score.scorer_name}: {value}"',
    ),
    (
        "src/evalstand/reporting/console.py",
        "an errored score is rendered as a zero",
        '    if score.error:\n        return f"{score.scorer_name}: errored ({score.error})"',
        '    if False:\n        return f"{score.scorer_name}: errored ({score.error})"',
    ),
    (
        "src/evalstand/reporting/console.py",
        "the excluded-score count is never stated",
        "    if run.errored_score_count:",
        "    if False:",
    ),
    (
        "src/evalstand/reporting/console.py",
        "captured error frames are dropped",
        '        parts.extend(Text(f"  {frame}", style="dim") for frame in result.error_frames)',
        "        pass",
    ),
    (
        "src/evalstand/reporting/console.py",
        "the task source hash is never shown",
        "    if run.task_source_hash:",
        "    if False:",
    ),
    # --- Phase 5.5: comparing two runs ---
    # --- the acceptance criterion itself ---
    (
        "src/evalstand/reporting/console.py",
        "the delta table calls a fall a regression",
        '    table.add_column("changed by", justify="right")',
        '    table.add_column("regression", justify="right")',
    ),
    (
        "src/evalstand/reporting/console.py",
        "the flip table is titled as regressions",
        '        title="cases whose pass state changed",',
        '        title="regressions and improvements",',
    ),
    (
        "src/evalstand/reporting/console.py",
        "the significance-testing caveat is dropped",
        '            "evalstand has no significance testing: a delta is an arithmetic "',
        '            "" if True else "evalstand has no significance testing: a delta is an arithmetic "',
    ),
    # --- Flip vs Amended Case: the evidence rule ---
    (
        "src/evalstand/comparison.py",
        "an edited case is reported as a flip",
        "        if case_id in amended_ids:",
        "        if False:",
    ),
    (
        "src/evalstand/comparison.py",
        "every case is treated as edited, so no flip is ever reported",
        "        if case_id in amended_ids:",
        "        if True:",
    ),
    (
        "src/evalstand/comparison.py",
        "only the expected value is hashed, so an edited input is missed",
        "        if case_id in hashes_after and hashes_after[case_id] != digest",
        "        if False",
    ),
    (
        "src/evalstand/comparison.py",
        "missing snapshots mark every case as amended",
        "    if not hashes_before or not hashes_after:\n        return set()",
        "    if not hashes_before or not hashes_after:\n        return set(hashes_before or {})",
    ),
    # --- verdicts must not be invented ---
    (
        "src/evalstand/comparison.py",
        "a case with no verdict is treated as failing",
        "    verdicts = [score.passed for score in result.scores if score.passed is not None]\n    if not verdicts:\n        return None",
        "    verdicts = [score.passed for score in result.scores if score.passed is not None]\n    if not verdicts:\n        return False",
    ),
    (
        "src/evalstand/comparison.py",
        "a case passes if any scorer passed it, not all",
        "    return all(verdicts)",
        "    return any(verdicts)",
    ),
    (
        "src/evalstand/comparison.py",
        "an unmeasured run's delta is computed as if it were zero",
        "        if self.before is None or self.after is None:\n            return None",
        "        if False:\n            return None",
    ),
    # --- repeats ---
    (
        "src/evalstand/comparison.py",
        "a repeated case is compared using its last result",
        "    return {case_id: result for case_id, result in seen.items() if case_id not in repeated}",
        "    return seen",
    ),
    # --- score moves ---
    (
        "src/evalstand/comparison.py",
        "every score move is listed, however small",
        "        if abs(scores_after[name] - scores_before[name]) >= threshold",
        "        if True",
    ),
    (
        "src/evalstand/comparison.py",
        "score moves are ordered smallest first",
        "        moves=sorted(moves, key=lambda m: abs(m.delta), reverse=True),",
        "        moves=sorted(moves, key=lambda m: abs(m.delta)),",
    ),
    (
        "src/evalstand/comparison.py",
        "cases present in only one run are not flagged",
        "        only_before=sorted(set(results_before) - set(results_after)),",
        "        only_before=[],",
    ),
    # --- the warnings a reader needs ---
    (
        "src/evalstand/reporting/console.py",
        "comparing two different evals is not flagged",
        "    if before.name != after.name:",
        "    if False:",
    ),
    (
        "src/evalstand/reporting/console.py",
        "a changed task source is not flagged",
        "        and before.task_source_hash != after.task_source_hash",
        "        and False",
    ),
    # --- Pre-Phase-6 audit: what history loses ---
    # --- G1: "nothing differs" while every case crashed ---
    (
        "src/evalstand/comparison.py",
        "G1: a case that stopped being measured is invisible (the shipped bug)",
        "        change = _measurement_change(case_id, first, second)\n        if change is not None:",
        "        change = _measurement_change(case_id, first, second)\n        if False:",
    ),
    (
        "src/evalstand/comparison.py",
        "G1b: measurement changes are detected but never carried",
        "        measurement_changes=measurement_changes,",
        "        measurement_changes=[],",
    ),
    (
        "src/evalstand/comparison.py",
        "G1c: is_empty ignores them, so 'nothing differs' returns",
        "            or self.measurement_changes",
        "            or []",
    ),
    (
        "src/evalstand/comparison.py",
        "G1d: a crashed case is folded into flips",
        '    if result.error:\n        return "not run"',
        '    if False:\n        return "not run"',
    ),
    (
        "src/evalstand/comparison.py",
        "G1e: an all-errored case is treated as measured",
        "    if result.scores and all(score.error for score in result.scores):",
        "    if False:",
    ),
    (
        "src/evalstand/comparison.py",
        "G1f: the error is not reported with the change",
        "        detail=_measurement_detail(after) or _measurement_detail(before),",
        "        detail=None,",
    ),
    (
        "src/evalstand/reporting/console.py",
        "G1g: the measurement table is never rendered",
        "    if comparison.measurement_changes:",
        "    if False:",
    ),
    # --- G2: results come back in the wrong order ---
    (
        "src/evalstand/storage.py",
        "G2: results are read back alphabetically, so q10 precedes q2",
        '"ORDER BY ordinal IS NULL, ordinal, case_id, repeat_index",',
        '"ORDER BY case_id, repeat_index",',
    ),
    (
        "src/evalstand/storage.py",
        "G2b: the ordinal is never written",
        "                ordinal,\n            ),",
        "                None,\n            ),",
    ),
    (
        "src/evalstand/storage.py",
        "G2c: rows with no ordinal sort first, jumbling the rest",
        '"ORDER BY ordinal IS NULL, ordinal, case_id, repeat_index",',
        '"ORDER BY ordinal, case_id, repeat_index",',
    ),
    # --- G3: a mismatched run is silently discarded ---
    (
        "src/evalstand/recording.py",
        "G3: a run from another batch is swallowed as a warning",
        "        if run.batch_id != self.batch.id:",
        "        if False:",
    ),
    (
        "src/evalstand/recording.py",
        "G3b: every run is refused, including matching ones",
        "        if run.batch_id != self.batch.id:",
        "        if True:",
    ),
    # --- Phase 6: the live view, watch mode, and the history screen ---tuple[str, str, str, str]] = [
    # --- the live view must show results as they land ---
    (
        "src/evalstand/runner.py",
        "results are announced only when the whole run finishes",
        "        _announce(config.on_result, result)\n        return result",
        "        return result",
    ),
    (
        "src/evalstand/runner.py",
        "a raising result sink takes down the run it was watching",
        "    try:\n        on_result(result)\n    except Exception:",
        "    on_result(result)\n    if False:",
    ),
    # --- what a row is allowed to claim ---
    (
        "src/evalstand/tui/state.py",
        "a broken scorer is reported as a failing model",
        "    if result.scores and all(score.error for score in result.scores):",
        "    if False:",
    ),
    (
        "src/evalstand/tui/state.py",
        "a continuous score without a verdict is called a pass",
        '        return "scored"',
        '        return "pass"',
    ),
    (
        "src/evalstand/tui/state.py",
        "an unpriced case is shown as free rather than unknown",
        "    if result.traces and not priced:",
        "    if False:",
    ),
    (
        "src/evalstand/tui/state.py",
        "the running cost hides that some calls were never priced",
        '        bound = f" (+{self.unpriced_calls} unpriced)" if self.unpriced_calls else ""',
        '        bound = ""',
    ),
    (
        "src/evalstand/tui/state.py",
        "the live pass rate counts unjudged cases as failures",
        '        return f"{self.passed}/{self.judged}"',
        '        return f"{self.passed}/{self.completed}"',
    ),
    (
        "src/evalstand/tui/state.py",
        "an unstarted run shows 0% progress rather than unknown",
        "        if not self.expected:\n            return None",
        "        if not self.expected:\n            return 0.0",
    ),
    (
        "src/evalstand/tui/state.py",
        "a duplicate execution is painted as a second row",
        "        if key in self._seen:\n            return None",
        "        if False:\n            return None",
    ),
    (
        "src/evalstand/tui/state.py",
        "a raising custom column takes down the whole row",
        "        try:\n            value = fn(result)\n        except Exception:",
        "        value = fn(result)\n        if False:",
    ),
    # --- the widgets ---
    (
        "src/evalstand/tui/app.py",
        "row selection is ignored, so enter never opens a case",
        "        if message.row_key.value is not None:\n            self._open(str(message.row_key.value))",
        "        return",
    ),
    (
        "src/evalstand/tui/app.py",
        "the detail view summarises the calls instead of showing them",
        '            Static(render_case(self.result, full=True), id="detail-body"),',
        '            Static(render_case(self.result, full=False), id="detail-body"),',
    ),
    # --- watch mode ---
    (
        "src/evalstand/tui/app.py",
        "a file change does not cancel the run in flight",
        "        self._cancel_in_flight()\n\n        self.state = RunState(",
        "        self.state = RunState(",
    ),
    (
        "src/evalstand/tui/app.py",
        "a batch cut short by an edit is recorded as completed",
        "            self.recorder.finish(cancelled=True)",
        "            self.recorder.finish(cancelled=False)",
    ),
    (
        "src/evalstand/tui/watch.py",
        "generated files trigger a re-run, so a run triggers the next",
        "    if resolved.suffix.lower() not in _WATCHED_SUFFIXES:\n        return False",
        "    if False:\n        return False",
    ),
    (
        "src/evalstand/tui/watch.py",
        "prompt text files are ignored, so watch mode misses the edit",
        '_WATCHED_SUFFIXES = frozenset({".py", ".txt", ".md", ".json", ".yaml", ".yml", ".jinja", ".j2"})',
        '_WATCHED_SUFFIXES = frozenset({".py"})',
    ),
    (
        "src/evalstand/tui/watch.py",
        "the debounce is long enough to break the one-second promise",
        "DEBOUNCE_MS = 300",
        "DEBOUNCE_MS = 1600",
    ),
    (
        "src/evalstand/tui/watch.py",
        "a nested root is watched twice, so one save starts two batches",
        "        if not any(other != root and other in root.parents for other in roots)",
        "        if True",
    ),
    # --- collecting evals without pytest ---
    (
        "src/evalstand/loading.py",
        "several evals are guessed between rather than refused",
        "    if len(evals) > 1:",
        "    if False:",
    ),
    (
        "src/evalstand/loading.py",
        "the registry is not cleared, so deleted evals linger",
        "    registry.clear()",
        "    pass",
    ),
    # --- the guard that must hold from every caller ---
    (
        "src/evalstand/comparison.py",
        "the partial-run guard refuses nothing",
        "        if (batch := batch_of(run.id)) is not None and not batch.is_comparable",
        "        if False",
    ),
    (
        "src/evalstand/tui/history.py",
        "the history view compares a run that never finished",
        "            refuse_partial_runs((before, after), self.store.batch_for)",
        "            pass",
    ),
    (
        "src/evalstand/tui/history.py",
        "a comparison is labelled backwards, inverting every delta",
        "        before, after = sorted((first, second), key=_when)",
        "        before, after = first, second",
    ),
    (
        "src/evalstand/tui/app.py",
        "the search box overrides the failures filter instead of composing",
        '        if self._only_failures and row.status == "pass":\n            return False\n        return self._search.lower() in row.case_id.lower()',
        '        if self._search:\n            return self._search.lower() in row.case_id.lower()\n        return not (self._only_failures and row.status == "pass")',
    ),
    (
        "src/evalstand/tui/app.py",
        "compare is offered against a run that has not finished",
        '        if self.finished is None:\n            self._set_status("the run is still going; compare when it finishes")\n            return',
        "        if False:\n            return",
    ),
    (
        "src/evalstand/tui/app.py",
        "a recorded run is never written, leaving a batch with no runs",
        "            if self.recorder is not None:\n                self.recorder.record(",
        "            if False:\n                self.recorder.record(",
    ),
    (
        "src/evalstand/tui/app.py",
        "a recorded run is filed under the default batch id, so it is refused",
        "            run = await run_eval(self.declared, config, batch_id=self._batch_id())",
        "            run = await run_eval(self.declared, config)",
    ),
    # --- Phase 7: the CI contract and the pull-request comment ---tuple[str, str, str, str]] = [
    # --- the exit-code contract a CI job gates on ---
    (
        "src/evalstand/plugin.py",
        "an execution error exits 1, reading as a quality regression",
        "            session.exitstatus = EXIT_EXECUTION_ERROR",
        "            session.exitstatus = EXIT_BELOW_THRESHOLD",
    ),
    (
        "src/evalstand/plugin.py",
        "a threshold breach overwrites the execution-error exit code",
        "            config._evalstand_error_failures = failures  # type: ignore[attr-defined]\n            session.exitstatus = EXIT_EXECUTION_ERROR\n            return",
        "            config._evalstand_error_failures = failures  # type: ignore[attr-defined]\n            session.exitstatus = EXIT_EXECUTION_ERROR",
    ),
    (
        "src/evalstand/plugin.py",
        "fail-on-error ignores errored scorers, catching only errored tasks",
        "            if any(r.error for r in run.results) or run.errored_score_count",
        "            if any(r.error for r in run.results)",
    ),
    (
        "src/evalstand/plugin.py",
        "fail-on-error fires when nothing actually errored",
        "        if failures:",
        "        if True:",
    ),
    (
        "src/evalstand/cli.py",
        "compare exits 1 for nothing-to-compare, hiding it as a regression",
        "EXIT_NOTHING_TO_COMPARE = 2",
        "EXIT_NOTHING_TO_COMPARE = 1",
    ),
    # --- the pull-request comment ---
    (
        "src/evalstand/reporting/markdown.py",
        "a pipe in model output is not escaped, shifting every column",
        '    text = str(value).replace("|", "\\\\|").replace("\\n", " ").replace("\\r", " ")',
        '    text = str(value).replace("\\n", " ").replace("\\r", " ")',
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "a newline in model output breaks the table into malformed rows",
        '    text = str(value).replace("|", "\\\\|").replace("\\n", " ").replace("\\r", " ")',
        '    text = str(value).replace("|", "\\\\|")',
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "a 4000-token output is pasted whole into the comment",
        '    return text if len(text) <= _MAX_CELL else text[: _MAX_CELL - 3] + "..."',
        "    return text",
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "the comment does not say the cost is a lower bound",
        "    if not unpriced:\n        return []",
        "    if True:\n        return []",
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "an eval whose scorers all errored vanishes from the summary table",
        '            lines.append(\n                f"| {_escape(run.name)} | {UNKNOWN} | {UNKNOWN} | {_cases(run)} | {cost} |"\n            )\n            continue',
        "            continue",
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "no evals having run reads the same as every eval passing",
        '        return "## evalstand\\n\\nNo evals ran."',
        '        return "## evalstand"',
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "errored scores are not declared, so the means look complete",
        "    errored_scores = sum(run.errored_score_count for run in runs)",
        "    errored_scores = 0",
    ),
    (
        "src/evalstand/reporting/markdown.py",
        "a continuous score with no verdict is listed as a failure",
        "            failed = [s for s in result.scores if s.passed is False]",
        "            failed = [s for s in result.scores if not s.passed]",
    ),
    (
        "src/evalstand/__init__.py",
        "the public trace export reverts to something that is not the real one",
        "from evalstand.tracing import trace",
        "from evalstand.api import scorer as trace",
    ),
    # --- the CLI's own argv, which the plugin tests cannot see ---
    (
        "src/evalstand/cli.py",
        "--fail-on-error never reaches pytest, so the gate never fires",
        '    if fail_on_error:\n        args.append("--fail-on-error")',
        "    pass",
    ),
    (
        "src/evalstand/cli.py",
        "--output never reaches pytest, so the PR comment is never produced",
        '    if output != "terminal":\n        args += ["--output", output]',
        "    pass",
    ),
    (
        "src/evalstand/cli.py",
        "the CLI normalises the exit code, erasing 2 vs 1",
        "    raise typer.Exit(code=pytest.main(args))",
        "    raise typer.Exit(code=1 if pytest.main(args) else 0)",
    ),
    (
        "src/evalstand/cli.py",
        "watch ignores --once and watches anyway",
        "        watch=not once,",
        "        watch=True,",
    ),
    (
        "src/evalstand/cli.py",
        "watch records to history without --store",
        "    recorder = None\n    if store:",
        "    recorder = None\n    if True:",
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
            prelude, anchor = EXTRA_PRELUDES.get(rel, (PRELUDE, PRELUDE_ANCHOR))
            mutated = mutated.replace(anchor, prelude + anchor, 1)
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
    stale: list[str] = []

    for rel, description, old, new in MUTANTS:
        killed, note = run_mutant(rel, description, old, new)
        label = "OK " if killed else ("STALE" if note.startswith("ANCHOR") else "GAP")
        print(f"  {label:5} {description}  [{note}]", flush=True)
        if note.startswith("ANCHOR"):
            stale.append(description)
        elif not killed:
            survivors.append((description, note))

    print(f"\n{len(MUTANTS) - len(survivors) - len(stale)}/{len(MUTANTS)} killed")

    # Reported apart from survivors, and first, because they are different
    # problems demanding different fixes. A survivor means the test suite has a
    # gap. A stale anchor means this harness has been testing *nothing* for that
    # behaviour, and calling it "no test asserts on this" sends the reader to
    # write a test that already exists.
    #
    # One went stale for four tasks: a list comprehension became a for loop, the
    # anchor stopped matching, and it reported as a survivor the whole time.
    if stale:
        print("\nSTALE ANCHORS — these mutants tested nothing; repair them:")
        for description in stale:
            print(f"  - {description}")

    if survivors:
        print("\nSURVIVORS — behaviour no test asserts on:")
        for description, note in survivors:
            print(f"  - {description}  [{note}]")

    return 1 if (survivors or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
