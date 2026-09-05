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
