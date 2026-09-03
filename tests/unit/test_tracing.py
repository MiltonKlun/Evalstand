"""Tracing (task 3.2).

The risk register names this the feature that decides whether the tool is real
or a wrapper, and names `contextvars` under `asyncio.gather` as the specific
thing to test early. So these tests exercise the parenting rule against real
concurrency rather than a simulation of it.

ADR 0006: a ContextVar holds the currently open node; entering sets it and keeps
the token, leaving resets the token in a `finally`. When a parent cannot be
determined confidently the node attaches to the root — a visible orphan is
honest, a wrongly-parented node is a plausible-looking lie.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from evalstand.models import Trace
from evalstand.tracing import (
    TraceCollector,
    current_collector,
    record_call,
    trace,
)


def _tree(traces: list[Trace]) -> dict[str | None, list[str]]:
    """Children by parent id, for asserting on shape rather than order."""
    children: dict[str | None, list[str]] = {}
    for node in traces:
        children.setdefault(node.parent_id, []).append(node.name)
    return {parent: sorted(names) for parent, names in children.items()}


def _named(traces: list[Trace], name: str) -> Trace:
    return next(node for node in traces if node.name == name)


class TestCollectorLifecycle:
    def test_collects_nothing_outside_a_collector(self) -> None:
        """A trace() outside a running case must not raise; it simply has
        nowhere to go. A user experimenting in a REPL should not hit an error."""
        with trace("orphan"):
            pass  # no collector active

    def test_a_collector_is_active_only_within_its_scope(self) -> None:
        assert current_collector.get() is None
        with TraceCollector() as collector:
            assert current_collector.get() is collector
        assert current_collector.get() is None

    def test_nested_collectors_do_not_leak(self) -> None:
        with TraceCollector() as outer:
            with TraceCollector() as inner:
                assert current_collector.get() is inner
            assert current_collector.get() is outer


class TestSyncNesting:
    def test_a_single_span_is_a_root(self) -> None:
        with TraceCollector() as collector, trace("task"):
            pass
        assert _tree(collector.traces) == {None: ["task"]}

    def test_a_nested_span_points_at_its_parent(self) -> None:
        with TraceCollector() as collector, trace("task"), trace("judge"):
            pass

        task = _named(collector.traces, "task")
        judge = _named(collector.traces, "judge")
        assert task.parent_id is None
        assert judge.parent_id == task.id

    def test_three_levels_nest(self) -> None:
        """Task 3.2's acceptance shape: a task making nested calls."""
        with TraceCollector() as collector, trace("task"), trace("retrieve"):
            with trace("embed"):
                pass

        task = _named(collector.traces, "task")
        retrieve = _named(collector.traces, "retrieve")
        embed = _named(collector.traces, "embed")
        assert retrieve.parent_id == task.id
        assert embed.parent_id == retrieve.id

    def test_siblings_share_a_parent(self) -> None:
        with TraceCollector() as collector, trace("task"):
            with trace("first"):
                pass
            with trace("second"):
                pass

        task = _named(collector.traces, "task")
        assert _named(collector.traces, "first").parent_id == task.id
        assert _named(collector.traces, "second").parent_id == task.id

    def test_a_raising_span_does_not_corrupt_its_siblings(self) -> None:
        """The `finally` reset is what makes this hold — without it the next
        sibling parents to a node that already closed."""
        with TraceCollector() as collector, trace("task"):
            with pytest.raises(ValueError), trace("boom"):
                raise ValueError("inner failure")
            with trace("after"):
                pass

        task = _named(collector.traces, "task")
        assert _named(collector.traces, "boom").parent_id == task.id
        assert _named(collector.traces, "after").parent_id == task.id

    def test_a_raising_span_is_still_recorded(self) -> None:
        """A call that failed still cost time and possibly money."""
        with TraceCollector() as collector, pytest.raises(ValueError), trace("boom"):
            raise ValueError("failed")

        boom = _named(collector.traces, "boom")
        assert boom.output is not None
        assert "ValueError" in str(boom.output)


class TestAsyncNesting:
    """The case the risk register singles out."""

    @pytest.mark.anyio
    async def test_gathered_siblings_share_a_parent(self) -> None:
        """contextvars copy per task, so siblings under gather parent correctly
        with no special handling. This is the claim ADR 0006 rests on."""

        async def child(name: str) -> None:
            with trace(name):
                await asyncio.sleep(0.001)

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(child("a"), child("b"), child("c"))

        task = _named(collector.traces, "task")
        for name in ("a", "b", "c"):
            assert _named(collector.traces, name).parent_id == task.id

    @pytest.mark.anyio
    async def test_gathered_siblings_nest_independently(self) -> None:
        """Each gathered branch keeps its own chain; they must not interleave."""

        async def branch(name: str) -> None:
            with trace(name):
                await asyncio.sleep(0.001)
                with trace(f"{name}-inner"):
                    await asyncio.sleep(0.001)

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(branch("x"), branch("y"))

        for name in ("x", "y"):
            outer = _named(collector.traces, name)
            assert _named(collector.traces, f"{name}-inner").parent_id == outer.id

    @pytest.mark.anyio
    async def test_a_raising_branch_does_not_corrupt_the_others(self) -> None:
        async def ok(name: str) -> None:
            with trace(name):
                await asyncio.sleep(0.002)

        async def bad() -> None:
            with trace("bad"):
                await asyncio.sleep(0.001)
                raise ValueError("branch failed")

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(ok("a"), bad(), ok("b"), return_exceptions=True)

        task = _named(collector.traces, "task")
        for name in ("a", "b", "bad"):
            assert _named(collector.traces, name).parent_id == task.id

    @pytest.mark.anyio
    async def test_deep_concurrency_keeps_every_node_reachable(self) -> None:
        """The property that matters for the TUI: the forest can be walked."""

        async def branch(index: int) -> None:
            with trace(f"b{index}"), trace(f"b{index}-call"):
                await asyncio.sleep(0.001)

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(*(branch(i) for i in range(20)))

        ids = {node.id for node in collector.traces}
        for node in collector.traces:
            assert node.parent_id is None or node.parent_id in ids
        assert len([n for n in collector.traces if n.parent_id is None]) == 1


class TestLostContext:
    """ADR 0006's fallback: attach to the root rather than guess.

    A raw ThreadPoolExecutor does not copy context, so a call made inside one
    has no parent to find. That is a real scenario — a user's task may hand work
    to a thread pool — and the honest answer is a visible orphan.
    """

    @pytest.mark.anyio
    async def test_a_call_from_an_uncopied_thread_attaches_to_the_root(self) -> None:
        collector = TraceCollector()
        with collector, trace("task"):
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool, lambda: collector.record_orphan("from-thread", duration_ms=1)
                )

        orphan = _named(collector.traces, "from-thread")
        assert orphan.parent_id is None, "an unknown parent must not be guessed"

    @pytest.mark.anyio
    async def test_asyncio_to_thread_keeps_its_parent(self) -> None:
        """to_thread copies context, so this one nests correctly. Worth pinning:
        it is the difference between the two thread paths."""

        def blocking() -> None:
            with trace("blocking"):
                pass

        with TraceCollector() as collector, trace("task"):
            await asyncio.to_thread(blocking)

        task = _named(collector.traces, "task")
        assert _named(collector.traces, "blocking").parent_id == task.id


class TestRecordedFields:
    def test_records_duration(self) -> None:
        with TraceCollector() as collector, trace("task"):
            pass
        assert _named(collector.traces, "task").duration_ms >= 0

    def test_records_a_start_time(self) -> None:
        with TraceCollector() as collector, trace("task"):
            pass
        assert _named(collector.traces, "task").started_at is not None

    def test_a_recorded_call_carries_model_tokens_and_cost(self) -> None:
        """Automatic capture from llm.py: the fields the TUI's trace view shows."""
        with TraceCollector() as collector, trace("task"):
            record_call(
                name="chat",
                model="gpt-4o-mini",
                input_tokens=12,
                output_tokens=3,
                cost_usd=0.00012,
                duration_ms=120,
                input=[{"role": "user", "content": "hi"}],
                output="hello",
            )

        call = _named(collector.traces, "chat")
        assert call.model == "gpt-4o-mini"
        assert call.input_tokens == 12
        assert call.output_tokens == 3
        assert call.cost_usd == pytest.approx(0.00012)
        assert call.duration_ms == 120

    def test_a_recorded_call_nests_under_the_open_span(self) -> None:
        with TraceCollector() as collector, trace("task"):
            record_call(name="chat", model="m", duration_ms=1)

        assert _named(collector.traces, "chat").parent_id == _named(collector.traces, "task").id

    def test_recording_outside_a_collector_is_a_no_op(self) -> None:
        record_call(name="chat", model="m", duration_ms=1)


class TestCostRollup:
    def test_node_costs_sum_to_the_case_total(self) -> None:
        """Task 3.2's acceptance: the sum of node costs equals the case total."""
        with TraceCollector() as collector, trace("task"):
            record_call(name="a", model="m", duration_ms=1, cost_usd=0.01)
            with trace("judge"):
                record_call(name="b", model="m", duration_ms=1, cost_usd=0.02)

        assert collector.total_cost_usd == pytest.approx(0.03)

    def test_unpriced_calls_do_not_count_as_free(self) -> None:
        """Consistent with llm.py: unknown is not zero."""
        with TraceCollector() as collector, trace("task"):
            record_call(name="a", model="m", duration_ms=1, cost_usd=None)

        assert collector.total_cost_usd is None

    def test_a_mix_of_priced_and_unpriced_is_reported_as_partial(self) -> None:
        """Summing only the priced calls would understate the real spend."""
        with TraceCollector() as collector, trace("task"):
            record_call(name="a", model="m", duration_ms=1, cost_usd=0.01)
            record_call(name="b", model="m", duration_ms=1, cost_usd=None)

        assert collector.total_cost_usd is None
        assert collector.priced_cost_usd == pytest.approx(0.01)
        assert collector.unpriced_call_count == 1

    def test_token_totals_roll_up(self) -> None:
        with TraceCollector() as collector, trace("task"):
            record_call(name="a", model="m", duration_ms=1, input_tokens=10, output_tokens=5)
            record_call(name="b", model="m", duration_ms=1, input_tokens=20, output_tokens=7)

        assert collector.total_input_tokens == 30
        assert collector.total_output_tokens == 12


class TestForestValidity:
    def test_collected_traces_form_a_valid_forest(self) -> None:
        """The Result model rejects cycles and dangling parents, so anything the
        collector produces must satisfy it or a run cannot be persisted."""
        from evalstand.models import Result

        with TraceCollector() as collector, trace("task"):
            with trace("a"):
                record_call(name="a-call", model="m", duration_ms=1)
            with trace("b"):
                pass

        Result(id="r1", case_id="q1", output="x", traces=collector.traces)

    @pytest.mark.anyio
    async def test_a_concurrent_forest_is_also_valid(self) -> None:
        from evalstand.models import Result

        async def branch(index: int) -> None:
            with trace(f"b{index}"):
                record_call(name=f"call{index}", model="m", duration_ms=1)

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(*(branch(i) for i in range(10)))

        Result(id="r1", case_id="q1", output="x", traces=collector.traces)


class TestAdversarialConcurrency:
    """The risk register's question: does this hold under real load?

    Each scenario must produce a valid forest — one root, no dangling parents,
    accepted by the Result model — because a run that cannot be persisted is a
    run that cannot be compared.
    """

    @staticmethod
    def _assert_valid_forest(collector: TraceCollector) -> None:
        from evalstand.models import Result

        ids = {node.id for node in collector.traces}
        dangling = [n.name for n in collector.traces if n.parent_id and n.parent_id not in ids]
        assert dangling == [], f"dangling parents: {dangling}"

        roots = [n for n in collector.traces if n.parent_id is None]
        assert len(roots) == 1, f"expected one root, got {len(roots)}"

        Result(id="r1", case_id="q1", output="x", traces=collector.traces)

    @pytest.mark.anyio
    async def test_a_cancelled_task_does_not_corrupt_the_tree(self) -> None:
        with TraceCollector() as collector, trace("task"):
            pending = asyncio.create_task(asyncio.sleep(10))
            await asyncio.sleep(0.001)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        self._assert_valid_forest(collector)

    @pytest.mark.anyio
    async def test_a_timeout_mid_span_still_closes_it(self) -> None:
        """A span abandoned by a timeout must still be recorded and closed, or
        everything after it inherits a parent that never finished."""
        with TraceCollector() as collector, trace("task"), pytest.raises(TimeoutError):
            async with asyncio.timeout(0.005):
                with trace("slow"):
                    await asyncio.sleep(1)

        self._assert_valid_forest(collector)
        assert _named(collector.traces, "slow").parent_id == _named(collector.traces, "task").id

    def test_deep_recursion_nests_correctly(self) -> None:
        def recurse(depth: int) -> None:
            if depth == 0:
                return
            with trace(f"d{depth}"):
                recurse(depth - 1)

        with TraceCollector() as collector, trace("task"):
            recurse(50)

        self._assert_valid_forest(collector)
        assert len(collector.traces) == 51

    @pytest.mark.anyio
    async def test_a_wide_gather_keeps_every_node_parented(self) -> None:
        async def leaf(index: int) -> None:
            with trace(f"l{index}"):
                record_call(name=f"c{index}", model="m", duration_ms=1)

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(*(leaf(i) for i in range(200)))

        self._assert_valid_forest(collector)
        assert len(collector.traces) == 401

    @pytest.mark.anyio
    async def test_nested_gathers_keep_their_branches_separate(self) -> None:
        """The shape that would expose interleaving: concurrency inside
        concurrency."""

        async def inner(outer_index: int, index: int) -> None:
            with trace(f"i{outer_index}-{index}"):
                await asyncio.sleep(0.001)

        async def outer(index: int) -> None:
            with trace(f"o{index}"):
                await asyncio.gather(*(inner(index, j) for j in range(5)))

        with TraceCollector() as collector, trace("task"):
            await asyncio.gather(*(outer(i) for i in range(5)))

        self._assert_valid_forest(collector)

        for outer_index in range(5):
            parent = _named(collector.traces, f"o{outer_index}")
            for j in range(5):
                child = _named(collector.traces, f"i{outer_index}-{j}")
                assert child.parent_id == parent.id, "a branch adopted another branch's child"
