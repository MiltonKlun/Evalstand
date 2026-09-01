# 0002 — Design scope

- **Status:** proposed (blocked on task 0.6)
- **Date:** 2026-09-01

## Context

`evalstand` is an independent Python implementation of an idea proven in
TypeScript by `evalite` (MIT): that local LLM evaluation should feel like
running a test suite.

This ADR records which behaviours `evalstand` adopts, which it does
differently, and why. **It cannot be completed until task 0.6 is done** —
studying the reference implementation's entry function, CLI, SQLite layer, and
web UI. Filling it in before that would be guesswork.

Two decisions are already settled and recorded here:

## Decision (partial)

1. **Textual TUI instead of a React web UI.** The reference ships a React web
   app. `evalstand` ships a terminal UI first, with a web UI as an optional
   stretch phase. Rationale: the TUI reaches functional parity faster, keeps
   the entire stack in Python, and demos well in a terminal recording.

2. **pytest instead of Vitest** as the collection and execution substrate,
   since that is the Python equivalent and gives bare-`pytest` compatibility
   for free.

3. **Study for design understanding, not code.** The reference is read to
   understand behaviour and interface decisions. No code is copied. Where
   TypeScript idiom does not carry into Python, the Python equivalent wins.

## Consequences

To be completed alongside the rest of this ADR after task 0.6.
