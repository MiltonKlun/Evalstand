# The web UI

`evalstand serve` opens a browser view of your run history.

```bash
pip install 'evalstand[web]'
evalstand serve
```

It prints the address it bound and how many runs it found:

```
evalstand on http://127.0.0.1:8420  (42 runs)
ctrl-c to stop
```

## What it shows

A list of past runs on the left; click one to see its cases, scores and
totals. The same numbers `evalstand show` prints, rendered by the same code —
the page computes nothing of its own, so it cannot drift from the terminal
([ADR 0009](adr/0009-web-ui-is-optional-and-read-only.md)).

## It does not run anything

`serve` reads history. It has no button that starts an eval, and no endpoint
that will.

Evals cost money. A page that spent it on load — or that let a stray click
spend it — would be a trap, especially one left open in a background tab.
`evalstand run` and `evalstand watch` remain the ways to execute.

## It binds localhost

By default the server is reachable only from the machine it runs on.

That is deliberate rather than cautious: the database holds every prompt and
completion your evals sent and received, which routinely includes whatever your
task put in them. `--host 0.0.0.0` exposes all of that to your network, so it is
a decision you make rather than a default you inherit.

```bash
evalstand serve --host 0.0.0.0 --port 9000   # reachable from the network
```

## Options

| Flag | What it does |
| --- | --- |
| `--host` | Interface to bind. `127.0.0.1` by default. |
| `--port`, `-p` | Port to listen on. `8420` by default. |
| `--db` | Serve a database other than the project's own. |
| `--open` | Open a browser once the server is listening. |

## The JSON API

Every view has a JSON endpoint behind it, so you can script against history
without parsing a terminal table.

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | Server version, schema version, run count. |
| `GET /api/evals` | Every eval name in history. |
| `GET /api/runs` | Recent runs, newest first. `?name=` and `?limit=`. |
| `GET /api/runs/{id}` | One run in full: results, scores, traces. |
| `GET /api/runs/{id}/results/{case_id}` | One result. `?repeat=` for a repeat. |
| `GET /api/batches/{id}` | A batch and the runs beneath it. |

### Reading the numbers honestly

Two fields exist because a plain number would mislead.

**`total_cost_usd` is a lower bound, not a total.** It sums only the calls that
could be priced. `cost_is_complete` says whether that is the whole cost, and
`unpriced_call_count` says how many calls are missing from it. A consumer that
sums `total_cost_usd` alone will understate real spend while looking
authoritative.

**`null` never means zero.** A `null` mean is a run nothing scored, not a run
that scored zero. A `null` `cache_hit_rate` is a run that made no calls, not one
with a 0% hit rate. The terminal renders these as `-`; JSON has a real null and
uses it.

`/api/runs` also excludes cancelled batches, whose aggregates describe a subset
of the cases. `/api/batches/{id}` will serve one if you ask for it by id, with
`is_comparable: false` attached.

## The live table

When an eval is running, finished cases stream into the open page over
server-sent events. The connection stays open with nothing running, so a
browser never misses the first results of a run while reconnecting.

A browser that stops reading — a paused background tab, a slow connection —
fills a bounded queue and starts losing the *oldest* events rather than the
newest. A live table that falls further behind the longer it runs is worse than
one that skips. Nothing a browser does can slow down or break the eval itself.

## Offline

htmx is shipped inside the package rather than fetched from a CDN, so the page
works on a laptop with no network. Nothing on it makes an outbound request.
