# Vendored assets

## `htmx.min.js`

- **Version:** 2.0.4
- **Source:** <https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js>
- **SHA-256:** `e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447`
- **Licence:** BSD Zero Clause (0BSD)

Vendored rather than loaded from a CDN because `serve` is a local-first
command. A user runs it on a laptop, sometimes offline, often on a network that
cannot reach unpkg — and a page whose interactivity depends on an outbound
request degrades to a static table without saying so.

`test_web_page.py` asserts the digest, so replacing this file with something
else fails the suite rather than silently shipping it.

To update: fetch the new version, record its digest here and in the test, and
check the page still swaps fragments.
