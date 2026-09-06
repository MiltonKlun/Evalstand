# A VHS script for the README's demo GIF (task 6.8).
#
#     cd examples/demo && vhs demo.tape
#
# Produces ../../docs/demo.gif. VHS drives a real terminal and records what it
# draws, so nothing here is staged: every frame is the tool actually running.
#
# The demo eval calls no provider — it answers from a dict and sleeps — so the
# recording costs nothing, needs no key, and does not drift as models change.
# What it demonstrates is the interface: rows landing one at a time, a trace
# tree, and a re-run triggered by an edit.
#
# Under 30 seconds and 5 MB, as the plan requires. If it outgrows either, cut
# the pauses rather than the content — a GIF that stops before the watch-mode
# re-run has dropped the one thing that is hard to explain in prose.

Output ../../docs/demo.gif

Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 12
Set TypingSpeed 45ms

# A clean slate, so a previous recording's edit does not leave the demo
# starting from the fixed answer and never showing a failure.
Hide
Type "git checkout capitals_eval.py 2>/dev/null; clear"
Enter
Show

Type "evalstand watch"
Enter

# Rows land one at a time. This pause is the whole point of the recording: a
# viewer should see the table fill, not find it already full.
Sleep 5s

# One case in full, with the trace tree.
Down
Down
Enter
Sleep 3s
Escape
Sleep 1s

# Only what needs attention.
Type "f"
Sleep 2s
Type "f"
Sleep 1s

# Watch mode: fix the wrong answer from another shell, and the eval re-runs
# on its own. `q` is never pressed before this — the re-run has to happen in
# the live view for the demo to mean anything.
Hide
Type "sed -i 's/\"peru\": \"Cusco\"/\"peru\": \"Lima\"/' capitals_eval.py"
Enter
Show
Sleep 5s

Type "q"
Sleep 1s
