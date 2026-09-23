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
#
# Recorded in PowerShell, because on Windows `bash` cannot be made to mean Git
# Bash here: process creation searches System32 *before* PATH, so the name
# always resolves to `system32\bash.exe` — the WSL launcher, a separate Linux
# with no access to this project's virtualenv. Putting Git Bash first on PATH
# does not help. The first attempt captured a WSL error screen and exited 0.
# `evalstand` must be on the PATH the recording inherits.
#
# On macOS or Linux, set the shell to bash and use this as the hidden setup:
#   git checkout capitals_eval.py; (sleep 15; sed -i 's/"peru": "Cusco"/"peru": "Lima"/' capitals_eval.py) & clear

Output ../../docs/demo.gif

Set Shell "powershell"
Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 12
Set TypingSpeed 45ms

# A clean slate, so a previous recording's edit does not leave the demo
# starting from the fixed answer and never showing a failure.
#
# The fix is scheduled here, as a background job, rather than typed later. An
# earlier version of this tape typed the edit while `evalstand watch` held the
# terminal — so the keystrokes went to the live view, not to a shell, the file
# never changed, and the re-run this recording exists to show never happened.
# A second terminal is what a user would really use; a timed job is the same
# thing without one.
#
# The prompt is replaced with a bare `>` so the recording does not publish the
# recording machine's home directory in the README.
Hide
Type `function prompt { '> ' }; git checkout capitals_eval.py 2>$null; $f = "$PWD\capitals_eval.py"; $null = Start-Job { param($f) while (-not (Get-Process evalstand -ErrorAction SilentlyContinue)) { Start-Sleep -Milliseconds 200 }; Start-Sleep 17; (Get-Content $f) -replace '"peru": "Cusco"', '"peru": "Lima"' | Set-Content $f } -ArgumentList $f; Clear-Host`
Enter
Show

Type "evalstand watch"
Enter

# Start-up is cut, and this is the one place the recording is edited. The live
# view takes several seconds to appear — importing litellm alone is over three,
# even for this eval, which never calls a model — and a GIF that spent a third
# of its length on a blank screen would show nothing. The delay is real and is
# recorded in examples/demo/README.md rather than hidden by omission.
#
# Waiting on the screen rather than a fixed time is also what makes the keys
# below land. The first recording pressed Down/Enter at a fixed five seconds,
# before the live view existed, and the trace tree never appeared.
Hide
Wait+Screen@30s /capitals/
Show

# Rows land one at a time. This pause is the whole point of the recording: a
# viewer should see the table fill, not find it already full.
Sleep 3s

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

# The scheduled edit lands about now, and the eval re-runs on its own inside
# the live view: peru flips from fail to pass without anyone pressing a key.
# Waited for rather than slept through, so the recording cannot end before the
# re-run it exists to show.
Wait+Screen@30s /passed\s+6\/6/
Sleep 2s

Type "q"
Sleep 1s
