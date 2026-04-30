#!/bin/bash
#
# ⚠ AUTO-EXECUTES AT SESSIONSTART — TREAT CHANGES AS CODE-REVIEW ⚠
#
# This script runs with the user's full privileges every time a Claude
# session boots in this repo. Any modification — by Claude, by a tool,
# by a compromised process — executes on the next session BEFORE the
# user has any opportunity to review.
#
# Hardening rules (do not relax):
#   - No `eval`, `source`, or shell-substitution of file content.
#   - No network calls (curl, wget, ssh, git remote ops).
#   - No commands beyond `stat`, `touch`, `cat`, `grep`, `head`, `date`,
#     `echo`, `cd`, `pwd`, basic arithmetic. (`cd`/`pwd` are needed for
#     the PLUGIN_DIR resolution.) If you want more, write a separate
#     script under internal/scripts/ that the user invokes explicitly.
#   - Keep total length short and audit-able (target: <100 lines).
#
# check_memo_inbox.sh — SessionStart hook: notifies Claude if a memo
# from the sister-repo Claude has been written or updated since the
# last time this side acknowledged it. Also surfaces a one-line
# reminder of the last outbound memo for cold-start context.
#
# Convention (established 2026-04-26):
#   - Inbound memo: internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md
#     (sage-fork's claude writes here when they have a message).
#   - Outbound memo: coderef/sage-fork/internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md
#     (we write here when we have a message for them).
#   - Last-seen marker: internal/.memo_inbox_seen_at — mtime tracks
#     when this side last processed the inbound memo.
#
# Behavior:
#   - Inbound memo's mtime > marker (or marker missing) → print
#     notification + auto-ack (touch marker). One-shot per memo update.
#   - Otherwise: print one-line reminder of the outbound memo's mtime
#     + first content line, so a cold-start session sees what we last
#     said without having to Read() the file. Silent on a fresh clone
#     with no outbound memo yet.
#
# Output goes to stderr so it lands in Claude's SessionStart context
# without polluting stdout.
#
# Sister-repo mirror: sage-fork's claude maintains the symmetric hook
# at coderef/sage-fork/internal/scripts/check_memos.sh. Same shape;
# paths inverted.
#
# SIGPIPE-under-pipefail safety: any read that pipes to `head` is
# wrapped with `|| true`, AND the marker touch happens BEFORE any
# pipe-prone read in the new-memo branch. This avoids the failure
# mode where grep|head SIGPIPEs, set -e aborts the script, and the
# marker never updates — looping the same notification every session.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INBOX="$PLUGIN_DIR/internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md"
OUTBOX="$PLUGIN_DIR/coderef/sage-fork/internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md"
MARKER="$PLUGIN_DIR/internal/.memo_inbox_seen_at"

# Local helpers: extract a short, sanitized first content line for
# preview output. Used in both the new-inbound branch and the
# outbound-reminder fallthrough. The grep regex anchors to lines
# starting with an alphanumeric or `*` (skips blank lines + heading
# delimiters); head -c 120 caps payload size; `|| true` absorbs the
# SIGPIPE-under-pipefail risk that head closing early would otherwise
# trigger.
_first_line() { grep -m1 -E '^[A-Za-z0-9*]' "$1" 2>/dev/null | head -c 120 || true; }
_fmt_ts() { date -d "@$1" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "unknown"; }

inbox_mtime=0
[[ -f "$INBOX" ]] && inbox_mtime=$(stat -c %Y "$INBOX")

marker_mtime=0
[[ -f "$MARKER" ]] && marker_mtime=$(stat -c %Y "$MARKER")

# inbox_mtime defaults to 0 when INBOX is missing, so the comparison
# alone correctly suppresses the branch — no separate -f guard needed.
if [[ "$inbox_mtime" -gt "$marker_mtime" ]]; then
    # Touch FIRST so even if the content read below fails under
    # pipefail, the marker update has already landed and we won't
    # re-notify the same memo every session.
    touch "$MARKER"

    echo >&2
    echo "📬 New memo from sage-fork claude (modified $(_fmt_ts "$inbox_mtime")):" >&2
    echo "   $INBOX" >&2
    first_content=$(_first_line "$INBOX")
    [[ -n "$first_content" ]] && echo "   first line: $first_content" >&2
    echo "   Read: Read(\"internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md\")" >&2
    echo "   Re-ack (rare): bash internal/scripts/ack_memo.sh" >&2
    exit 0
fi

# No new inbound. Surface a low-cost reminder of what we last said
# (outbound memo) so a cold-start session has the recent context
# without an extra Read(). Silent if outbound doesn't exist yet.
if [[ -f "$OUTBOX" ]]; then
    out_mtime=$(stat -c %Y "$OUTBOX")
    echo >&2
    echo "ℹ  Last outbound to sage-fork claude (sent $(_fmt_ts "$out_mtime")):" >&2
    out_first=$(_first_line "$OUTBOX")
    [[ -n "$out_first" ]] && echo "   $out_first" >&2
fi

exit 0
