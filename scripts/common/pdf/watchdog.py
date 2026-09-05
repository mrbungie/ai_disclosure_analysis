"""
scripts/common/pdf/watchdog.py — a per-page deadline for the VLM backends.

WHY. A corpus run is ~120,000 VLM pages (see docs/analytics/
pdf-backend-poc.md). While measuring that, a PaddleOCR-VL call hung
indefinitely on a single page — model loaded, GPU at 0%, no error, no
progress, for over 30 minutes until it was killed. It was not the page:
the same page went through fine on the next attempt. Whatever the cause
(a wedged CUDA context left by a previously killed process is the best
guess), the failure mode is what matters — a run that hangs silently at
page 3,000 of 120,000 is worse than one that fails, because nothing
downstream can tell the difference between "still working" and "dead".

So every VLM page call gets a deadline. A page that blows it is recorded
as a failure and the run moves on, which turns an unbounded hang into one
lost page and a log line.

SIGALRM, not a thread or a subprocess: the VLM backends declare
max_workers=1 and run inline on the main thread (see
scripts/cl/02_extract_text.py's _iter_results), which is exactly where
signal-based timeouts work. `page_deadline` no-ops off the main thread
rather than raising, so the same backend still works if someone later
runs it somewhere signals aren't available — it just loses the guard.
"""

import contextlib
import signal
import threading


class PageTimeout(Exception):
    """One page exceeded its deadline. Caught per page, never fatal."""


@contextlib.contextmanager
def page_deadline(seconds: int):
    """Raise PageTimeout if the block hasn't finished in `seconds`."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(_signum, _frame):
        raise PageTimeout(f"page exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    # setitimer, not alarm(): alarm() truncates to whole seconds and the
    # deadline is short enough that rounding matters.
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
