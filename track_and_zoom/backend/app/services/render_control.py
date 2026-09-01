"""Cooperative cancellation for the long on-demand renders.

Cancellation started out riding the progress callback, which needed no service
signature changes — but it only stops a render at the moments a service happens
to report progress, and both render pipelines spend real time in stretches that
report nothing:

  * sam2_tracker._extract_frames  — decodes the WHOLE video to JPEGs (ffmpeg)
  * SAM2's init_state             — preprocesses every extracted frame
  * follow_zoom._mux_with_audio   — ffmpeg remux/encode of the finished video
  * supervision._reencode_to_h264 — ffmpeg H.264 pass over the finished video
  * any frame loop whose progress hook is guarded by an unknown frame count

Clicking cancel inside one of those did nothing at all, sometimes for minutes.
So the hook is now explicit and threaded into the services: `should_cancel` is
checked directly in every loop, and the ffmpeg steps run under
`run_cancellable`, which kills the child process instead of waiting it out.

`check()` raises rather than returning a flag on purpose: the render pipelines
are deep call stacks, and an exception unwinds them through their existing
`finally` blocks — releasing the video capture, the writer, and the temp frame
directory — which a returned bool would have to be plumbed through by hand at
every level, with one missed branch leaking a file handle.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from typing import Callable, Optional, Sequence

log = logging.getLogger("vision.render_control")

ShouldCancel = Optional[Callable[[], bool]]


class RenderCanceled(Exception):
    """The operator stopped this render. A normal outcome, not a failure —
    callers report it as `canceled` and discard the partial output."""


def check(should_cancel: ShouldCancel) -> None:
    """Raise RenderCanceled if cancellation has been requested. Cheap enough to
    call every frame; `None` means "not cancellable" and is a no-op."""
    if should_cancel is not None and should_cancel():
        raise RenderCanceled()


def run_cancellable(cmd: Sequence[str], should_cancel: ShouldCancel, *,
                    poll_s: float = 0.2, timeout: float | None = None) -> None:
    """Run a subprocess that can be stopped part-way.

    `subprocess.run(check=True)` blocks until the child exits, so a cancel
    arriving during a multi-minute ffmpeg pass was simply ignored. This polls
    instead and terminates the child the moment cancellation is requested —
    SIGTERM first so ffmpeg can close its output file, SIGKILL if it ignores
    that.

    Raises RenderCanceled if stopped, CalledProcessError on a genuine non-zero
    exit, so callers keep the failure semantics they already had.
    """
    # stderr goes to a FILE, never a pipe. A pipe has a ~64KB kernel buffer, and
    # this loop polls without reading it — so a chatty child (ffmpeg logs every
    # frame) fills the buffer and blocks forever in pipe_write, with the job
    # frozen at whatever percentage it had reached. `subprocess.run` avoided
    # this only because communicate() drains concurrently; this loop does not,
    # so the buffer has to be somewhere unbounded.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf)
        started = time.monotonic()
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if should_cancel is not None and should_cancel():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    raise RenderCanceled()
                if timeout is not None and (time.monotonic() - started) > timeout:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise subprocess.TimeoutExpired(list(cmd), timeout)
                time.sleep(poll_s)
        finally:
            # Never leave a child running behind an exception unwinding past here.
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        if rc != 0:
            errf.seek(0)
            err = errf.read()[-2000:]
            raise subprocess.CalledProcessError(rc, list(cmd), stderr=err)
