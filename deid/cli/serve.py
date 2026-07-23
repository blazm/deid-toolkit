"""`deid serve` — run the explore app with auto-reload on file changes.

Designed for tmux or any long-running terminal session.
Detects code changes and restarts the server automatically.

Usage:
    # Start in background tmux session:
    tmux new -d -s deid
    deid serve

    # Or start directly (blocks the terminal):
    deid serve --port 8501

    # Stop:
    tmux kill-session -t deid

    # Or kill the process:
    pkill -f 'streamlit.*deid/explore/app.py'
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import typer

# ── File watcher (optional dependency) ────────────────────
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

app = typer.Typer(
    name="serve",
    help=(
        "Run the explore app with auto-reload. "
        "Detects file changes and restarts the server. "
        "Run inside tmux for background operation."
    ),
)

# Directories relative to repo root to watch
WATCH_DIRS = [
    "deid/explore",
    "deid/evaluation",
    "deid/techniques",
    "root_dir/datasets/labels",
]


def _find_python() -> str:
    """Find the best python executable (conda/venv aware)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        p = Path(conda_prefix) / "python"
        if p.exists():
            return str(p)
    if os.environ.get("VIRTUAL_ENV"):
        p = Path(os.environ["VIRTUAL_ENV"]) / "python"
        if p.exists():
            return str(p)
    return sys.executable


def _kill(proc: Optional[subprocess.Popen]):
    """Safely terminate a subprocess, wait up to 3s."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _start_server(python: str, script: str, port: int) -> subprocess.Popen:
    """Start a streamlit server and return the Popen object."""
    cmd = [
        python, "-m", "streamlit", "run", script,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


# ───────── watchdog-based serve ────────────────────────────────

if HAS_WATCHDOG:

    class _ChangeDetector(FileSystemEventHandler):
        """Sets an event when a non-directory file change is detected."""

        def __init__(self):
            super().__init__()
            self._event = threading.Event()

        def on_any_event(self, event):
            if not event.is_directory:
                self._event.set()

        def consume(self, debounce: float = 0.3) -> bool:
            """Sleep `debounce` seconds to absorb batch saves, then return True
            if a change was detected (and clear it)."""
            self._event.clear()
            time.sleep(debounce)
            return self._event.is_set()

    def _serve(python: str, script: str, port: int):
        """Main serve loop with watchdog file watching and auto-restart."""

        root = Path(__file__).resolve().parent.parent.parent
        dirs = [str(root / d) for d in WATCH_DIRS if (root / d).exists()]

        if not dirs:
            print("Warning: no watch directories found.")
            _serve_loop_simple(python, script, port)
            return

        print(f"  Watching: {', '.join(Path(d).name for d in dirs)}")
        print()

        observer = Observer()
        detector = _ChangeDetector()
        for d in dirs:
            observer.schedule(detector, d, recursive=True)
        observer.start()

        try:
            proc: Optional[subprocess.Popen] = None

            while True:
                # Start a new streamlit server
                proc = _start_server(python, script, port)
                print(f"\n[{time.strftime('%H:%M:%S')}] Server running on http://localhost:{port}")

                # Drain stdout until server dies or files change
                while proc.poll() is None:
                    # Check for file changes
                    if detector.consume(debounce=0.3):
                        print(f"\n[{time.strftime('%H:%M:%S')}] Files changed — restarting...")
                        _kill(proc)
                        proc = None
                        break

                    # Drain buffered stdout
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        if line.strip():
                            print(line.rstrip())

                    time.sleep(0.15)
        except KeyboardInterrupt:
            print("\n[Ctrl+C] Stopping...")
        finally:
            _kill(proc)
            observer.stop()
            observer.join()


# ───────── simple serve (no watchdog) ───────────────────────────

def _serve_loop_simple(python: str, script: str, port: int):
    """Simple restart-on-exit loop (used when watchdog is not available or no dirs found)."""
    while True:
        proc = _start_server(python, script, port)
        print(f"\n[{time.strftime('%H:%M:%S')}] Server running on http://localhost:{port}")
        print("[No watchdog — will restart only on exit.]")

        for line in proc.stdout:
            if line.strip():
                print(line.rstrip())

        proc.wait()
        print(f"\n[{time.strftime('%H:%M:%S')}] Restarting in 2s...")
        time.sleep(2)


# ───────── CLI entry ────────────────────────────────────────────

@app.command()
def main(
    port: int = typer.Option(8501, "-p", "--port", help="Streamlit server port."),
):
    root = Path(__file__).resolve().parent.parent.parent
    script = str(root / "deid" / "explore" / "app.py")

    if not Path(script).exists():
        typer.echo(f"Error: {script} not found.", fg="red")
        raise typer.Exit(1)

    python = _find_python()

    watchdog_ok = HAS_WATCHDOG
    if not watchdog_ok:
        typer.echo(
            "Warning: watchdog not installed. Using fallback loop (restarts only on exit).",
            fg="yellow",
        )
        typer.echo("  Install it: pip install watchdog", fg="yellow")

    print("=== DeID Explore — Auto-Reload Server ===")
    print(f"  Python  : {python}")
    print(f"  Port    : {port}")
    print(f"  Watchdog: {'enabled' if watchdog_ok else 'disabled'}")
    print("  Press Ctrl+C to stop.")
    print()

    if HAS_WATCHDOG:
        _serve(python, script, port)
    else:
        _serve_loop_simple(python, script, port)

    print("Stopped.")
