"""EphemeralTraceStore: in-memory or encrypted-temp span buffer that never
survives the process. Cleanup is registered on atexit and on SIGINT/SIGTERM/
SIGHUP so a captured session snapshot (which may hold PII) doesn't outlive
the run.
"""
from __future__ import annotations

import atexit
import json
import os
import secrets
import signal
import tempfile
from pathlib import Path
from typing import Any


class EphemeralTraceStore:
    def __init__(self, mode: str = "memory", temp_dir: str | None = None):
        self.mode = mode
        self.spans: list[dict[str, Any]] = []
        self._cleaned = False
        self._prev_handlers: dict[int, Any] = {}
        self._cipher = None
        self._temp_path: Path | None = None
        self._temp_fh = None

        if mode == "encrypted_temp":
            from cryptography.fernet import Fernet

            self._key = Fernet.generate_key()
            self._cipher = Fernet(self._key)
            fd, path = tempfile.mkstemp(prefix="o2a_", suffix=".enc", dir=temp_dir)
            os.chmod(path, 0o600)
            self._temp_path = Path(path)
            self._temp_fh = os.fdopen(fd, "r+b")

        atexit.register(self.cleanup)
        for signame in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                prev = signal.signal(sig, self._signal_cleanup)
                self._prev_handlers[sig] = prev
            except (ValueError, OSError):
                pass

    def _signal_cleanup(self, signum, frame):
        self.cleanup()
        prev = self._prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def add(self, span: dict[str, Any]) -> None:
        if self._cleaned:
            return
        self.spans.append(span)
        if self.mode == "encrypted_temp" and self._cipher is not None:
            line = self._cipher.encrypt(json.dumps(span, default=str).encode()) + b"\n"
            self._temp_fh.write(line)
            self._temp_fh.flush()

    def read_all(self) -> list[dict]:
        return list(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def cleanup(self) -> dict:
        if self._cleaned:
            return {"already_clean": True}

        spans_purged = len(self.spans)
        for span in self.spans:
            span.clear()
        self.spans.clear()

        temp_shredded = False
        if self._temp_path is not None:
            try:
                if self._temp_fh is not None:
                    self._temp_fh.close()
                if self._temp_path.exists():
                    self._shred(self._temp_path)
                    temp_shredded = True
            except OSError:
                pass

        if self._cipher is not None:
            self._key = b"\x00" * len(self._key)
            self._cipher = None

        self._cleaned = True
        return {
            "spans_purged": spans_purged,
            "temp_shredded": temp_shredded,
            "mode": self.mode,
        }

    @staticmethod
    def _shred(path: Path) -> None:
        size = path.stat().st_size
        with open(path, "r+b") as f:
            f.write(secrets.token_bytes(size))
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            f.write(b"\x00" * size)
            f.flush()
            os.fsync(f.fileno())
        path.unlink()
