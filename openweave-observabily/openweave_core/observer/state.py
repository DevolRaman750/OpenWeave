"""Durable cursor for the observer so restarts don't re-analyze old traces.

State is a tiny JSON file: the timestamp of the newest trace we've processed
plus a bounded ring of recently-seen trace ids (to de-dupe traces that share
the boundary timestamp). Writes are atomic (temp file + os.replace) so a crash
mid-write can't corrupt the cursor.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Optional

_RING_SIZE = 2000


class ObserverState:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.last_timestamp: Optional[str] = None
        self._recent_ids: deque[str] = deque(maxlen=_RING_SIZE)
        self._recent_set: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.last_timestamp = data.get("last_timestamp")
        for tid in data.get("recent_ids", []):
            self._remember(str(tid))

    def _remember(self, trace_id: str) -> None:
        if trace_id in self._recent_set:
            return
        if len(self._recent_ids) == self._recent_ids.maxlen:
            evicted = self._recent_ids[0]
            self._recent_set.discard(evicted)
        self._recent_ids.append(trace_id)
        self._recent_set.add(trace_id)

    def seen(self, trace_id: str) -> bool:
        return trace_id in self._recent_set

    def mark_processed(self, trace_id: str, timestamp: Optional[str]) -> None:
        self._remember(trace_id)
        if timestamp and (self.last_timestamp is None or timestamp >= self.last_timestamp):
            self.last_timestamp = timestamp

    def save(self) -> None:
        payload = {
            "last_timestamp": self.last_timestamp,
            "recent_ids": list(self._recent_ids),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent) or ".", prefix=".obs_state_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
