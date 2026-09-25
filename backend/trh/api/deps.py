from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Request

from trh.db import store


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = store.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()
