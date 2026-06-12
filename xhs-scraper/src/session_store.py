"""
session_store.py — 会话与进度存储

跟踪每次「微会话」的时间和行为记录，供调度器做频率判断，
以及 AI 决策引擎读取采集进度。

用法:
    store = SessionStore()              # 默认 archive.db
    store = SessionStore(":memory:")    # 测试用
    if store.should_run_now():
        sid = store.log_session_start()
        # ... 执行采集 ...
        store.log_session_end(sid, actions=2, notes_opened=1, keyword="牙疼")
"""

import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta


SESSION_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    actions      INTEGER DEFAULT 0,
    notes_opened INTEGER DEFAULT 0,
    keyword      TEXT,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS keyword_progress (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword      TEXT NOT NULL,
    note_id      TEXT NOT NULL UNIQUE,
    collected_at TEXT NOT NULL
);
"""


def _now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _today_start() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


class SessionStore:
    """会话与进度存储。线程不安全 — 单进程使用。"""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "输出数据" / "archive.db"
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(SESSION_TABLES_SQL)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # 频率判断
    # ------------------------------------------------------------------

    def should_run_now(self, min_gap_hours: float = 3.0, max_daily: int = 4) -> bool:
        """频率判断：睡眠时间 / 最小间隔 / 每日上限。

        注意：随机跳过由编排层（xhs_accumulator）负责，保持本层可预测。
        """
        now = datetime.now()

        # ① 睡眠时间跳过（0:00-7:00）
        if 0 <= now.hour < 7:
            return False

        # ② 周末降低频率
        effective_max = max_daily
        if now.weekday() >= 5:  # 周六日
            effective_max = max(effective_max - 2, 1)

        # ③ 每日上限
        if self.get_today_session_count() >= effective_max:
            return False

        # ④ 最小间隔
        last_time = self.get_last_session_time()
        if last_time is not None:
            gap = (now - last_time).total_seconds() / 3600
            if gap < min_gap_hours:
                return False

        return True

    def get_last_session_time(self) -> datetime | None:
        cur = self._conn.execute(
            "SELECT started_at FROM sessions ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            return datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        return None

    def get_today_session_count(self) -> int:
        today = _today_start().strftime('%Y-%m-%d %H:%M:%S')
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE started_at >= ?", (today,)
        )
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def log_session_start(self) -> int:
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (_now_str(),)
        )
        self._conn.commit()
        return cur.lastrowid

    def log_session_end(self, session_id: int, *,
                        actions: int = 0, notes_opened: int = 0,
                        keyword: str = '', note: str = ''):
        self._conn.execute(
            "UPDATE sessions SET ended_at=?,actions=?,notes_opened=?,keyword=?,note=?"
            " WHERE id=?",
            (_now_str(), actions, notes_opened, keyword, note, session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 采集进度
    # ------------------------------------------------------------------

    def get_collection_progress(self, keyword: str) -> tuple:
        """返回 (已采集数, None) — target 由调用方管理。"""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM keyword_progress WHERE keyword=?", (keyword,)
        )
        return cur.fetchone()[0], None

    def mark_note_collected(self, keyword: str, note_id: str):
        self._conn.execute(
            "INSERT OR IGNORE INTO keyword_progress (keyword, note_id, collected_at)"
            " VALUES (?,?,?)",
            (keyword, note_id, _now_str()),
        )
        self._conn.commit()

    def get_all_collected_note_ids(self) -> set[str]:
        cur = self._conn.execute("SELECT note_id FROM keyword_progress")
        return {row[0] for row in cur.fetchall()}


# =============================================================================
# 自测
# =============================================================================
if __name__ == "__main__":
    store = SessionStore(":memory:")

    # 测试：无历史时应该运行
    ok = store.should_run_now(min_gap_hours=0)
    assert ok, "no history -> should_run_now=True"

    # 测试：会话记录
    sid = store.log_session_start()
    time.sleep(0.01)
    store.log_session_end(sid, actions=2, notes_opened=1, keyword="牙疼")
    assert store.get_today_session_count() >= 1
    last = store.get_last_session_time()
    assert last is not None

    # 测试：刚跑完 -> min_gap 未过 -> 不应运行
    ok = store.should_run_now(min_gap_hours=100)
    assert not ok, "just ran -> should not run"

    # 测试：进度追踪
    store.mark_note_collected("牙疼", "note_001")
    store.mark_note_collected("牙疼", "note_002")
    store.mark_note_collected("牙疼", "note_003")
    collected, _ = store.get_collection_progress("牙疼")
    assert collected == 3, f"expected 3, got {collected}"

    all_ids = store.get_all_collected_note_ids()
    assert "note_001" in all_ids

    # 测试：重复标记不重复计数
    store.mark_note_collected("牙疼", "note_001")
    collected, _ = store.get_collection_progress("牙疼")
    assert collected == 3, f"dup should still be 3, got {collected}"

    print("session_store.py: all tests passed")
    store.close()
