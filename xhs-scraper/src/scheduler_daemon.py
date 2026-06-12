"""
scheduler_daemon.py — 调度守护进程
====================================
在配置的时间窗口内自动执行采集会话。

用法:
    python -m src.scheduler_daemon

或:
    from src.scheduler_daemon import SchedulerDaemon
    daemon = SchedulerDaemon()
    daemon.run()
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta

from src.config_loader import load_config, validate_time_window
from src.chrome_manager import ChromeManager
from src.collector_engine import CollectionEngine
from src.session_store import SessionStore
from src._common import GracefulExiter


class SchedulerDaemon:
    """
    调度守护进程。

    核心逻辑：
      1. 加载配置 -> 初始化组件
      2. 循环：
         a. 检查当前是否在时间窗口内
         b. 不在窗口内 -> 计算下次运行时间，休眠到那时
         c. 在窗口内 -> 执行一次采集会话 -> 休息 -> 回到 b
      3. 收到 SIGINT/SIGTERM 时优雅退出
    """

    def __init__(self, config_path="config.yaml"):
        self._config_path = config_path
        self._config = load_config(config_path)
        self._store = SessionStore(self._config.collection.db_path)
        self._chrome_mgr = ChromeManager(self._config, log_fn=self._log)
        self._engine = CollectionEngine(self._config, self._chrome_mgr, self._store)
        self._exiter = GracefulExiter()
        self._today_collected = 0
        self._expiry_retry_count = 0  # Session 过期指数退避计数器

    # ---- Public API ----

    def run(self):
        """主循环：运行直到收到中断信号。"""
        self._log("调度器启动")
        self._log(f"关键词: {self._config.scheduler.keywords}")
        self._log(f"时间段: {self._config.scheduler.time_windows}")
        self._log(f"日采集上限: {self._config.scheduler.daily_count}")

        try:
            self._chrome_mgr.ensure_connection()
        except Exception as e:
            self._log(f"Chrome 初始连接失败: {e}")

        while not self._exiter.should_exit:
            try:
                self._tick()
            except Exception as e:
                self._log(f"调度循环异常: {e}")
                self._sleep_interruptible(60)

        self._log("调度器已停止")

    def get_status(self) -> dict:
        """返回当前状态字典（供外部监控使用）。"""
        now = datetime.now()
        return {
            "running": not self._exiter.should_exit,
            "time": now.isoformat(),
            "today_collected": self._today_collected,
            "daily_limit": self._config.scheduler.daily_count,
            "in_window": self._is_within_window(now),
            "windows": self._config.scheduler.time_windows,
            "keywords": self._config.scheduler.keywords,
            "next_run": self._format_next_run(self._calculate_next_run(now)),
        }

    # ---- Ticks ----

    def _tick(self):
        """单次调度检查。"""
        try:
            self._config = load_config(self._config_path)
        except Exception as e:
            self._log(f"重载配置失败: {e}")

        now = datetime.now()

        if not self._is_within_window(now):
            next_time = self._calculate_next_run(now)
            if next_time is None:
                self._sleep_until_tomorrow(now)
                self._today_collected = 0
                return

            wait = (next_time - now).total_seconds()
            if wait > 0:
                self._log(
                    f"下一窗口于 {next_time.strftime('%H:%M')} 开始"
                    f"，等待 {wait/60:.0f} 分钟"
                )
                self._sleep_interruptible(wait)
            return

        # 在窗口内
        if self._today_collected >= self._config.scheduler.daily_count:
            self._log(f"已达日采集上限 {self._config.scheduler.daily_count}")
            end = self._get_window_end(now)
            if end and end > now:
                wait = (end - now).total_seconds() + 10
                self._sleep_interruptible(wait)
            return

        self._execute_session()

        if self._exiter.should_exit:
            return

        rest_min, rest_max = self._config.scheduler.rest_between_sessions
        rest = random.randint(rest_min, rest_max)
        self._log(
            f"会话结束，休息 {rest} 分钟"
            f" (今日已采集 {self._today_collected} 条)"
        )
        self._sleep_interruptible(rest * 60)

    def _execute_session(self):
        """执行一次采集会话：遍历关键词，调用采集引擎。"""
        try:
            page = self._chrome_mgr.ensure_connection()
            expired, reason = self._chrome_mgr.detect_session_expiry(page)
            if expired:
                # Session 过期 → 指数退避
                self._chrome_mgr.notify_user(
                    f"Session 可能已过期，请检查并重新登录。\n原因: {reason}",
                    title="xhs-scraper: Session 过期",
                )
                # 指数退避：5分 → 10分 → 20分 → 最多 60 分
                self._expiry_retry_count += 1
                cooldown = min(3600, (5 * 60) * (2 ** (self._expiry_retry_count - 1)))
                self._log(f"Session 过期，等待 {cooldown//60} 分钟后重试")
                self._sleep_interruptible(cooldown)
                return

            # Session 有效 → 重置退避计数器
            self._expiry_retry_count = 0
        except Exception as e:
            self._log(f"Session 检查失败: {e}")
            return

        try:
            pool_ids = self._store.get_all_collected_note_ids()
        except Exception:
            pool_ids = set()

        keywords = list(self._config.scheduler.keywords)
        random.shuffle(keywords)

        per_keyword = max(1, self._config.scheduler.daily_count // max(len(keywords), 1))

        for keyword in keywords:
            if self._exiter.should_exit:
                break

            if random.random() < self._config.scheduler.random_skip_chance:
                self._log(f"随机跳过关键词: {keyword}")
                continue

            remaining = self._config.scheduler.daily_count - self._today_collected
            count = min(
                per_keyword,
                self._config.session_limits.per_session_cap,
                remaining,
            )
            if count <= 0:
                break

            self._log(f"开始采集 '{keyword}' (目标 {count} 条)")

            session_id = self._store.log_session_start()
            try:
                collected = self._engine.collect_notes(keyword, count, pool_ids)
                note_count = len(collected)
            except Exception as e:
                self._log(f"采集 '{keyword}' 失败: {e}")
                note_count = 0

            try:
                self._store.log_session_end(
                    session_id,
                    actions=note_count,
                    notes_opened=note_count,
                    keyword=keyword,
                )
            except Exception:
                pass

            self._today_collected += note_count
            self._log(
                f"'{keyword}': 采集 {note_count} 条"
                f" (今日累计 {self._today_collected})"
            )

            if self._exiter.should_exit:
                break

    # ---- 时间窗口计算 ----

    def _parse_windows(self, for_date=None):
        """解析配置中的时间段，返回 [(start, end), ...] 列表。"""
        if for_date is None:
            for_date = datetime.now().date()
        windows = []
        for w in self._config.scheduler.time_windows:
            sh, sm, eh, em = validate_time_window(w)
            start = datetime(for_date.year, for_date.month, for_date.day, sh, sm)
            end = datetime(for_date.year, for_date.month, for_date.day, eh, em)
            windows.append((start, end))
        return windows

    def _is_within_window(self, now=None):
        """检查当前时间是否在任一窗口内。"""
        now = now or datetime.now()
        for start, end in self._parse_windows(now.date()):
            if start <= now <= end:
                return True
        return False

    def _get_window_end(self, now=None):
        """返回当前所在窗口的结束时间，不在窗口内则返回 None。"""
        now = now or datetime.now()
        for start, end in self._parse_windows(now.date()):
            if start <= now <= end:
                return end
        return None

    def _calculate_next_run(self, now=None):
        """
        计算下次运行时间。

        返回值:
            datetime — 下次运行时间
            None    — 今日窗口已全部结束
        """
        now = now or datetime.now()
        windows = self._parse_windows(now.date())

        # 当前在窗口内 -> 立即运行
        for start, end in windows:
            if start <= now <= end:
                return now

        # 有即将到来的窗口 -> 在窗口内随机选一个时间
        for start, end in windows:
            if now < start:
                jitter = random.randint(0, 900)  # 0-15 min
                return start + timedelta(seconds=jitter)

        # 所有窗口已过
        return None

    @staticmethod
    def _format_next_run(next_time):
        if next_time is None:
            return "明天"
        return next_time.strftime('%H:%M')

    def _sleep_until_tomorrow(self, now=None):
        """休眠到明天凌晨（略微错开 0 点峰值）。"""
        now = now or datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        offset = random.randint(30, 300)  # 0.5-5 min after midnight
        wake = tomorrow + timedelta(seconds=offset)
        wait = (wake - now).total_seconds()
        self._log(
            f"今日窗口已全部结束，休眠至 {wake.strftime('%m/%d %H:%M')}"
        )
        self._sleep_interruptible(wait)

    # ---- 休眠 ----

    def _sleep_interruptible(self, seconds: float) -> bool:
        """
        可中断休眠：每 60 秒检查一次 should_exit。
        返回 True 表示正常完成，False 表示被中断。
        """
        end_time = time.time() + seconds
        while time.time() < end_time and not self._exiter.should_exit:
            remaining = min(60, end_time - time.time())
            if remaining <= 0:
                break
            time.sleep(remaining)
        return not self._exiter.should_exit

    # ---- 日志 ----

    @staticmethod
    def _log(msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] [daemon] {msg}", flush=True)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    daemon = SchedulerDaemon()
    try:
        daemon.run()
    except KeyboardInterrupt:
        daemon._log("收到 KeyboardInterrupt，退出")
