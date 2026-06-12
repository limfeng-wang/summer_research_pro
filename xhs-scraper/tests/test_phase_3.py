"""
test_phase_3.py — 调度守护进程 测试
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# =============================================================================
# 时间窗口计算测试
# =============================================================================

class TestTimeWindowCalculation:
    """测试 SchedulerDaemon 的时间窗口计算逻辑。"""

    def _make_daemon(self, time_windows=None):
        """创建测试用 daemon 实例（绕过 __init__ 避免文件 IO）。"""
        from src.scheduler_daemon import SchedulerDaemon

        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._config = MagicMock()
        daemon._config.scheduler.time_windows = time_windows or [
            "08:00-11:30", "13:00-17:00", "19:00-23:00"
        ]
        daemon._config.scheduler.keywords = ["牙痛"]
        daemon._config.scheduler.daily_count = 80
        daemon._config.scheduler.rest_between_sessions = (10, 30)
        daemon._config.scheduler.random_skip_chance = 0.0
        daemon._config.session_limits.per_session_cap = 50
        daemon._config.collection.db_path = ":memory:"
        daemon._store = MagicMock()
        daemon._chrome_mgr = MagicMock()
        daemon._engine = MagicMock()
        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = False
        daemon._today_collected = 0
        return daemon

    def test_is_within_window_true(self):
        """验证在窗口内返回 True。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 10, 0)  # 10:00, within 08:00-11:30
        assert daemon._is_within_window(now) is True

    def test_is_within_window_false(self):
        """验证在窗口外返回 False。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 12, 0)  # 12:00, outside
        assert daemon._is_within_window(now) is False

    def test_is_within_window_multiple_windows(self):
        """验证多窗口配置下的检测。"""
        daemon = self._make_daemon(["08:00-11:30", "13:00-17:00"])

        assert daemon._is_within_window(datetime(2026, 5, 19, 9, 0)) is True
        assert daemon._is_within_window(datetime(2026, 5, 19, 12, 0)) is False
        assert daemon._is_within_window(datetime(2026, 5, 19, 14, 0)) is True
        assert daemon._is_within_window(datetime(2026, 5, 19, 18, 0)) is False

    def test_is_within_window_boundary(self):
        """验证窗口边界（包含起始和结束时刻）。"""
        daemon = self._make_daemon(["09:00-12:00"])

        # 边界应包含在内
        assert daemon._is_within_window(datetime(2026, 5, 19, 9, 0)) is True
        assert daemon._is_within_window(datetime(2026, 5, 19, 12, 0)) is True
        # 边界外
        assert daemon._is_within_window(datetime(2026, 5, 19, 8, 59)) is False
        assert daemon._is_within_window(datetime(2026, 5, 19, 12, 1)) is False

    def test_calculate_next_run_within_window(self):
        """验证在窗口内返回 now。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 10, 0)
        result = daemon._calculate_next_run(now)
        assert result == now

    def test_calculate_next_run_before_window(self):
        """验证在窗口前返回 start + jitter。"""
        daemon = self._make_daemon(["13:00-17:00"])
        now = datetime(2026, 5, 19, 10, 0)  # 10:00, before 13:00

        with patch.object(random, 'randint', return_value=300):
            result = daemon._calculate_next_run(now)
            expected = datetime(2026, 5, 19, 13, 0) + timedelta(seconds=300)
            assert result == expected

    def test_calculate_next_run_all_passed(self):
        """验证所有窗口结束后返回 None。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 15, 0)  # after 08:00-11:30
        result = daemon._calculate_next_run(now)
        assert result is None

    def test_calculate_next_run_earliest_window(self):
        """验证多个窗口中选取最近的一个。"""
        daemon = self._make_daemon(["08:00-11:30", "14:00-17:00"])
        now = datetime(2026, 5, 19, 12, 0)  # between windows

        with patch.object(random, 'randint', return_value=0):
            result = daemon._calculate_next_run(now)
            expected = datetime(2026, 5, 19, 14, 0)
            assert result == expected

    def test_get_window_end_returns_correct(self):
        """验证 _get_window_end 返回当前窗口的结束时间。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 10, 0)
        result = daemon._get_window_end(now)
        expected = datetime(2026, 5, 19, 11, 30)
        assert result == expected

    def test_get_window_end_outside_window(self):
        """验证不在窗口内时返回 None。"""
        daemon = self._make_daemon(["08:00-11:30"])
        now = datetime(2026, 5, 19, 14, 0)
        result = daemon._get_window_end(now)
        assert result is None

    def test_format_next_run_none(self):
        """验证 None 显示为 '明天'。"""
        from src.scheduler_daemon import SchedulerDaemon
        assert SchedulerDaemon._format_next_run(None) == "明天"

    def test_format_next_run_time(self):
        """验证有效时间格式化。"""
        from src.scheduler_daemon import SchedulerDaemon
        dt = datetime(2026, 5, 19, 14, 30)
        result = SchedulerDaemon._format_next_run(dt)
        assert result == "14:30"


# =============================================================================
# 调度器执行测试
# =============================================================================

class TestSchedulerExecution:
    """测试 SchedulerDaemon 的执行逻辑。"""

    def _make_daemon(self, **overrides):
        from src.scheduler_daemon import SchedulerDaemon
        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._config = MagicMock()
        daemon._config.scheduler.time_windows = ["08:00-11:30"]
        daemon._config.scheduler.keywords = ["牙痛", "牙疼"]
        daemon._config.scheduler.daily_count = 80
        daemon._config.scheduler.rest_between_sessions = (10, 30)
        daemon._config.scheduler.random_skip_chance = 0.0
        daemon._config.scheduler.bottom_retry_minutes = (15, 25)
        daemon._config.session_limits.per_session_cap = 50
        daemon._config.collection.db_path = ":memory:"

        daemon._store = MagicMock()
        daemon._store.get_all_collected_note_ids.return_value = set()
        daemon._store.log_session_start.return_value = 1

        daemon._chrome_mgr = MagicMock()
        daemon._chrome_mgr.ensure_connection.return_value = MagicMock()
        daemon._chrome_mgr.detect_session_expiry.return_value = (False, "")

        daemon._engine = MagicMock()
        daemon._engine.collect_notes.return_value = [
            ("n1", "t1"), ("n2", "t2")
        ]

        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = False
        daemon._today_collected = 0

        for k, v in overrides.items():
            setattr(daemon, k, v)

        return daemon

    def test_execute_session_normal(self):
        """验证正常采集流程。"""
        daemon = self._make_daemon()
        daemon._execute_session()

        daemon._chrome_mgr.ensure_connection.assert_called()
        daemon._chrome_mgr.detect_session_expiry.assert_called()
        # 每个关键词调用一次 collect_notes
        assert daemon._engine.collect_notes.call_count == len(daemon._config.scheduler.keywords)
        assert daemon._store.log_session_start.call_count == len(daemon._config.scheduler.keywords)
        assert daemon._store.log_session_end.call_count == len(daemon._config.scheduler.keywords)
        # 2 个关键词 × 2 条/次
        assert daemon._today_collected == 4

    def test_execute_session_expired(self):
        """验证 Session 过期时不采集。"""
        daemon = self._make_daemon()
        daemon._chrome_mgr.detect_session_expiry.return_value = (True, "登录跳转")

        with patch.object(daemon, '_sleep_interruptible', return_value=True):
            daemon._execute_session()

        daemon._engine.collect_notes.assert_not_called()
        daemon._store.log_session_start.assert_not_called()
        daemon._chrome_mgr.notify_user.assert_called_once()

    def test_execute_session_random_skip(self):
        """验证随机跳过所有关键词。"""
        daemon = self._make_daemon()
        daemon._config.scheduler.random_skip_chance = 1.0

        with patch.object(daemon, '_sleep_interruptible', return_value=True):
            daemon._execute_session()

        daemon._engine.collect_notes.assert_not_called()

    def test_execute_session_exit_during_collection(self):
        """验证采集过程中收到退出信号可停止。"""
        daemon = self._make_daemon()

        def side_effect(*args, **kwargs):
            daemon._exiter.should_exit = True
            return [("n1", "t1")]

        daemon._engine.collect_notes.side_effect = side_effect

        daemon._execute_session()

        assert daemon._engine.collect_notes.call_count == 1

    def test_execute_session_collect_notes_failure(self):
        """验证 collect_notes 异常时优雅处理不崩溃。"""
        daemon = self._make_daemon()
        daemon._engine.collect_notes.side_effect = Exception("采集失败")

        daemon._execute_session()

        assert daemon._today_collected == 0
        # 每个关键词都会记一条 session_end（note_count=0）
        assert daemon._store.log_session_end.call_count == len(daemon._config.scheduler.keywords)

    def test_execute_session_session_check_failure(self):
        """验证 Session 检查异常时不采集。"""
        daemon = self._make_daemon()
        daemon._chrome_mgr.ensure_connection.side_effect = Exception("连接失败")

        daemon._execute_session()

        daemon._engine.collect_notes.assert_not_called()

    def test_execute_session_counts_remaining(self):
        """验证采集数量受每日剩余额度限制。"""
        daemon = self._make_daemon()
        daemon._today_collected = 78  # daily_count=80, only 2 remaining

        daemon._execute_session()

        args, kwargs = daemon._engine.collect_notes.call_args
        # collect_notes(keyword, count, pool_ids)
        assert kwargs.get('count') == 2 or args[1] == 2

    def test_execute_session_multiple_keywords(self):
        """验证多关键词时每个关键词都采集。"""
        daemon = self._make_daemon()
        daemon._config.scheduler.keywords = ["牙痛", "牙疼", "口腔溃疡"]

        daemon._execute_session()

        # Should have called collect_notes for each keyword
        assert daemon._engine.collect_notes.call_count == 3

    # ---- _tick 测试 ----

    def test_tick_within_window_executes_session(self):
        """验证在窗口内 _tick 执行采集。"""
        daemon = self._make_daemon()

        with patch('src.scheduler_daemon.load_config') as mock_load, \
             patch.object(daemon, '_is_within_window', return_value=True), \
             patch.object(daemon, '_execute_session') as mock_exec, \
             patch.object(daemon, '_sleep_interruptible', return_value=True):

            mock_load.return_value = daemon._config
            daemon._today_collected = 0
            daemon._tick()
            mock_exec.assert_called_once()

    def test_tick_outside_window_waits_for_next(self):
        """验证在窗口外 _tick 等待下一窗口。"""
        daemon = self._make_daemon()
        next_time = datetime.now() + timedelta(hours=2)

        with patch('src.scheduler_daemon.load_config') as mock_load, \
             patch.object(daemon, '_is_within_window', return_value=False), \
             patch.object(daemon, '_calculate_next_run', return_value=next_time), \
             patch.object(daemon, '_sleep_interruptible') as mock_sleep:

            mock_load.return_value = daemon._config
            daemon._tick()
            mock_sleep.assert_called_once()

    def test_tick_all_windows_passed_sleeps_till_tomorrow(self):
        """验证所有窗口结束后休眠到明天并重置计数器。"""
        daemon = self._make_daemon()

        with patch('src.scheduler_daemon.load_config') as mock_load, \
             patch.object(daemon, '_is_within_window', return_value=False), \
             patch.object(daemon, '_calculate_next_run', return_value=None), \
             patch.object(daemon, '_sleep_until_tomorrow') as mock_sleep:

            mock_load.return_value = daemon._config
            daemon._today_collected = 15
            daemon._tick()
            mock_sleep.assert_called_once()
            assert daemon._today_collected == 0

    def test_tick_at_daily_limit_skips_session(self):
        """验证已达日限时不执行采集。"""
        daemon = self._make_daemon()
        daemon._today_collected = 80  # daily limit

        with patch('src.scheduler_daemon.load_config') as mock_load, \
             patch.object(daemon, '_is_within_window', return_value=True), \
             patch.object(daemon, '_get_window_end', return_value=None):

            mock_load.return_value = daemon._config
            daemon._tick()
            daemon._engine.collect_notes.assert_not_called()


# =============================================================================
# 休眠中断测试
# =============================================================================

class TestSleepInterruptible:
    """测试 SchedulerDaemon 的可中断休眠。"""

    def _make_daemon(self):
        from src.scheduler_daemon import SchedulerDaemon
        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = False
        return daemon

    def test_sleep_interruptible_normal(self):
        """验证正常休眠完成返回 True。"""
        daemon = self._make_daemon()
        result = daemon._sleep_interruptible(0.05)
        assert result is True

    def test_sleep_interruptible_interrupted(self):
        """验证被中断时返回 False。"""
        daemon = self._make_daemon()
        daemon._exiter.should_exit = True

        result = daemon._sleep_interruptible(100)
        assert result is False

    def test_sleep_interruptible_zero(self):
        """验证 0 秒休眠正常返回。"""
        daemon = self._make_daemon()
        result = daemon._sleep_interruptible(0)
        assert result is True

    def test_sleep_interruptible_negative(self):
        """验证负数秒数不阻塞。"""
        daemon = self._make_daemon()
        result = daemon._sleep_interruptible(-1)
        assert result is True


# =============================================================================
# 状态报告测试
# =============================================================================

class TestSchedulerStatus:
    """测试 SchedulerDaemon 的状态报告。"""

    def test_get_status_structure(self):
        """验证状态字典的结构和字段。"""
        from src.scheduler_daemon import SchedulerDaemon

        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._config = MagicMock()
        daemon._config.scheduler.time_windows = ["08:00-11:30"]
        daemon._config.scheduler.keywords = ["牙痛"]
        daemon._config.scheduler.daily_count = 80
        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = False
        daemon._today_collected = 15

        with patch.object(daemon, '_is_within_window', return_value=True), \
             patch.object(daemon, '_calculate_next_run',
                          return_value=datetime(2026, 5, 19, 10, 0)):

            status = daemon.get_status()

            assert isinstance(status, dict)
            assert status["running"] is True
            assert status["today_collected"] == 15
            assert status["daily_limit"] == 80
            assert status["in_window"] is True
            assert status["next_run"] == "10:00"

    def test_get_status_not_running(self):
        """验证退出状态。"""
        from src.scheduler_daemon import SchedulerDaemon

        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._config = MagicMock()
        daemon._config.scheduler.time_windows = ["08:00-11:30"]
        daemon._config.scheduler.keywords = ["牙痛"]
        daemon._config.scheduler.daily_count = 80
        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = True
        daemon._today_collected = 0

        with patch.object(daemon, '_is_within_window', return_value=False), \
             patch.object(daemon, '_calculate_next_run', return_value=None):

            status = daemon.get_status()

            assert status["running"] is False
            assert status["next_run"] == "明天"


# =============================================================================
# GracefulExiter 信号处理测试
# =============================================================================

class TestGracefulExiter:
    """测试信号处理机制。"""

    def test_exiter_initially_false(self):
        """验证初始状态为 False。"""
        from src._common import GracefulExiter
        exiter = GracefulExiter()
        assert exiter.should_exit is False
        # 恢复信号处理
        import signal
        signal.signal(signal.SIGINT, signal.default_int_handler)
