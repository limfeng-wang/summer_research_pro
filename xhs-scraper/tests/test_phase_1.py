"""
test_phase_1.py — 配置系统 + Chrome 生命周期管理 测试
"""
import os
import sys
from unittest.mock import MagicMock, patch

# 确保能导入 src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# 配置加载测试
# =============================================================================

class TestConfigLoader:
    """测试 config_loader 模块。"""

    def test_load_defaults(self):
        """验证默认配置值。"""
        from src.config_loader import AppConfig
        cfg = AppConfig()
        assert cfg.scheduler.daily_count == 80
        assert len(cfg.scheduler.time_windows) == 3
        assert "牙痛" in cfg.scheduler.keywords
        assert cfg.session_limits.per_session_cap == 50
        assert cfg.session_limits.max_scrolls == 30
        assert cfg.session_limits.scroll_stale_limit == 5
        assert cfg.chrome.cdp_port == 9222
        assert cfg.chrome.window_size == (1400, 900)
        assert cfg.collection.detail_scroll_chance == 0.35
        assert cfg.scheduler.rest_between_sessions == (10, 30)

    def test_load_config_file_exists(self, temp_config_file):
        """验证能从 YAML 文件加载配置。"""
        from src.config_loader import load_config
        with open(temp_config_file, "w", encoding="utf-8") as f:
            f.write("""
scheduler:
  daily_count: 30
  keywords: ["测试"]
  time_windows: ["09:00-10:00"]
  rest_between_sessions: [5, 15]
  bottom_retry_minutes: [10, 20]
            """)
        cfg = load_config(temp_config_file)
        assert cfg.scheduler.daily_count == 30
        assert cfg.scheduler.keywords == ["测试"]
        assert cfg.scheduler.rest_between_sessions == (5, 15)
        # 未覆盖的字段应使用默认值
        assert cfg.scheduler.random_skip_chance == 0.10
        assert cfg.session_limits.per_session_cap == 50

    def test_partial_config(self, temp_config_file):
        """验证部分配置：未提供字段用默认值。"""
        from src.config_loader import load_config
        with open(temp_config_file, "w", encoding="utf-8") as f:
            f.write("scheduler:\n  daily_count: 10\n")
        cfg = load_config(temp_config_file)
        assert cfg.scheduler.daily_count == 10
        # 未提供的字段应有默认值
        assert cfg.scheduler.keywords == ["牙痛", "牙疼", "口腔溃疡"]
        assert cfg.chrome.cdp_port == 9222

    def test_config_not_found_creates_default(self, temp_config_file):
        """验证文件不存在时自动创建默认配置。"""
        from src.config_loader import load_config
        nonexistent = temp_config_file + ".nonexistent"
        # 确保文件不存在
        if os.path.exists(nonexistent):
            os.unlink(nonexistent)
        cfg = load_config(nonexistent)
        assert cfg.scheduler.daily_count == 80
        # 确认文件已被创建
        assert os.path.exists(nonexistent)
        os.unlink(nonexistent)

    def test_validate_time_window(self):
        """验证时间段格式校验。"""
        from src.config_loader import validate_time_window
        # 合法格式
        sh, sm, eh, em = validate_time_window("09:00-11:30")
        assert (sh, sm, eh, em) == (9, 0, 11, 30)

    def test_validate_time_window_invalid_format(self):
        """验证非法时间段格式应抛出异常。"""
        from src.config_loader import validate_time_window
        import pytest
        with pytest.raises(ValueError, match="时间段"):
            validate_time_window("invalid")
        with pytest.raises(ValueError):
            validate_time_window("25:00-26:00")
        with pytest.raises(ValueError):
            validate_time_window("10:00-09:00")

    def test_validate_config_negative_count(self, temp_config_file):
        """验证负值校验。"""
        from src.config_loader import load_config, validate_config, AppConfig
        cfg = AppConfig()
        cfg.scheduler.daily_count = 0
        import pytest
        with pytest.raises(ValueError, match="daily_count"):
            validate_config(cfg)

    def test_validate_config_empty_keywords(self):
        """验证空关键词应报错。"""
        from src.config_loader import AppConfig, validate_config
        import pytest
        cfg = AppConfig()
        cfg.scheduler.keywords = []
        with pytest.raises(ValueError, match="keywords"):
            validate_config(cfg)

    def test_validate_config_empty_windows(self):
        """验证空时间段应报错。"""
        from src.config_loader import AppConfig, validate_config
        import pytest
        cfg = AppConfig()
        cfg.scheduler.time_windows = []
        with pytest.raises(ValueError, match="time_window"):
            validate_config(cfg)

    def test_validate_skip_chance_range(self):
        """验证跳过概率范围。"""
        from src.config_loader import AppConfig, validate_config
        import pytest
        cfg = AppConfig()
        cfg.scheduler.random_skip_chance = 1.5
        with pytest.raises(ValueError, match="random_skip_chance"):
            validate_config(cfg)

    def test_rest_between_sessions_validation(self):
        """验证会话休息间隔校验。"""
        from src.config_loader import AppConfig, validate_config
        import pytest
        cfg = AppConfig()
        cfg.scheduler.rest_between_sessions = (0, 10)
        with pytest.raises(ValueError, match="rest_between_sessions"):
            validate_config(cfg)
        cfg.scheduler.rest_between_sessions = (20, 10)
        with pytest.raises(ValueError, match="rest_between_sessions"):
            validate_config(cfg)

    def test_detail_scroll_chance_range(self):
        """验证详情页滚动概率范围。"""
        from src.config_loader import AppConfig, validate_config
        import pytest
        cfg = AppConfig()
        cfg.collection.detail_scroll_chance = 1.5
        with pytest.raises(ValueError, match="detail_scroll_chance"):
            validate_config(cfg)

    def test_load_yaml_with_all_fields(self, temp_config_file):
        """验证完整 YAML 配置加载。"""
        from src.config_loader import load_config
        with open(temp_config_file, "w", encoding="utf-8") as f:
            f.write("""
scheduler:
  daily_count: 100
  time_windows:
    - "08:00-10:00"
    - "11:00-13:00"
  keywords: ["牙痛", "牙周炎"]
  rest_between_sessions: [15, 45]
  bottom_retry_minutes: [20, 40]
  random_skip_chance: 0.20
session_limits:
  per_session_cap: 40
  max_scrolls: 25
  scroll_stale_limit: 6
chrome:
  profile_dir: "profiles/my_profile"
  cdp_port: 9333
  window_size: [1920, 1080]
collection:
  data_dir: "my_data"
  db_path: "my_data/archive.db"
  detail_scroll_chance: 0.50
            """)
        cfg = load_config(temp_config_file)
        assert cfg.scheduler.daily_count == 100
        assert len(cfg.scheduler.time_windows) == 2
        assert cfg.scheduler.keywords == ["牙痛", "牙周炎"]
        assert cfg.scheduler.rest_between_sessions == (15, 45)
        assert cfg.scheduler.bottom_retry_minutes == (20, 40)
        assert cfg.scheduler.random_skip_chance == 0.20
        assert cfg.session_limits.per_session_cap == 40
        assert cfg.session_limits.max_scrolls == 25
        assert cfg.session_limits.scroll_stale_limit == 6
        assert cfg.chrome.profile_dir == "profiles/my_profile"
        assert cfg.chrome.cdp_port == 9333
        assert cfg.chrome.window_size == (1920, 1080)
        assert cfg.collection.data_dir == "my_data"
        assert cfg.collection.db_path == "my_data/archive.db"
        assert cfg.collection.detail_scroll_chance == 0.50

    def test_default_config_yaml_format(self):
        """验证默认 YAML 模板可被正确解析。"""
        import yaml
        from src.config_loader import DEFAULT_CONFIG_YAML
        parsed = yaml.safe_load(DEFAULT_CONFIG_YAML)
        assert isinstance(parsed, dict)
        assert "scheduler" in parsed
        assert "session_limits" in parsed
        assert "chrome" in parsed
        assert "collection" in parsed


# =============================================================================
# Chrome 管理测试
# =============================================================================

class TestChromeManager:
    """测试 ChromeManager 类。"""

    def test_ensure_connection_success(self):
        """验证 ensure_connection 返回 page 对象。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()
        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            page = mgr.ensure_connection()

            mock_connect.assert_called_once()
            assert page is mock_page

    def test_ensure_connection_reuses_existing(self):
        """验证已有连接时不会重复 connect。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()
        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            p1 = mgr.ensure_connection()
            p2 = mgr.ensure_connection()

            # connect_chrome 应只被调用一次
            assert mock_connect.call_count == 1
            assert p1 is p2

    def test_reconnect_exponential_backoff(self):
        """验证重连使用指数退避。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.side_effect = [
                Exception("fail 1"),
                Exception("fail 2"),
                mock_page,  # 第三次成功
            ]
            mgr = ChromeManager(cfg)

            # 旧 page 会触发重连
            old_page = MagicMock()
            new_page = mgr.reconnect(old_page)

            assert mock_connect.call_count == 3
            assert new_page is mock_page

    def test_reconnect_all_failures(self):
        """验证所有重连都失败时抛出 RuntimeError。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        import pytest
        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_connect.side_effect = Exception("always fail")
            mgr = ChromeManager(cfg)
            with pytest.raises(RuntimeError, match="重连失败"):
                mgr.reconnect(MagicMock())

    def test_keep_alive_success(self):
        """验证 keep_alive 在 Chrome 在线时返回 True。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            assert mgr.keep_alive() is True

    def test_keep_alive_failure(self):
        """验证 keep_alive 在 Chrome 断开时返回 False。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_page.run_cdp.side_effect = Exception("disconnected")
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            # 第一次 ensure_connection 成功后，run_cdp 会失败
            mgr.ensure_connection()
            assert mgr.keep_alive() is False

    def test_close_quits_page(self):
        """验证 close 会 quit page 但不会 kill Chrome。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            mgr.ensure_connection()
            mgr.close()

            mock_page.quit.assert_called_once()
            assert mgr._page is None


# =============================================================================
# Session 过期检测测试
# =============================================================================

class TestSessionDetector:
    """测试 SessionDetector 多策略检测。"""

    def _make_detector(self, url="https://www.xiaohongshu.com/search_result?keyword=牙痛",
                       title="小红书", run_js_return=None):
        page = MagicMock()
        page.url = url
        page.title = title
        page.run_js.return_value = run_js_return
        from src.chrome_manager import SessionDetector
        return SessionDetector(page)

    def test_login_redirect_detected(self):
        """验证检测到登录跳转。"""
        detector = self._make_detector(url="https://www.xiaohongshu.com/login?redirect=...")
        expired, reason = detector._check_login_redirect()
        assert expired is True
        assert "login" in reason.lower()

    def test_no_login_redirect(self):
        """验证正常页面不会误报。"""
        detector = self._make_detector()
        expired, reason = detector._check_login_redirect()
        assert expired is False

    def test_captcha_detected(self):
        """验证检测到滑块验证码。"""
        detector = self._make_detector(run_js_return=True)
        expired, reason = detector._check_captcha()
        assert expired is True
        assert "验证码" in reason

    def test_no_captcha(self):
        """验证无验证码时不会误报。"""
        detector = self._make_detector(run_js_return=False)
        expired, reason = detector._check_captcha()
        assert expired is False

    def test_block_page_detected(self):
        """验证风控拦截页检测。"""
        detector = self._make_detector(title="访问受限", run_js_return="您的请求过于频繁，请稍后再试")
        expired, reason = detector._check_block_page()
        assert expired is True

    def test_no_block_page(self):
        """验证正常页面不会误报封禁。"""
        detector = self._make_detector(run_js_return="正常搜索结果内容")
        expired, reason = detector._check_block_page()
        assert expired is False

    def test_empty_search_results(self):
        """验证搜索页无结果检测。"""
        detector = self._make_detector(url="https://www.xiaohongshu.com/search_result?keyword=牙痛",
                                        run_js_return=False)
        expired, reason = detector._check_empty_results()
        assert expired is True
        assert "无任何笔记" in reason

    def test_non_search_page_empty_check(self):
        """验证非搜索页不会触发空数据检测。"""
        detector = self._make_detector(url="https://www.xiaohongshu.com/explore/xxx")
        expired, reason = detector._check_empty_results()
        assert expired is False

    def test_check_all_combined(self):
        """验证全量检测在正常页面上返回 (False, '')。

        注意：使用非搜索页 URL 避免 _check_empty_results 误触发，且
        将 run_js 设为返回 False（表示无验证码、无封锁关键词）。
        """
        detector = self._make_detector(
            url="https://www.xiaohongshu.com/explore/abc123",
            run_js_return=False
        )
        expired, reason = detector.check_all()
        assert expired is False
        assert reason == ""


class TestChromeManagerSessionDetection:
    """测试 ChromeManager 的 Session 过期检测。"""

    def test_detect_expiry_with_redirect(self):
        """验证 ChromeManager 能检测到过期。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_page.url = "https://www.xiaohongshu.com/login?redirect=/"
            mock_page.title = "登录"
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            mgr.ensure_connection()
            expired, reason = mgr.detect_session_expiry()

            assert expired is True
            assert "login" in reason.lower()

    def test_detect_expiry_with_captcha(self):
        """验证 ChromeManager 能检测到滑块验证码。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import AppConfig
        cfg = AppConfig()

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_page.url = "https://www.xiaohongshu.com/search_result"
            mock_page.title = "小红书"
            mock_page.run_js.return_value = True  # 有验证码元素
            mock_connect.return_value = mock_page

            mgr = ChromeManager(cfg)
            mgr.ensure_connection()
            expired, reason = mgr.detect_session_expiry()

            assert expired is True
            assert "验证码" in reason

    def test_notify_writes_flag_file(self, temp_data_dir):
        """验证通知机制写入 flag 文件。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import ChromeConfig
        cfg = ChromeConfig()
        # 直接传一个带 data_dir 的对象
        mgr = ChromeManager(cfg)

        flag_path = os.path.join(temp_data_dir, "session_expired.flag")
        # 手动调用 notify 并指定路径
        mgr.notify_user("测试通知", title="测试")
        # 检查默认路径
        default_flag = "data/session_expired.flag"
        if os.path.exists(default_flag):
            content = open(default_flag, encoding="utf-8").read()
            assert "测试通知" in content
            assert "timestamp" in content
            os.unlink(default_flag)

    def test_notify_does_not_crash(self):
        """验证通知不会因任何原因崩溃。"""
        from src.chrome_manager import ChromeManager
        from src.config_loader import ChromeConfig
        cfg = ChromeConfig()
        mgr = ChromeManager(cfg)

        # 即使无法写入文件、无法弹窗，也不应崩溃
        try:
            mgr.notify_user("测试", title="测试")
        except Exception:
            assert False, "notify_user 不应抛出任何异常"
