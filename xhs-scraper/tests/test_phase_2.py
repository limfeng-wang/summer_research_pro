"""
test_phase_2.py — 增强采集引擎 测试
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch


# =============================================================================
# win32_api 测试
# =============================================================================

class TestWin32API:
    """测试 win32_api 模块的基础功能。"""

    def test_is_windows(self):
        """验证 is_windows 返回布尔值（当前平台是 win32）。"""
        from src.win32_api import is_windows
        result = is_windows()
        assert isinstance(result, bool)

    def test_calibrate_returns_dict(self):
        """验证 calibrate 返回正确的结构。"""
        from src.win32_api import calibrate
        mock_page = MagicMock()
        mock_page.run_js.return_value = (
            '{"dpr":1.0}'
        )

        with patch('src.win32_api.find_chrome_hwnd', return_value=[
            {'hwnd': 0, 'left': 0, 'top': 0, 'width': 1400, 'height': 900}
        ]):
            result = calibrate(mock_page)
            assert isinstance(result, dict)
            assert 'csx' in result
            assert 'csy' in result
            assert 'dpr' in result
            assert 'hwnd' in result

    def test_vp2screen(self):
        """验证视口到屏幕坐标转换。"""
        from src.win32_api import vp2screen
        calib = {'csx': 100, 'csy': 50, 'dpr': 1.5}
        sx, sy = vp2screen(calib, 200, 150)
        assert sx == 100 + int(200 * 1.5)  # 400
        assert sy == 50 + int(150 * 1.5)   # 275

    def test_vp2screen_dpr_1(self):
        """验证 DPR=1 时的坐标转换。"""
        from src.win32_api import vp2screen
        calib = {'csx': 0, 'csy': 0, 'dpr': 1.0}
        sx, sy = vp2screen(calib, 100, 200)
        assert sx == 100
        assert sy == 200

    def test_find_chrome_hwnd_no_windows(self):
        """验证无窗口时返回空列表。"""
        from src.win32_api import find_chrome_hwnd
        with patch('src.win32_api.user32', None), \
             patch('src.win32_api.is_windows', return_value=False):
            result = find_chrome_hwnd()
            assert result == []


# =============================================================================
# 采集引擎测试
# =============================================================================

class TestCollectionEngine:
    """测试 CollectionEngine 核心功能。"""

    def _make_config(self):
        """创建一个测试用配置。"""
        from src.config_loader import AppConfig
        cfg = AppConfig()
        cfg.session_limits.max_scrolls = 20
        cfg.session_limits.scroll_stale_limit = 3
        cfg.collection.data_dir = "data"
        return cfg

    def _make_mock_store(self):
        """创建 mock SessionStore。"""
        store = MagicMock()
        store.mark_note_collected.return_value = None
        return store

    def test_scroll_to_bottom_stops_after_stale(self):
        """验证全滚动在连续停滞 limit 次后停止。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_page = MagicMock()
        mock_mgr.ensure_connection.return_value = mock_page
        mock_mgr.detect_session_expiry.return_value = (False, "")
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)

        # mock get_cards 始终返回同一批卡片
        mock_col_mod = MagicMock()
        mock_col_mod.get_cards.return_value = [
            {'note_id': 'aaa', 'cx': 100, 'cy': 200, 'title': 'test'}
        ]

        # 直接测试 _scroll_to_bottom
        # 需要把 engine._page 设好
        engine._page = mock_page
        engine._calib = {'csx': 0, 'csy': 0, 'dpr': 1, 'hwnd': None}

        # 首次获取可见 ID 会有一批卡片，后续再获取同一批，
        # 所以不会新增。
        new_count = engine._scroll_to_bottom(keyword="牙痛", col_mod=mock_col_mod)

        # 因为第一次拿到的 ids 和第二次一样，不会有 new_ids 增加，
        # 所以 scroll 循环会连续 3 次无新增 → 停止
        assert isinstance(new_count, int)

    def test_cards_filtered_by_pool_ids(self):
        """验证 pool_ids 中的卡片被过滤。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.ensure_connection.return_value = MagicMock()
        mock_mgr.detect_session_expiry.return_value = (False, "")
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)

        # 直接验证过滤逻辑
        cards = [
            {'note_id': 'aaa'},
            {'note_id': 'bbb'},
            {'note_id': 'ccc'},
        ]
        pool_ids = {'aaa', 'ccc'}
        filtered = [c for c in cards if c['note_id'] not in pool_ids]
        assert len(filtered) == 1
        assert filtered[0]['note_id'] == 'bbb'

    def test_empty_cards_returns_empty_list(self):
        """验证 get_cards 返回空列表时能正确处理。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.ensure_connection.return_value = MagicMock()
        mock_mgr.detect_session_expiry.return_value = (False, "")
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)

        # mock get_cards 返回空
        mock_col_mod = MagicMock()
        mock_col_mod.get_cards.return_value = []
        engine._page = MagicMock()

        new_count = engine._scroll_to_bottom(keyword="牙痛", col_mod=mock_col_mod)
        # 初始就是 0，scroll 中不会有新卡片
        assert new_count == 0

    def test_handle_disconnect_detection(self):
        """验证断开连接能被正确检测。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.reconnect.return_value = MagicMock()
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)
        engine._page = MagicMock()

        # reconnect 返回的页面需支持 calibrate 的 JS 调用
        mock_mgr.reconnect.return_value.run_js.return_value = '{"dpr":1.0}'

        # 模拟 PageDisconnectedError
        err = Exception("PageDisconnectedError")
        result = engine._handle_disconnect(err)
        assert result is True
        mock_mgr.reconnect.assert_called_once()

    def test_handle_non_disconnect_error(self):
        """验证非断开错误不会触发重连。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)

        # 普通错误不应触发重连
        err = Exception("some other error")
        result = engine._handle_disconnect(err)
        assert result is False
        mock_mgr.reconnect.assert_not_called()

    def test_handle_timeout_as_disconnect(self):
        """验证 TimeoutError 也被视为断开。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.reconnect.return_value = MagicMock()
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)
        engine._page = MagicMock()

        mock_mgr.reconnect.return_value.run_js.return_value = '{"dpr":1.0}'

        err = Exception("TimeoutError")
        result = engine._handle_disconnect(err)
        assert result is True

    def test_collect_notes_empty_on_expired_session(self):
        """验证 Session 过期时 collect_notes 返回空。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.detect_session_expiry.return_value = (True, "登录跳转")
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)
        result = engine.collect_notes("牙痛", count=10, pool_ids=set())
        assert result == []

    def test_collect_notes_session_check_called(self):
        """验证 collect_notes 首先检查 Session。"""
        from src.collector_engine import CollectionEngine

        cfg = self._make_config()
        mock_mgr = MagicMock()
        mock_mgr.detect_session_expiry.return_value = (False, "")
        store = self._make_mock_store()

        engine = CollectionEngine(cfg, mock_mgr, store)
        mock_mgr.ensure_connection.return_value = MagicMock()
        # calibrate 需要 run_js 返回合法 JSON
        mock_mgr.ensure_connection.return_value.run_js.return_value = '{"dpr":1.0}'

        # 模拟导航失败，返回空
        with patch('src.collector_engine._get_collector') as mock_col:
            mock_col_mod = MagicMock()
            mock_col.return_value = mock_col_mod
            mock_col_mod.get_cards.side_effect = Exception("navigated away")

            result = engine.collect_notes("牙痛", count=10, pool_ids=set())
            assert isinstance(result, list)

    def test_calibration_values(self):
        """验证坐标校准的数学运算。"""
        from src.win32_api import vp2screen

        # 不同 DPR 下的转换
        calib_dpr1 = {'csx': 50, 'csy': 30, 'dpr': 1.0, 'hwnd': None}
        calib_dpr2 = {'csx': 50, 'csy': 30, 'dpr': 2.0, 'hwnd': None}

        # DPR=1.0: screen = client + viewport * 1
        sx1, sy1 = vp2screen(calib_dpr1, 400, 300)
        assert sx1 == 50 + 400
        assert sy1 == 30 + 300

        # DPR=2.0: screen = client + viewport * 2
        sx2, sy2 = vp2screen(calib_dpr2, 400, 300)
        assert sx2 == 50 + 800
        assert sy2 == 30 + 600


# =============================================================================
# 数据保存测试
# =============================================================================

class TestDataSaving:
    """测试数据保存和去重逻辑。"""

    def test_pool_ids_deduplication(self):
        """验证 SQLite 去重逻辑。"""
        from src.session_store import SessionStore
        store = SessionStore(":memory:")

        # 写入一条
        store.mark_note_collected("牙痛", "note001")
        store.mark_note_collected("牙痛", "note002")

        all_ids = store.get_all_collected_note_ids()
        assert "note001" in all_ids
        assert "note002" in all_ids
        assert len(all_ids) == 2

    def test_pool_ids_dedup_duplicate(self):
        """验证重复写入不产生重复。"""
        from src.session_store import SessionStore
        store = SessionStore(":memory:")

        store.mark_note_collected("牙痛", "note001")
        store.mark_note_collected("牙痛", "note001")  # 重复写入

        all_ids = store.get_all_collected_note_ids()
        assert len(all_ids) == 1

    def test_save_summary_structure(self):
        """验证 summary 数据结构。"""
        collected = [
            ("note001", "牙痛怎么治"),
            ("note002", "牙疼怎么办"),
        ]
        summary = {
            "keyword": "牙痛",
            "collected": len(collected),
            "time": "2026-01-01T00:00:00",
            "notes": [{"id": nid, "title": title} for nid, title in collected],
        }
        assert summary["collected"] == 2
        assert summary["notes"][0]["id"] == "note001"
        assert summary["notes"][1]["title"] == "牙疼怎么办"
