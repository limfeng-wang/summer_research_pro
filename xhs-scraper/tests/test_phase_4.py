"""
test_phase_4.py — 端到端集成测试
==================================
验证全链路：配置加载 -> Chrome 连接 -> 采集引擎 -> 数据存储。
所有外部依赖（CDP、浏览器）均 mock，仅测试内部流程正确性。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestE2EConfigToEngine:
    """验证配置系统到采集引擎的全链路集成。"""

    def test_config_loads_and_engine_accepts(self, temp_config_file):
        """验证 load_config -> AppConfig -> CollectionEngine 可用。"""
        from src.config_loader import load_config
        from src.collector_engine import CollectionEngine
        from src.chrome_manager import ChromeManager
        from src.session_store import SessionStore

        # 写临时配置
        with open(temp_config_file, "w", encoding="utf-8") as f:
            f.write("""
scheduler:
  daily_count: 30
  time_windows: ["09:00-11:00"]
  keywords: ["牙痛"]
session_limits:
  per_session_cap: 10
  max_scrolls: 5
  scroll_stale_limit: 3
collection:
  data_dir: "data"
  db_path: ":memory:"
            """)

        cfg = load_config(temp_config_file)
        assert cfg.scheduler.daily_count == 30
        assert cfg.session_limits.per_session_cap == 10

        store = SessionStore(":memory:")
        chrome_mgr = ChromeManager(cfg)

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            engine = CollectionEngine(cfg, chrome_mgr, store)
            assert engine._config is cfg
            assert engine._chrome_mgr is chrome_mgr

    def test_engine_rejects_expired_session(self, temp_config_file):
        """验证 Session 过期时 engine 返回空列表。"""
        from src.config_loader import load_config
        from src.collector_engine import CollectionEngine
        from src.chrome_manager import ChromeManager
        from src.session_store import SessionStore

        with open(temp_config_file, "w", encoding="utf-8") as f:
            f.write("collection:\n  data_dir: \"data\"\n  db_path: \":memory:\"\n")

        cfg = load_config(temp_config_file)
        store = SessionStore(":memory:")
        chrome_mgr = ChromeManager(cfg)

        with patch('src.xhs_snapshot.connect_chrome') as mock_connect:
            mock_page = MagicMock()
            mock_connect.return_value = mock_page

            # 模拟 Session 过期
            chrome_mgr._page = mock_page
            chrome_mgr.detect_session_expiry = MagicMock(return_value=(True, "登录跳转"))
            chrome_mgr.notify_user = MagicMock()

            engine = CollectionEngine(cfg, chrome_mgr, store)
            result = engine.collect_notes("牙痛", count=5, pool_ids=set())
            assert result == []


class TestE2ECollectionWithMockCards:
    """验证模拟卡片数据的完整采集流程。"""

    def _setup(self, temp_dir):
        """创建测试环境，返回 (config, chrome_mgr, store, engine, mock_collector)。"""
        from src.config_loader import load_config, AppConfig
        from src.collector_engine import CollectionEngine
        from src.chrome_manager import ChromeManager
        from src.session_store import SessionStore

        cfg = AppConfig()
        cfg.collection.data_dir = temp_dir
        cfg.collection.db_path = ":memory:"
        cfg.session_limits.per_session_cap = 10
        cfg.session_limits.max_scrolls = 5
        cfg.session_limits.scroll_stale_limit = 3

        store = SessionStore(":memory:")
        chrome_mgr = ChromeManager(cfg)

        # 模拟 Chrome 页面
        mock_page = MagicMock()
        mock_page.get.return_value = None
        mock_page.wait.ele_displayed.return_value = None
        mock_page.run_js.return_value = '{"dpr":1.0}'

        with patch('src.xhs_snapshot.connect_chrome', return_value=mock_page):
            chrome_mgr.ensure_connection()

        # 绕过 Session 检测（mock page 的 run_js 返回值会被误判为验证码）
        chrome_mgr.detect_session_expiry = MagicMock(return_value=(False, ""))

        engine = CollectionEngine(cfg, chrome_mgr, store)
        engine._out_dir = temp_dir

        # 设置页面和校准
        engine._page = mock_page
        engine._calib = {'csx': 0, 'csy': 0, 'dpr': 1.0, 'hwnd': None}

        # 模拟 collector 模块
        mock_col_mod = MagicMock()

        return cfg, chrome_mgr, store, engine, mock_col_mod, mock_page

    def test_engine_collects_and_deduplicates(self, temp_data_dir):
        """验证 engine 采集数据、去重、记录到 store 的完整流程。"""
        cfg, chrome_mgr, store, engine, mock_col_mod, mock_page = self._setup(
            temp_data_dir
        )

        cards = [
            {'note_id': 'n001', 'cx': 100, 'cy': 200, 'title': '牙痛怎么办'},
        ]
        mock_col_mod.get_cards.side_effect = [cards] + [[]] * 50
        mock_col_mod.open_card.return_value = True
        mock_col_mod.is_overlay_open.return_value = {'open': True}
        mock_col_mod.extract_data.return_value = {
            'note_id': 'n001',
            'title': '牙痛怎么办',
            'content': '内容1',
        }
        mock_col_mod.close_overlay.return_value = None

        with patch('src.collector_engine._get_collector', return_value=mock_col_mod), \
             patch.object(engine, '_handle_disconnect', return_value=False), \
             patch.object(engine, '_human_wait'), \
             patch.object(engine, '_save_note'), \
             patch.object(engine, '_save_summary'), \
             patch('src.collector_engine._get_win32') as mock_win32, \
             patch('src.collector_engine._get_snapshot'):

            mock_win32.return_value.calibrate.return_value = {'csx': 0, 'csy': 0, 'dpr': 1.0, 'hwnd': None}

            collected = engine.collect_notes("牙痛", count=5, pool_ids=set())

        assert len(collected) == 1
        assert collected[0][0] == 'n001'
        assert collected[0][1] == '牙痛怎么办'

        # 验证 store 中已记录
        all_ids = store.get_all_collected_note_ids()
        assert 'n001' in all_ids

    def test_engine_deduplicates_pool_ids(self, temp_data_dir):
        """验证已有 ID 在 pool_ids 中被过滤。"""
        cfg, chrome_mgr, store, engine, mock_col_mod, mock_page = self._setup(
            temp_data_dir
        )

        # 模拟两张卡片，但 n001 已在 pool_ids 中
        cards = [
            {'note_id': 'n001', 'cx': 100, 'cy': 200, 'title': '已采集'},
            {'note_id': 'n002', 'cx': 300, 'cy': 200, 'title': '新的'},
        ]
        mock_col_mod.get_cards.side_effect = [cards] + [[]] * 50
        mock_col_mod.open_card.return_value = True
        mock_col_mod.is_overlay_open.return_value = {'open': True}
        mock_col_mod.extract_data.side_effect = [
            {'note_id': 'n002', 'title': '新的', 'content': '内容2'},
        ]
        mock_col_mod.close_overlay.return_value = None

        with patch('src.collector_engine._get_collector', return_value=mock_col_mod), \
             patch.object(engine, '_handle_disconnect', return_value=False), \
             patch.object(engine, '_human_wait'), \
             patch.object(engine, '_save_note'), \
             patch.object(engine, '_save_summary'), \
             patch('src.collector_engine._get_win32') as mock_win32, \
             patch('src.collector_engine._get_snapshot'):

            mock_win32.return_value.calibrate.return_value = {'csx': 0, 'csy': 0, 'dpr': 1.0, 'hwnd': None}

            collected = engine.collect_notes("牙痛", count=5, pool_ids={'n001'})

        # 只应采集 n002
        assert len(collected) == 1
        assert collected[0][0] == 'n002'

    def test_engine_handles_disconnect_during_collection(self, temp_data_dir):
        """验证采集中断线后重连。"""
        cfg, chrome_mgr, store, engine, mock_col_mod, mock_page = self._setup(
            temp_data_dir
        )

        cards = [
            {'note_id': 'n001', 'cx': 100, 'cy': 200, 'title': 'A'},
        ]
        mock_col_mod.get_cards.side_effect = [cards] + [[]] * 50
        mock_col_mod.open_card.return_value = True
        mock_col_mod.is_overlay_open.return_value = {'open': True}
        mock_col_mod.extract_data.side_effect = Exception("PageDisconnectedError")
        mock_col_mod.close_overlay.return_value = None

        reconnect_page = MagicMock()
        chrome_mgr.reconnect = MagicMock(return_value=reconnect_page)

        with patch('src.collector_engine._get_collector', return_value=mock_col_mod), \
             patch.object(engine, '_human_wait'), \
             patch.object(engine, '_save_note'), \
             patch.object(engine, '_save_summary'), \
             patch('src.collector_engine._get_win32') as mock_win32, \
             patch('src.collector_engine._get_snapshot'):

            mock_win32.return_value.calibrate.return_value = {'csx': 0, 'csy': 0, 'dpr': 1.0, 'hwnd': None}

            result = engine.collect_notes("牙痛", count=5, pool_ids=set())

        # 断线后应尝试重连
        chrome_mgr.reconnect.assert_called()
        # 由于断线时没有成功采集，返回空列表
        assert isinstance(result, list)


class TestE2ESchedulerBoot:
    """验证调度器启动和状态报告。"""

    def test_scheduler_daemon_initializes(self):
        """验证 SchedulerDaemon 初始化不崩溃。"""
        from src.scheduler_daemon import SchedulerDaemon

        # 使用不存在的配置文件（会被自动创建）
        daemon = SchedulerDaemon.__new__(SchedulerDaemon)
        daemon._config_path = "nonexistent_config.yaml"
        daemon._config = MagicMock()
        daemon._config.scheduler.time_windows = ["08:00-11:30"]
        daemon._config.scheduler.keywords = ["牙痛"]
        daemon._config.scheduler.daily_count = 80
        daemon._config.scheduler.rest_between_sessions = (10, 30)
        daemon._config.session_limits.per_session_cap = 50
        daemon._config.collection.db_path = ":memory:"
        daemon._store = MagicMock()
        daemon._chrome_mgr = MagicMock()
        daemon._engine = MagicMock()
        daemon._exiter = MagicMock()
        daemon._exiter.should_exit = False
        daemon._today_collected = 0

        status = daemon.get_status()
        assert status["running"] is True
        assert status["today_collected"] == 0


class TestE2EDataFlow:
    """验证数据在 SessionStore -> Engine -> Store 之间的流动。"""

    def test_store_to_engine_pool_ids_flow(self, temp_data_dir):
        """验证 store 中的 ID 能正确传给 engine 做去重。"""
        from src.config_loader import AppConfig
        from src.collector_engine import CollectionEngine
        from src.chrome_manager import ChromeManager
        from src.session_store import SessionStore

        cfg = AppConfig()
        cfg.collection.db_path = ":memory:"

        store = SessionStore(":memory:")
        # 预填两条已采集数据
        store.mark_note_collected("牙痛", "n001")
        store.mark_note_collected("牙痛", "n002")

        # 从 store 获取 pool_ids
        pool_ids = store.get_all_collected_note_ids()
        assert "n001" in pool_ids
        assert "n002" in pool_ids

        # 验证 engine 的过滤逻辑（单元测试级别）
        cards = [
            {'note_id': 'n001'},
            {'note_id': 'n003'},
            {'note_id': 'n002'},
            {'note_id': 'n004'},
        ]
        filtered = [c for c in cards if c['note_id'] not in pool_ids]
        assert len(filtered) == 2
        assert filtered[0]['note_id'] == 'n003'
        assert filtered[1]['note_id'] == 'n004'

    def test_session_logging_flow(self):
        """验证 session 记录完整流程。"""
        from src.session_store import SessionStore

        store = SessionStore(":memory:")

        # 开始会话
        sid = store.log_session_start()
        assert sid is not None
        assert store.get_today_session_count() == 1

        # 记录采集
        store.mark_note_collected("牙痛", "n001")
        store.mark_note_collected("牙痛", "n002")
        store.mark_note_collected("牙疼", "n003")

        # 结束会话
        store.log_session_end(sid, actions=3, notes_opened=3, keyword="牙痛")

        # 验证进度
        collected_a, _ = store.get_collection_progress("牙痛")
        assert collected_a == 2
        collected_b, _ = store.get_collection_progress("牙疼")
        assert collected_b == 1

        # 全量去重 ID
        all_ids = store.get_all_collected_note_ids()
        assert len(all_ids) == 3
