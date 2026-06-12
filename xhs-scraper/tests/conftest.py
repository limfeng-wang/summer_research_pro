"""
conftest.py — 共享 fixtures 和 mock 工具
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_config_file():
    """返回一个临时配置文件路径，测试结束后自动清理。"""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


@pytest.fixture
def temp_data_dir():
    """返回一个临时数据目录。"""
    tmp = tempfile.mkdtemp()
    yield tmp
    import shutil
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass


@pytest.fixture
def mock_page():
    """
    返回一个模拟的 ChromiumPage 对象。
    所有调用都返回可控的 MagicMock 值。
    """
    page = MagicMock()
    page.url = "https://www.xiaohongshu.com/search_result?keyword=牙痛"
    page.title = "搜索结果 - 小红书"
    page.run_js.return_value = "[]"
    page.run_cdp.return_value = {}
    return page


@pytest.fixture
def mock_chrome_manager():
    """返回一个自动 mock 的 ChromeManager。"""
    with patch('src.chrome_manager.ChromeManager') as mock:
        instance = mock.return_value
        instance.ensure_connection.return_value = MagicMock()
        instance.reconnect.return_value = MagicMock()
        instance.detect_session_expiry.return_value = (False, "")
        yield instance
