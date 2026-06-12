"""
account_manager.py — 多账号管理

职责:
  - 从配置中读取可用账号列表 (Chrome profile 目录)
  - 随机/固定选择一个 profile
  - 自动回退：无 accounts 配置时用 chrome.profile_dir

用法:
    from account_manager import AccountManager
    mgr = AccountManager(config, project_root="/path/to/project")
    profile = mgr.select_profile()  # "profiles/chrome_main"
"""

import random
from pathlib import Path


class AccountManager:
    def __init__(self, config, project_root=None):
        self._config = config
        self._root = Path(project_root or Path(__file__).resolve().parent.parent)

    def _get_profiles(self):
        """获取可用 profile 列表。"""
        try:
            profiles = self._config.accounts.profiles
            if profiles:
                return profiles
        except AttributeError:
            pass
        try:
            return [self._config.chrome.profile_dir]
        except AttributeError:
            return ["profiles/chrome_main"]

    def select_profile(self) -> str:
        """
        选择一个 Chrome profile 路径。

        如果 accounts.random_select=True，从列表中随机选择一个；
        否则返回第一个（默认）profile。
        """
        profiles = self._get_profiles()
        if not profiles:
            return "profiles/chrome_main"
        try:
            random_select = self._config.accounts.random_select
        except AttributeError:
            random_select = False
        if random_select and len(profiles) > 1:
            return random.choice(profiles)
        return profiles[0]

    def get_all_profiles(self) -> list:
        return self._get_profiles()

    def resolve_path(self, profile_name: str) -> str:
        """将相对 profile 路径转为绝对路径。"""
        p = Path(profile_name)
        if p.is_absolute():
            return str(p)
        return str(self._root / profile_name)
