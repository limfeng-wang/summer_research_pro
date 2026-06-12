"""
config_loader.py — 配置加载与校验
===============================
用法:
    from config_loader import load_config
    cfg = load_config("config.yaml")
    print(cfg.scheduler.daily_count)
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SchedulerConfig:
    daily_count: int = 80
    time_windows: List[str] = field(default_factory=lambda: [
        "08:00-11:30", "13:00-17:00", "19:00-23:00"
    ])
    keywords: List[str] = field(default_factory=lambda: [
        "牙痛", "牙疼", "口腔溃疡"
    ])
    rest_between_sessions: Tuple[int, int] = (10, 30)
    bottom_retry_minutes: Tuple[int, int] = (15, 25)
    random_skip_chance: float = 0.10


@dataclass
class SessionLimitsConfig:
    per_session_cap: int = 50
    max_scrolls: int = 2000
    scroll_stale_limit: int = 5


@dataclass
class ChromeConfig:
    profile_dir: str = "profiles/chrome_main"
    cdp_port: int = 9222
    window_size: Tuple[int, int] = (1400, 900)


@dataclass
class AccountConfig:
    profiles: List[str] = field(default_factory=lambda: ["profiles/chrome_main"])
    random_select: bool = True


@dataclass
class CollectionConfig:
    data_dir: str = "data"
    db_path: str = "data/archive.db"
    detail_scroll_chance: float = 0.35


@dataclass
class AppConfig:
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    session_limits: SessionLimitsConfig = field(default_factory=SessionLimitsConfig)
    chrome: ChromeConfig = field(default_factory=ChromeConfig)
    accounts: AccountConfig = field(default_factory=AccountConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)


# =============================================================================
# Default config template (written when file doesn't exist)
# =============================================================================

DEFAULT_CONFIG_YAML = """\
# xhs-scraper 配置
# 编辑此文件后保存，调度器会自动加载最新配置。
# 所有值都有合理的默认值，可只写想覆盖的字段。

scheduler:
  # 每日总采集量（所有关键词合计，达到后当日停止）
  daily_count: 80

  # 允许运行的时间段（24小时制，可多个）
  # 在每个窗口内，系统会持续循环：采集 → 休息 → 再采集
  # 直到窗口结束、达到日限、或滚动到底且无新数据
  time_windows:
    - "08:00-11:30"
    - "13:00-17:00"
    - "19:00-23:00"

  # 搜索关键词
  keywords:
    - "牙痛"
    - "牙疼"
    - "口腔溃疡"

  # 每个窗口内，两次会话之间的随机休息间隔（分钟）
  rest_between_sessions: [10, 30]

  # 滚动到底且无新卡片后，等待多久再重试一次（分钟）
  # 防止"假底"——内容可能是异步延迟加载的
  bottom_retry_minutes: [15, 25]

  # 随机跳过概率 0.0-1.0
  random_skip_chance: 0.10

session_limits:
  # 单次安全上限（不是目标值，防止某个关键词刷太多）
  per_session_cap: 50

  # 单次会话最多滚动次数
  max_scrolls: 2000

  # 连续多少次滚动无新卡片后，判定为"到底了"
  scroll_stale_limit: 5

chrome:
  profile_dir: "profiles/chrome_main"
  cdp_port: 9222
  window_size: [1400, 900]

# 多账号管理 — 每个 profile 独立存储 cookies/session
accounts:
  # Chrome profile 列表，运行采集时随机选取
  profiles:
    - "profiles/chrome_main"
    # - "profiles/chrome_account2"
  random_select: true

collection:
  data_dir: "data"
  db_path: "data/archive.db"
  detail_scroll_chance: 0.35
"""


# =============================================================================
# Validation
# =============================================================================

_TIME_WINDOW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def validate_time_window(window: str) -> Tuple[int, int, int, int]:
    """校验时间段格式 "HH:MM-HH:MM"，返回 (start_h, start_m, end_h, end_m)。"""
    m = _TIME_WINDOW_RE.match(window)
    if not m:
        raise ValueError(
            f"无效的时间段格式: '{window}'，应为 'HH:MM-HH:MM'，例如 '09:00-11:00'"
        )
    sh, sm, eh, em = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    if not (0 <= sh < 24 and 0 <= sm < 60 and 0 <= eh < 24 and 0 <= em < 60):
        raise ValueError(f"时间数值越界: '{window}'")
    start_total = sh * 60 + sm
    end_total = eh * 60 + em
    if start_total >= end_total:
        raise ValueError(f"开始时间必须早于结束时间: '{window}'")
    return sh, sm, eh, em


def validate_config(cfg: AppConfig):
    """校验配置值，抛出 ValueError。"""
    if cfg.scheduler.daily_count < 1:
        raise ValueError("daily_count 必须 >= 1")
    if not cfg.scheduler.time_windows:
        raise ValueError("至少需要一个 time_window")
    for w in cfg.scheduler.time_windows:
        validate_time_window(w)
    if not cfg.scheduler.keywords:
        raise ValueError("keywords 不能为空")
    if not (0.0 <= cfg.scheduler.random_skip_chance <= 1.0):
        raise ValueError("random_skip_chance 必须在 0.0 ~ 1.0 之间")

    rest = cfg.scheduler.rest_between_sessions
    if isinstance(rest, (list, tuple)) and len(rest) == 2:
        if rest[0] < 1 or rest[1] < rest[0]:
            raise ValueError("rest_between_sessions 必须为 [min, max] 且 min >= 1")
    else:
        raise ValueError("rest_between_sessions 必须为 [min, max]")

    brm = cfg.scheduler.bottom_retry_minutes
    if isinstance(brm, (list, tuple)) and len(brm) == 2:
        if brm[0] < 1 or brm[1] < brm[0]:
            raise ValueError("bottom_retry_minutes 必须为 [min, max] 且 min >= 1")
    else:
        raise ValueError("bottom_retry_minutes 必须为 [min, max]")

    if cfg.session_limits.per_session_cap < 1:
        raise ValueError("per_session_cap 必须 >= 1")
    if cfg.session_limits.max_scrolls < 1:
        raise ValueError("max_scrolls 必须 >= 1")
    if cfg.session_limits.scroll_stale_limit < 1:
        raise ValueError("scroll_stale_limit 必须 >= 1")
    if not (0 <= cfg.collection.detail_scroll_chance <= 1):
        raise ValueError("detail_scroll_chance 必须在 0.0 ~ 1.0 之间")


# =============================================================================
# Load / merge helpers
# =============================================================================

def _merge_dict_into_dataclass(dc, data: dict):
    """将 dict 中的非 None 值合并到 dataclass 实例中。"""
    for key, value in data.items():
        if value is not None and hasattr(dc, key):
            setattr(dc, key, value)


def _dict_to_config(data: dict) -> AppConfig:
    """将解析后的 YAML dict 转换为 AppConfig，缺失字段用默认值。"""
    cfg = AppConfig()

    if "scheduler" in data and isinstance(data["scheduler"], dict):
        sched = data["scheduler"]
        # 处理 list→tuple 字段
        if "rest_between_sessions" in sched and isinstance(sched["rest_between_sessions"], list):
            sched["rest_between_sessions"] = tuple(sched["rest_between_sessions"])
        if "bottom_retry_minutes" in sched and isinstance(sched["bottom_retry_minutes"], list):
            sched["bottom_retry_minutes"] = tuple(sched["bottom_retry_minutes"])
        _merge_dict_into_dataclass(cfg.scheduler, sched)

    if "session_limits" in data and isinstance(data["session_limits"], dict):
        _merge_dict_into_dataclass(cfg.session_limits, data["session_limits"])

    if "chrome" in data and isinstance(data["chrome"], dict):
        chrome_data = data["chrome"].copy()
        if "window_size" in chrome_data and isinstance(chrome_data["window_size"], list):
            chrome_data["window_size"] = tuple(chrome_data["window_size"])
        _merge_dict_into_dataclass(cfg.chrome, chrome_data)

    if "accounts" in data and isinstance(data["accounts"], dict):
        _merge_dict_into_dataclass(cfg.accounts, data["accounts"])

    if "collection" in data and isinstance(data["collection"], dict):
        _merge_dict_into_dataclass(cfg.collection, data["collection"])

    return cfg


# =============================================================================
# Public API
# =============================================================================

def load_config(path: str = "config.yaml") -> AppConfig:
    """
    加载 YAML 配置文件。

    若文件不存在，自动创建默认配置并写入，然后返回默认配置。
    支持部分配置（缺失字段用 dataclass 默认值填充）。
    """
    if not os.path.exists(path):
        try:
            log(f"配置文件 '{path}' 不存在，已创建默认配置")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CONFIG_YAML)
        except (OSError, PermissionError) as e:
            log(f"警告: 无法创建配置文件 '{path}': {e}，使用默认配置")
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        log(f"警告: 配置文件 '{path}' 内容为空，使用默认配置")
        return AppConfig()

    cfg = _dict_to_config(raw)
    validate_config(cfg)
    return cfg


def log(msg: str):
    """简易日志输出，不引入外部依赖。"""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [config] {msg}", flush=True)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    print(f"关键词: {cfg.scheduler.keywords}")
    print(f"日均采集: {cfg.scheduler.daily_count}")
    print(f"时间段: {cfg.scheduler.time_windows}")
    print(f"Session上限: per_session_cap={cfg.session_limits.per_session_cap}")
    print(f"窗口内休息间隔: {cfg.scheduler.rest_between_sessions} 分钟")
    print(f"到底重试间隔: {cfg.scheduler.bottom_retry_minutes} 分钟")
    print(f"Chrome端口: {cfg.chrome.cdp_port}")
    print("✅ 配置文件加载成功")
