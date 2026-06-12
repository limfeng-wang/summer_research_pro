# xhs-scraper — 小红书搜索页数据采集工具

基于 Chrome CDP 协议的小红书搜索内容采集系统。不逆向、不破解，通过持久化 Chrome Profile 保持登录态，模拟真人操作频率，安全稳定地采集公开的搜索结果数据。

> ⚠️ **仅供学术研究使用**，请勿用于商业用途或大规模高频采集。

## 工作原理

```
用户扫码登录(一次) → Chrome 持久化 Profile → 调度器按时间窗口自动运行
                                                      ↓
                                              搜索关键词 → 滚动加载
                                                      ↓
                                              SSR 提取 + DOM 回退
                                                      ↓
                                              SQLite + JSON 存储
```

**核心特点：**

- **持久化登录** — 首次扫码后 session 保存在 Chrome Profile，后续无需重复登录
- **反检测** — 7 项浏览器特征遮蔽 + 人类行为模拟（随机间隔、滚动停顿）
- **双路径提取** — SSR `__INITIAL_STATE__` 为主，DOM 解析为回退
- **时间窗口调度** — 可配时间段自动运行，随机跳过、随机休息，模拟真人作息
- **Windows API 鼠标模拟** — `SendInput` 级别的点击事件，不触发 CDP 自动化检测

## 环境要求

- **Python** ≥ 3.8
- **Google Chrome**（任意最新版本）
- **Windows**（鼠标模拟模块依赖 Win32 API；macOS/Linux 可运行核心采集，但需移除 `win32_api` 依赖）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/xhs-scraper.git
cd xhs-scraper
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 创建配置文件

```bash
copy config.example.yaml config.yaml
```

然后编辑 `config.yaml`，修改你想要搜索的关键词和时间窗口。

### 4. 启动 Chrome 并扫码登录

```bash
python chrome_launcher.py
```

这会启动一个独立的 Chrome 窗口，自动打开小红书首页。**在弹出的 Chrome 窗口中手动扫码登录**（和手机 App 扫码一样）。

登录成功后，Chrome 会保持在后台运行。不要关闭启动脚本的终端。

### 5. 启动采集

新开一个终端：

```bash
python -m src.scheduler_daemon
```

调度器会在配置的时间窗口内自动采集。你也可以直接单次采集：

```bash
python src/xhs_collector_v2.py --keyword "牙痛" --max 50
```

### 6. 查看数据

采集的数据保存在 `data/` 目录下：

- `data/archive.db` — SQLite 数据库（所有笔记的结构化数据）
- `data/collect_{关键词}_{时间戳}/` — 每次采集的 JSON + 截图

使用数据工具导出：

```bash
python -m src.data_tool export --format csv --output results.csv
```

## 项目结构

```
xhs-scraper/
├── chrome_launcher.py        # Chrome CDP 启动器（含扫码登录流程）
├── run_night_owl.bat         # 一键启动批处理
├── config.example.yaml       # 配置示例
├── requirements.txt          # Python 依赖
│
├── src/
│   ├── scheduler_daemon.py   # 【主入口】调度守护进程
│   ├── collector_engine.py   # 采集引擎
│   ├── chrome_manager.py     # Chrome 生命周期管理 + Session 过期检测
│   ├── config_loader.py      # YAML 配置加载与校验
│   ├── session_store.py      # 会话进度持久化 (SQLite)
│   ├── state_reader.py       # SSR brace-matching 提取算法
│   ├── feed_adapter.py       # 数据结构化字段映射
│   ├── fetch_policy.py       # 详情页开启多级决策
│   ├── agent_brain.py        # AI 决策引擎 (LLM + 规则双模式)
│   ├── account_manager.py    # 多账号随机选取
│   ├── data_tool.py          # 数据导出工具
│   │
│   ├── xhs_snapshot.py       # 【核心库】Chrome连接 + SSR提取 + 截图
│   ├── xhs_collector_v2.py   # 主采集器 (精简版 50 条管线)
│   ├── xhs_accumulator.py    # 长期微会话积累系统
│   │
│   ├── win32_api.py          # Windows API 鼠标/窗口操作
│   ├── step1_mouse_takeover.py  # 鼠标接管实验
│   ├── step2_position_lock.py   # DPI 坐标校准
│   ├── step3_click_sim.py       # SendInput 点击模拟
│   ├── step3b_click_diag.py     # CDP vs SendInput 事件对比
│   ├── step3c_event_compare.py  # CDP 完整事件序列验证
│   ├── step4_exit_sim.py        # 浮层关闭模拟
│   │
│   └── _common.py            # 反检测 JS (7项) + 共享工具
│
├── docs/
│   ├── PROJECT_ARCHITECTURE.md  # 系统架构文档
│   └── STATUS.md                # 项目状态报告
│
└── tests/
    ├── test_phase_1.py         # 调度 + 账号体系测试
    ├── test_phase_2.py         # 搜索采集测试
    ├── test_phase_3.py         # 详情页提取测试
    └── test_phase_4.py         # 鼠标模拟测试
```

## 配置说明

详见 `config.example.yaml`。关键配置项：

| 配置 | 说明 |
|------|------|
| `scheduler.keywords` | 搜索关键词列表 |
| `scheduler.time_windows` | 允许运行的时间段 |
| `scheduler.daily_count` | 每日采集总量上限 |
| `scheduler.rest_between_sessions` | 会话间随机休息（分钟） |
| `session_limits.per_session_cap` | 单次会话采集上限 |
| `chrome.profile_dir` | Chrome Profile 目录（首次自动创建） |

## 常见问题

**Q: 启动后提示 "Session 已过期"？**  
A: 关闭 Chrome，重新运行 `python chrome_launcher.py`，再次扫码登录即可。

**Q: 采集不到数据？**  
A: 检查 Chrome 是否还保持着登录状态、网络是否正常。可以先用浏览器打开 `https://www.xiaohongshu.com` 确认登录有效。

**Q: macOS/Linux 能用吗？**  
A: 核心采集流程（搜索、提取、存储）可以运行，但 `win32_api` 鼠标模拟模块仅 Windows 可用。非 Windows 环境需将点击操作改为 CDP 方式。

**Q: 会被封号吗？**  
A: 项目设计时已内置多项安全策略（持久化 Profile、随机间隔、低频采集、不主动触发风控），但任何自动化操作都存在风险。建议使用小号，控制每日采集量。

## 免责声明

本项目仅供学术研究和技术学习使用。使用者应遵守小红书平台的用户协议和相关法律法规，不得将本项目用于任何商业用途或大规模数据采集。作者不对使用本项目产生的任何后果承担责任。
