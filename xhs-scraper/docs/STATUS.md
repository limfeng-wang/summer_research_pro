# 项目状态报告 — 小红书数据采集

> 日期: 2026-05-17 | 状态: **Phase 1 完成，Phase 2 代码写完，采用持久化 Profile 方案**

## 已完成的

| 模块 | 状态 | 说明 |
|------|------|------|
| Phase 1 调度+账号体系 (8个模块) | ✅ 通过 | scheduler / accounts / merge_dbs / session_runner |
| `behavior_sim.py` | ✅ 实测通过 | `human_inertia_scroll` 和 `human_move_and_click` 在 XHS 页面实测可用 |
| `search_collector.py` | ✅ 已写 | `page.listen` + fetch 双通道，待有 session 后调试详情提取路径 |
| SSR 详情页提取 (`state_reader.py`) | ✅ 可用 | explore 页 SSR 含完整笔记数据 |
| API 端点确认 (`search/notes`) | ✅ 已确认 | 需登录态，返回 code=0 时有 items |
| `chrome_launcher.py --login` | ✅ 新增 | 一次性扫码登录流程，session 保存在 Chrome Profile |

## 已清理（废弃方法）

| 文件 | 处理 |
|------|------|
| `xhs_session_cookies.json` | ❌ 删除 |
| `get_session_from_real_chrome.py` | 🚫 标记为 DEPRECATED |
| `login_playwright.py` | 🚫 标记为 DEPRECATED |

**弃用原因**：XHS 将 session 绑定在浏览器指纹 + 完整 Profile 上，单纯搬运 Cookie 值无法通过校验。

## 当前策略

不再搬运 Cookie。改用 Chrome 持久化 Profile 方案：

```
首次:  python chrome_launcher.py --login
       → 关闭所有 Chrome → 以 CDP 模式重启 → 手动扫码登录一次
       → session 自动保存在 Chrome Profile 磁盘上
       → Chrome 保持在后台运行

日常:  python chrome_launcher.py --auto
       python xhs_browse/safe_collector.py --keyword "牙痛" --max 20
       → 连接同一个 Chrome，session 自然有效
```

## 待验证

如果登录成功但 DrissionPage 连接后 session 失效，说明 DrissionPage 自身暴露了自动化特征。届时需要 Plan B：将 `safe_collector.py` 从 DrissionPage 迁移到原生 CDP WebSocket（传输层替换，行为模拟逻辑保留）。

## 优先任务

1. **执行首次登录** — `python chrome_launcher.py --login`
2. **测试采集** — `python xhs_browse/safe_collector.py --keyword "牙痛" --max 5`
3. **如失败 → Plan B** — 将 DrissionPage 调用替换为原生 CDP WebSocket
