"""
_common.py — 共享工具模块

集中管理反检测脚本、人类行为模拟参数、通用工具函数。
"""

import json
import random
import signal
import time

# ---------------------------------------------------------------------------
# 反检测 JS — 注入到每个新页面（7 项技术，来自 shuicici/xiaohongshu-scraper）
# ---------------------------------------------------------------------------
ANTI_DETECTION_JS = """
try {
    /* 1. Navigator.webdriver → false（保留属性但设为 false，比 delete 更自然） */
    Object.defineProperty(navigator, 'webdriver', {
        get: function() { return false; },
        configurable: true
    });

    /* 2. 伪造 window.chrome 环境（补全 csi/loadTimes 等真实 Chrome 属性） */
    if (!window.chrome) {
        window.chrome = {
            runtime: { connect: function(){}, sendMessage: function(){} },
            app: {},
            csi: function(){ return {onT: 0, onTt: 0, onTte: 0}; },
            loadTimes: function(){ return {
                requestTime: 0.0, startLoadTime: 0.0, commitLoadTime: 0.0,
                finishDocumentLoadTime: 0.0, finishLoadTime: 0.0,
                wasFetchedViaSpdy: false, wasNpnNegotiated: false, wasAlternateProtocolAvailable: false,
                connectionInfo: 'http/1.1', npnNegotiatedProtocol: 'http/1.1',
            };},
        };
    }

    /* 3. 清除自动化框架痕迹 */
    var _automationProps = ['_phantom', '__nightmare', 'callPhantom', 'puppeteer', 'callSelenium'];
    for (var _i = 0; _i < _automationProps.length; _i++) {
        var _p = _automationProps[_i];
        if (_p in window) delete window[_p];
    }

    /* 4. 清除 CDP 检测属性 */
    var _cdpProps = ['__cdp', '__CDP', '__WEBDRIVER_BRIDGE_TESTS', '__selenium_evaluate', '__webdriverFunc'];
    for (var _j = 0; _j < _cdpProps.length; _j++) {
        var _q = _cdpProps[_j];
        if (_q in window) delete window[_q];
    }

    /* 5. 伪造浏览器语言（中文优先） */
    if (navigator.languages && navigator.languages[0] !== 'zh-CN') {
        Object.defineProperty(navigator, 'languages', {
            get: function() { return ['zh-CN', 'zh', 'en-US', 'en']; },
            configurable: true
        });
    }
    Object.defineProperty(navigator, 'language', {
        get: function() { return 'zh-CN'; },
        configurable: true
    });

    /* 6. 伪造浏览器插件列表 */
    if (navigator.plugins && navigator.plugins.length === 0) {
        var _fakes = [
            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
            { name: 'Widevine Content Decryption Module', filename: 'widevinecdm.dll' },
            { name: 'Native Client', filename: 'pnacl.dll' },
            { name: 'Adobe Flash Player', filename: 'pepflashplayer.dll' }
        ];
        var _arr = { length: _fakes.length };
        for (var _k = 0; _k < _fakes.length; _k++) {
            _arr[_k] = _fakes[_k];
        }
        _arr.item = function(i) { return this[i] || null; };
        _arr.namedItem = function(name) {
            for (var _m = 0; _m < this.length; _m++) {
                if (this[_m].name === name) return this[_m];
            }
            return null;
        };
        _arr[Symbol.iterator] = function() {
            var _idx = 0;
            var _items = this;
            return {
                next: function() {
                    if (_idx < _items.length) { _idx++; return { value: _items[_idx - 1], done: false }; }
                    return { done: true };
                }
            };
        };
        Object.defineProperty(navigator, 'plugins', {
            get: function() { return _arr; },
            configurable: true
        });
    }

    /* 7. 伪造权限查询（覆盖所有敏感权限，避免暴露自动化状态） */
    if (navigator.permissions && navigator.permissions.query) {
        var _origQuery = navigator.permissions.query.bind(navigator.permissions);
        var _safePermissions = { notifications: 'prompt', 'clipboard-read': 'prompt',
            geolocation: 'prompt', camera: 'prompt', microphone: 'prompt' };
        navigator.permissions.query = function(desc) {
            if (desc && desc.name && _safePermissions[desc.name] !== undefined) {
                return Promise.resolve({ state: _safePermissions[desc.name], onchange: null });
            }
            return _origQuery(desc);
        };
    }

    /* 8. 伪造硬件并发数（真实用户通常在 4-16 之间） */
    if (navigator.hardwareConcurrency === undefined || navigator.hardwareConcurrency <= 2) {
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: function() { return 8; }, configurable: true
        });
    }

    /* 9. 伪造设备内存（真实用户通常在 4-8 GB） */
    if (navigator.deviceMemory === undefined || navigator.deviceMemory <= 2) {
        Object.defineProperty(navigator, 'deviceMemory', {
            get: function() { return 8; }, configurable: true
        });
    }
} catch(e) {}
"""

# ---------------------------------------------------------------------------
# 人类行为模拟参数
# ---------------------------------------------------------------------------
HUMAN_CFG = {
    'click_rate': 0.45,
    'break_every_min': 3,
    'break_every_max': 8,
    'break_duration_min': 20,
    'break_duration_max': 60,
    'dwell_weights': [
        (0.35, 0.5, 2.0),
        (0.35, 3.0, 8.0),
        (0.22, 10.0, 30.0),
        (0.08, 35.0, 70.0),
    ],
    'scroll_ranges': [
        (0.25, 80, 250),
        (0.45, 300, 600),
        (0.30, 700, 1300),
    ],
    'detail_scroll_chance': 0.35,
}

DELAY_CFG = {
    'reaction':    (0.3, 0.8),
    'hover':       (0.1, 0.3),
    'read_quick':  (0.5, 1.2),
    'read_normal': (1.2, 2.5),
    'post_scroll': (0.3, 0.5),
    'nav_back':    (0.5, 1.0),
}


def weighted_choice(options):
    """[(weight, val1, val2), ...] → weighted random rest"""
    total = sum(w for w, *_ in options)
    if total <= 0:
        rest = options[0][1:]
        return rest if len(rest) > 1 else rest[0]
    r = random.random() * total
    cum = 0
    for w, *rest in options:
        cum += w
        if r <= cum:
            return rest if len(rest) > 1 else rest[0]
    return options[-1][1:] if len(options[-1]) > 2 else options[-1][1]


def human_dwell_time() -> float:
    low, high = weighted_choice(HUMAN_CFG['dwell_weights'])
    return random.uniform(low, high)


def random_scroll_amount() -> int:
    low, high = weighted_choice(HUMAN_CFG['scroll_ranges'])
    return random.randint(low, high)


def human_delay(page, action='reaction'):
    """在 page 对象上执行随机延迟"""
    a, b = DELAY_CFG.get(action, (0.3, 0.8))
    page.wait(a, b)


def wheel_scroll(page, delta_y: int = 300):
    """模拟滚轮滚动（兼容旧脚本）"""
    page.run_js(f"""
        (() => {{
            const c = document.querySelector('.note-scroller')
                || document.querySelector('[class*="feed"]')
                || document.querySelector('[class*="waterfall"]')
                || document.documentElement;
            c.dispatchEvent(new WheelEvent('wheel', {{
                deltaY: {delta_y}, deltaMode: 0,
                bubbles: true, cancelable: true, view: window
            }}));
        }})();
    """)


def emit(data: dict):
    """输出 JSON 日志，兼容 Windows GBK 终端"""
    try:
        print(json.dumps(data, ensure_ascii=False), flush=True)
    except UnicodeEncodeError:
        print(json.dumps(data, ensure_ascii=True), flush=True)


class GracefulExiter:
    """捕获 Ctrl+C，设置退出标志让主循环优雅退出"""
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        emit({"status": "shutdown", "msg": "收到中断信号，正在优雅退出..."})
        self.should_exit = True


# ---------------------------------------------------------------------------
# SQLite — notes 表定义
# ---------------------------------------------------------------------------
NOTES_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT,
    note_id         TEXT UNIQUE,
    title           TEXT,
    content         TEXT,
    author_name     TEXT,
    author_id       TEXT,
    liked_count     INTEGER DEFAULT 0,
    collected_count INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    share_count     INTEGER DEFAULT 0,
    publish_time    TEXT,
    images          TEXT,
    video_url       TEXT,
    hashtags        TEXT,
    source_url      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
