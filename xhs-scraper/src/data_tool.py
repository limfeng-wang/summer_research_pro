"""
data_tool.py — 数据管理 CLI

统一管理采集数据：汇总、合并、搜索、导出、统计、去重。

用法:
    python src/data_tool.py collect  --output data/unified.db --dir src/输出数据/ --dir data/
    python src/data_tool.py merge    --output data/unified.db --sources data/a.db data/b.db
    python src/data_tool.py search   --db data/unified.db --keyword "牙痛" --min-likes 100
    python src/data_tool.py export   --db data/unified.db --format json --output export.json
    python src/data_tool.py stats    --db data/unified.db
    python src/data_tool.py dedup    --db data/unified.db
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

NOTES_TABLE = """CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT,
    note_id TEXT UNIQUE,
    title TEXT,
    content TEXT,
    author_name TEXT,
    author_id TEXT,
    liked_count INTEGER DEFAULT 0,
    collected_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    publish_time TEXT,
    images TEXT,
    video_url TEXT,
    hashtags TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


# =============================================================================
# Helpers
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or 'utf-8'
        safe = msg.encode(enc, errors='replace').decode(enc)
        print(f"[{ts}] {safe}", flush=True)


def _extract_keyword(dirname: str) -> str:
    """从目录名 'collect_牙痛_20260611_090038' 提取关键词。"""
    parts = dirname.split('_')
    if len(parts) >= 2:
        return parts[1]
    return ""


# =============================================================================
# NoteDB — SQLite wrapper
# =============================================================================

class NoteDB:
    def __init__(self, db_path: str):
        self._path = db_path
        self._conn = None

    def connect(self):
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(NOTES_TABLE)
        self._conn.commit()
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert(self, note: dict, keyword=None):
        """插入单条笔记（INSERT OR IGNORE 去重）。"""
        if keyword:
            note['keyword'] = keyword
        images_json = json.dumps(note.get('images', []), ensure_ascii=False) if isinstance(note.get('images'), list) else note.get('images', '[]')
        hashtags_json = json.dumps(note.get('hashtags', []), ensure_ascii=False) if isinstance(note.get('hashtags'), list) else note.get('hashtags', '[]')
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO notes
                   (keyword, note_id, title, content, author_name, author_id,
                    liked_count, collected_count, comment_count, share_count,
                    publish_time, images, video_url, hashtags)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (note.get('keyword', ''), note.get('note_id', ''),
                 note.get('title', ''), note.get('content', ''),
                 note.get('author_name', ''), note.get('author_id', ''),
                 note.get('liked_count', 0) or 0,
                 note.get('collected_count', 0) or 0,
                 note.get('comment_count', 0) or 0,
                 note.get('share_count', 0) or 0,
                 note.get('publish_time', ''),
                 images_json, note.get('video_url', ''),
                 hashtags_json))
            return True
        except sqlite3.IntegrityError:
            return False

    def merge_from(self, source_db: str) -> tuple:
        """合并另一个 .db 文件到当前库，返回 (inserted, skipped)。"""
        try:
            src = sqlite3.connect(source_db)
        except Exception as e:
            log(f"  无法打开 {source_db}: {e}")
            return 0, 0
        try:
            rows = list(src.execute("SELECT keyword, note_id, title, content, author_name, author_id, liked_count, collected_count, comment_count, share_count, publish_time, images, video_url, hashtags FROM notes"))
        except Exception:
            src.close()
            return 0, 0
        src.close()

        total_before = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        self._conn.executemany(
            """INSERT OR IGNORE INTO notes
               (keyword, note_id, title, content, author_name, author_id,
                liked_count, collected_count, comment_count, share_count,
                publish_time, images, video_url, hashtags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        self._conn.commit()
        total_after = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        inserted = total_after - total_before
        skipped = len(rows) - inserted
        return inserted, skipped

    def search(self, keyword=None, author=None, min_likes=0, min_collects=0, limit=200):
        """搜索笔记。"""
        wheres = []
        params = []
        if keyword:
            wheres.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        if author:
            wheres.append("author_name LIKE ?")
            params.append(f"%{author}%")
        if min_likes > 0:
            wheres.append("liked_count >= ?")
            params.append(min_likes)
        if min_collects > 0:
            wheres.append("collected_count >= ?")
            params.append(min_collects)
        where = "WHERE " + " AND ".join(wheres) if wheres else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT note_id, title, author_name, liked_count, collected_count, comment_count, publish_time FROM notes {where} ORDER BY liked_count DESC LIMIT ?",
            params).fetchall()
        return rows

    def stats(self) -> dict:
        """返回统计摘要。"""
        return {
            'total': self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
            'keywords': {
                r[0]: r[1] for r in
                self._conn.execute("SELECT keyword, COUNT(*) FROM notes GROUP BY keyword ORDER BY COUNT(*) DESC").fetchall()
            },
            'top_authors': [
                {'name': r[0], 'count': r[1], 'avg_likes': round(r[2], 1)}
                for r in self._conn.execute(
                    "SELECT author_name, COUNT(*), AVG(liked_count) FROM notes GROUP BY author_name ORDER BY COUNT(*) DESC LIMIT 10"
                ).fetchall()
            ],
            'date_range': self._conn.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM notes"
            ).fetchone(),
            'avg_likes': round(self._conn.execute("SELECT AVG(liked_count) FROM notes").fetchone()[0] or 0, 1),
            'avg_collects': round(self._conn.execute("SELECT AVG(collected_count) FROM notes").fetchone()[0] or 0, 1),
            'total_likes': self._conn.execute("SELECT SUM(liked_count) FROM notes").fetchone()[0] or 0,
            'with_images': self._conn.execute("SELECT COUNT(*) FROM notes WHERE images != '[]' AND images != ''").fetchone()[0],
            'with_video': self._conn.execute("SELECT COUNT(*) FROM notes WHERE video_url != ''").fetchone()[0],
        }

    def dedup_report(self) -> list:
        """查找重复 title（宽松匹配）。"""
        rows = self._conn.execute("""
            SELECT title, COUNT(*) AS cnt, GROUP_CONCAT(note_id, ',') AS ids
            FROM notes
            WHERE title != ''
            GROUP BY title
            HAVING cnt > 1
            ORDER BY cnt DESC
            LIMIT 20
        """).fetchall()
        return rows

    def export_all(self, filters=None):
        """导出全部笔记（生成器）。"""
        query = "SELECT keyword, note_id, title, content, author_name, author_id, liked_count, collected_count, comment_count, share_count, publish_time, images, video_url, hashtags FROM notes"
        if filters:
            wheres = []
            params = []
            if filters.get('keyword'):
                wheres.append("keyword LIKE ?")
                params.append(f"%{filters['keyword']}%")
            if wheres:
                query += " WHERE " + " AND ".join(wheres)
            query += " ORDER BY liked_count DESC"
            return self._conn.execute(query, params).fetchall()
        query += " ORDER BY liked_count DESC"
        return self._conn.execute(query).fetchall()


# =============================================================================
# Collector — scan directories for JSON notes
# =============================================================================

def collect_to_db(db: NoteDB, directories: list):
    """扫描目录树，提取 JSON 笔记并入库。"""
    total_inserted = 0
    total_skipped = 0

    for dir_path in directories:
        root = Path(dir_path)
        if not root.exists():
            log(f"目录不存在，跳过: {dir_path}")
            continue

        # 搜索 collect_* 目录
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name
            if not (name.startswith('collect_') or name.startswith('snapshot_')):
                continue

            keyword = _extract_keyword(name)
            notes = []

            # 优先读 all_notes.json
            all_json = subdir / "all_notes.json"
            if all_json.exists():
                try:
                    with open(all_json, 'r', encoding='utf-8') as f:
                        notes = json.load(f)
                        if isinstance(notes, list) and notes:
                            if not notes[0].get('keyword'):
                                for n in notes:
                                    n['keyword'] = keyword
                except Exception:
                    pass

            # 回退：逐个读 note_parsed.json
            if not notes:
                for note_dir in sorted(subdir.iterdir()):
                    if not note_dir.is_dir() or not note_dir.name.startswith('note_'):
                        continue
                    parsed = note_dir / "note_parsed.json"
                    if parsed.exists():
                        try:
                            with open(parsed, 'r', encoding='utf-8') as f:
                                n = json.load(f)
                                n['keyword'] = keyword
                                notes.append(n)
                        except Exception:
                            pass

            inserted = 0
            for i, n in enumerate(notes):
                if db.insert(n):
                    inserted += 1
                if (i + 1) % 20 == 0:
                    db._conn.commit()
            db._conn.commit()  # final batch
            skipped = len(notes) - inserted
            total_inserted += inserted
            total_skipped += skipped

            if notes:
                log(f"  {name}: {inserted} 新 / {skipped} 跳过 ({len(notes)} 总)")
            else:
                log(f"  {name}: 无笔记数据")

    return total_inserted, total_skipped


# =============================================================================
# CLI
# =============================================================================

def cmd_collect(args):
    db = NoteDB(args.output)
    db.connect()
    log(f"输出: {args.output}")
    log(f"扫描目录: {args.dir}")
    inserted, skipped = collect_to_db(db, args.dir)
    log(f"\n完成: {inserted} 新入库, {skipped} 去重跳过")
    db.close()


def cmd_merge(args):
    db = NoteDB(args.output)
    db.connect()
    log(f"输出: {args.output}")
    total_ins, total_skip = 0, 0
    for src in args.sources:
        if not os.path.exists(src):
            log(f"文件不存在，跳过: {src}")
            continue
        ins, skip = db.merge_from(src)
        total_ins += ins
        total_skip += skip
        log(f"  {os.path.basename(src)}: +{ins} / -{skip}")
    log(f"\n完成: {total_ins} 新入库, {total_skip} 去重跳过")
    db.close()


def cmd_search(args):
    db = NoteDB(args.db)
    db.connect()
    rows = db.search(keyword=args.keyword, author=args.author,
                     min_likes=args.min_likes, min_collects=args.min_collects,
                     limit=args.limit)
    if not rows:
        log("无匹配结果。")
    else:
        log(f"找到 {len(rows)} 条:")
        # 表头
        log(f"{'note_id':<18} {'title':<30} {'author':<14} {'likes':<7} {'collects':<9} {'time':<12}")
        log("-" * 95)
        for r in rows:
            nid = (r[0] or '')[:16]
            title = (r[1] or '')[:28]
            author = (r[2] or '')[:12]
            likes = str(r[3] or 0)
            collects = str(r[4] or 0)
            time_str = (r[6] or '')[:10]
            log(f"{nid:<18} {title:<30} {author:<14} {likes:<7} {collects:<9} {time_str:<12}")
    db.close()


def cmd_export(args):
    db = NoteDB(args.db)
    db.connect()
    filters = {}
    if hasattr(args, 'keyword') and args.keyword:
        filters['keyword'] = args.keyword
    rows = db.export_all(filters)
    log(f"导出 {len(rows)} 条到 {args.output}")

    if args.format == 'csv':
        with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['keyword', 'note_id', 'title', 'content', 'author_name', 'author_id',
                             'liked_count', 'collected_count', 'comment_count', 'share_count',
                             'publish_time', 'images', 'video_url', 'hashtags'])
            for r in rows:
                # 标准化 images / hashtags 为 JSON 字符串
                row = list(r)
                for i in (11, 13):  # images, hashtags columns
                    try:
                        if isinstance(row[i], str) and row[i].startswith('['):
                            pass  # already JSON string
                        elif isinstance(row[i], list):
                            row[i] = json.dumps(row[i], ensure_ascii=False)
                    except Exception:
                        row[i] = '[]'
                writer.writerow(row)
    else:
        notes = []
        for r in rows:
            n = {
                'keyword': r[0], 'note_id': r[1], 'title': r[2], 'content': r[3],
                'author_name': r[4], 'author_id': r[5],
                'liked_count': r[6], 'collected_count': r[7], 'comment_count': r[8],
                'share_count': r[9], 'publish_time': r[10],
                'images': r[11], 'video_url': r[12], 'hashtags': r[13],
            }
            # 把 JSON 字符串还原为数组
            for field in ('images', 'hashtags'):
                try:
                    if isinstance(n[field], str) and n[field].startswith('['):
                        n[field] = json.loads(n[field])
                except Exception:
                    n[field] = []
            notes.append(n)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

    log(f"已保存: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    db.close()


def cmd_stats(args):
    db = NoteDB(args.db)
    db.connect()
    s = db.stats()
    log(f"\n{'='*50}")
    log(f"  总笔记数: {s['total']}")
    log(f"  总点赞数: {s['total_likes']}")
    log(f"  平均点赞: {s['avg_likes']}")
    log(f"  平均收藏: {s['avg_collects']}")
    log(f"  有图片:   {s['with_images']}")
    log(f"  有视频:   {s['with_video']}")
    if s['date_range'][0]:
        log(f"  时间范围: {s['date_range'][0]} ~ {s['date_range'][1]}")
    log(f"  关键词分布:")
    for kw, cnt in s['keywords'].items():
        log(f"    {kw}: {cnt}")
    log(f"  Top 作者:")
    for a in s['top_authors']:
        log(f"    {a['name'][:14]:<16} {a['count']}篇  avg likes={a['avg_likes']}")
    log(f"{'='*50}")
    db.close()


def cmd_dedup(args):
    db = NoteDB(args.db)
    db.connect()
    rows = db.dedup_report()
    if not rows:
        log("无重复笔记。")
    else:
        log(f"发现 {len(rows)} 组疑似重复:\n")
        for r in rows:
            title = (r[0] or '')[:50]
            log(f"  [{r[1]}x] {title}")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="xhs-scraper 数据管理工具")
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="扫描目录，汇总 JSON 笔记到 SQLite")
    p_collect.add_argument("--output", required=True, help="输出 .db 路径")
    p_collect.add_argument("--dir", nargs="+", required=True, help="要扫描的目录")

    p_merge = sub.add_parser("merge", help="合并多个 .db 文件")
    p_merge.add_argument("--output", required=True, help="输出 .db 路径")
    p_merge.add_argument("--sources", nargs="+", required=True, help="源 .db 文件")

    p_search = sub.add_parser("search", help="搜索笔记")
    p_search.add_argument("--db", required=True, help="数据库路径")
    p_search.add_argument("--keyword", default=None)
    p_search.add_argument("--author", default=None)
    p_search.add_argument("--min-likes", type=int, default=0)
    p_search.add_argument("--min-collects", type=int, default=0)
    p_search.add_argument("--limit", type=int, default=200)

    p_export = sub.add_parser("export", help="导出数据")
    p_export.add_argument("--db", required=True, help="数据库路径")
    p_export.add_argument("--format", choices=["json", "csv"], default="json")
    p_export.add_argument("--output", required=True, help="导出文件路径")
    p_export.add_argument("--keyword", default=None, help="按关键词过滤")

    p_stats = sub.add_parser("stats", help="统计摘要")
    p_stats.add_argument("--db", required=True, help="数据库路径")

    p_dedup = sub.add_parser("dedup", help="去重报告")
    p_dedup.add_argument("--db", required=True, help="数据库路径")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        'collect': cmd_collect, 'merge': cmd_merge, 'search': cmd_search,
        'export': cmd_export, 'stats': cmd_stats, 'dedup': cmd_dedup,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
