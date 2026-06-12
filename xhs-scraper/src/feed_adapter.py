#!/usr/bin/env python3
"""
feed_adapter.py — 数据适配层，隔离小红书前端字段名变化。

所有对外暴露的字段路径集中定义在此，XHS 改字段名只改这一个文件。
"""

from datetime import datetime


# ---------------------------------------------------------------------------
# 安全取值工具
# ---------------------------------------------------------------------------

def deep_get(obj, path, default=None):
    """
    安全的嵌套字典取值。

    - `deep_get(d, 'a.b.c')` → d['a']['b']['c']
    - `deep_get(d, ['a.b.c', 'a.b.d'])` → 尝试多个路径，取第一个非 None
    """
    if isinstance(path, (list, tuple)):
        for p in path:
            v = deep_get(obj, p, None)
            if v is not None:
                return v
        return default
    for key in path.split('.'):
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
        if obj is None:
            return default
    return obj


def deep_int(obj, path, default=0):
    v = deep_get(obj, path, None)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# FeedCardAdapter — 搜索结果页卡片数据
# ---------------------------------------------------------------------------

class FeedCardAdapter:
    """
    将 __INITIAL_STATE__.search.feeds[n] 的原始条目转换为统一格式。

    已知 XHS 字段名并存情况：
      - user.nickname / user.nickName（大小写不同）
      - interactInfo.likedCount / interactInfo.liked（字段冗余）
      - interactInfo.collectedCount / interactInfo.collected
    """

    @staticmethod
    def parse(raw: dict) -> dict:
        card = raw.get('noteCard') or raw
        user = card.get('user') if isinstance(card.get('user'), dict) else {}
        ii = card.get('interactInfo') if isinstance(card.get('interactInfo'), dict) else {}

        return {
            'note_id':    raw.get('id') or card.get('noteId') or '',
            'xsec_token': raw.get('xsecToken') or card.get('xsecToken') or '',
            'title':      card.get('displayTitle') or card.get('title') or '',
            'type':       card.get('type') or 'normal',

            'author_name': deep_get(user, ['nickname', 'nickName']) or '',
            'author_id':   user.get('userId') or '',

            'liked_count':     deep_int(ii, ['likedCount', 'liked']),
            'collected_count': deep_int(ii, ['collectedCount', 'collected']),
            'comment_count':   deep_int(ii, ['commentCount', 'commenCount']),
            'share_count':     deep_int(ii, ['sharedCount', 'shareCount']),
        }


# ---------------------------------------------------------------------------
# DetailPageAdapter — 详情页笔记完整数据
# ---------------------------------------------------------------------------

class DetailPageAdapter:
    """
    将详情页 noteDetailMap[noteId].note 的原始数据转换为统一格式。
    输入必须是 SSR script 提取的原始 dict（非 Vue reactive proxy）。
    """

    @staticmethod
    def parse(note: dict) -> dict | None:
        if not note or not note.get('noteId'):
            return None

        user = note.get('user') or {}
        ii = note.get('interactInfo') or {}

        ts = note.get('time', 0)
        pub_time = ''
        if ts:
            try:
                pub_time = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            except (OSError, ValueError):
                pass

        images = note.get('imageList') or []
        images_str = '; '.join(
            img.get('urlDefault', '') for img in images if isinstance(img, dict) and img.get('urlDefault')
        )

        tags = []
        for t in (note.get('tagList') or []):
            if isinstance(t, dict) and t.get('name'):
                tags.append(t['name'])

        video_url = ''
        video = note.get('video')
        if isinstance(video, dict):
            media = video.get('media') or {}
            streams = media.get('stream') or {}
            for codec in ('h265', 'h264'):
                for s in (streams.get(codec) or []):
                    if isinstance(s, dict) and s.get('masterUrl'):
                        video_url = s['masterUrl']
                        break
                if video_url:
                    break

        return {
            'note_id':     note.get('noteId', ''),
            'title':       note.get('title', ''),
            'content':     note.get('desc', ''),
            'author_name': deep_get(user, ['nickname', 'nickName']) or '',
            'author_id':   user.get('userId', ''),
            'liked_count':     deep_int(ii, ['likedCount', 'liked']),
            'collected_count': deep_int(ii, ['collectedCount', 'collected']),
            'comment_count':   deep_int(ii, ['commentCount', 'commenCount']),
            'share_count':     deep_int(ii, ['sharedCount', 'shareCount']),
            'publish_time': pub_time,
            'images':      images_str,
            'video_url':   video_url,
            'hashtags':    '; '.join(tags),
        }
