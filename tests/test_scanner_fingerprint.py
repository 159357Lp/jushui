"""扫描器缓存指纹的回归测试。

守卫的是一条静默失效路径：parse_gguf 新增字段时，文件大小和 mtime 都不会变，
若指纹只由这两项组成，已入库的模型记录会永远命中缓存、永远不会重解析，
新字段（KV 几何、专家维度）对存量模型就等于没加——规划器只会一路走兜底估算，
而日志上看不出任何异常。PARSER_SCHEMA 进指纹就是为了切断这条路。
"""
from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.infra.gguf_parser import PARSER_SCHEMA  # noqa: E402

# lmloader.infra 的惰性导出把单例对象命名为 scanner，和子模块同名，
# `from lmloader.infra import scanner` 拿到的是那个实例而不是模块。
sc = importlib.import_module("lmloader.infra.scanner")


def stat_of(size=100, mtime=200):
    return SimpleNamespace(st_size=size, st_mtime_ns=mtime)


def test_schema_bump_invalidates_every_cached_record(monkeypatch):
    before = sc._fingerprint(stat_of())
    monkeypatch.setattr(sc, "PARSER_SCHEMA", PARSER_SCHEMA + 1)
    after = sc._fingerprint(stat_of())
    assert before != after, "解析器加了字段但缓存指纹没变 → 存量模型永远读不到新字段"


def test_fingerprint_starts_with_schema_version():
    assert sc._fingerprint(stat_of()).startswith(f"{PARSER_SCHEMA}:")


def test_file_change_still_invalidates():
    assert sc._fingerprint(stat_of(size=101)) != sc._fingerprint(stat_of(size=100))
    assert sc._fingerprint(stat_of(mtime=201)) != sc._fingerprint(stat_of(mtime=200))
