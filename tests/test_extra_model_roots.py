"""从别的盘挑模型：登记、记住目录、复扫 的回归测试。

D 盘放不下，模型库挪到了别的盘——这个功能的全部价值在于"后续加载还能找到那只
文件"。所以要守卫的不是"能不能登记"，而是登记之后：路径会不会被"扫描文件"抹掉、
记住的目录下次扫描会不会真的再走一遍、同名两条路径都在盘上时会不会被悄悄改指。

每个"会生效"的判据都配一条反向对照：文件真没了必须被清除、目录真移除了必须不再走、
旧路径真没了才允许改指。缺了对照，测试只会一直绿。
"""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path

import pytest
from gguf import GGUFWriter

# lmloader.infra 的惰性导出把单例命名为 scanner，和子模块同名，
# `from lmloader.infra import scanner` 拿到的是实例而不是模块。
sc = importlib.import_module("lmloader.infra.scanner")
db_mod = importlib.import_module("lmloader.infra.database")
from lmloader.config import settings  # noqa: E402


def write_gguf(path: Path, name: str = "Probe") -> Path:
    """A real (tiny) GGUF: the scanner parses the header, so a stub file won't register."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), "llama")
    w.add_name(name)
    w.add_block_count(1)
    w.add_context_length(512)
    w.add_embedding_length(8)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A fresh DB plus two directories: the configured one and an external drive."""
    file_db = tmp_path / "lmloader.db"
    monkeypatch.setattr(db_mod.db, "db_path", file_db)
    asyncio.run(db_mod.db.init())

    cfg_dir = tmp_path / "d-drive" / "models"
    ext_dir = tmp_path / "e-drive" / "library"
    cfg_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)

    async def no_ollama(self):
        return []

    monkeypatch.setattr(sc.ModelScanner, "_scan_ollama", no_ollama)
    # settings.model_dirs 是只读 property，真正的字段在它包着的 pydantic 模型上
    monkeypatch.setattr(settings.config, "model_dirs", [str(cfg_dir)])
    return type("Roots", (), {"cfg": cfg_dir, "ext": ext_dir, "scanner": sc.scanner})


def rows():
    return {m.name: m.path for m in asyncio.run(db_mod.db.list_models())}


def roots_of(scn):
    return asyncio.run(scn._roots())


# ---------------------------------------------------------------- 登记与留存


def test_picked_external_file_keeps_its_path_through_a_later_scan(roots):
    """用户问的是"保证后续加载"：外部路径登记后，扫描文件不能把它清掉。"""
    target = write_gguf(roots.ext / "alpha.gguf")
    result = asyncio.run(roots.scanner.register_paths([str(target)]))

    assert result["registered"] == ["alpha.gguf"]
    assert rows()["alpha.gguf"] == str(target)

    pruned = asyncio.run(roots.scanner.scan_and_register())

    assert "alpha.gguf" not in pruned["removed"], (
        "外部盘的模型一按\"扫描文件\"就被当成残留删掉了 —— 用户下次得重新挑一遍"
    )
    assert rows()["alpha.gguf"] == str(target)


def test_scan_still_prunes_a_record_whose_file_really_disappeared(roots):
    """反向对照：上一条没删是因为文件还在，不是因为清理逻辑不生效。"""
    target = write_gguf(roots.ext / "gone.gguf")
    asyncio.run(roots.scanner.register_paths([str(target)]))
    target.unlink()

    pruned = asyncio.run(roots.scanner.scan_and_register())

    assert "gone.gguf" in pruned["removed"]
    assert "gone.gguf" not in rows()


# ------------------------------------------------------------------ 记住目录


def test_picked_directory_is_walked_again_by_the_next_scan(roots):
    """记下的目录必须进扫描根，之后新放进去的模型不用重新挑。"""
    first = write_gguf(roots.ext / "alpha.gguf")
    result = asyncio.run(roots.scanner.register_paths([str(roots.ext)]))
    assert result["roots"] == [str(roots.ext)]
    assert str(roots.ext) in roots_of(roots.scanner)

    # 挑完之后才放进目录的模型：只能靠"记住目录"被发现
    write_gguf(roots.ext / "beta.gguf")
    asyncio.run(roots.scanner.scan_and_register())
    assert "beta.gguf" not in result["registered"]
    assert Path(rows()["beta.gguf"]).parent == roots.ext


def test_removed_root_is_no_longer_walked(roots):
    """反向对照：移除后同目录再放模型就不该登记 —— 证明上一条的生效来自记忆本身。"""
    asyncio.run(roots.scanner.register_paths([str(roots.ext)]))
    removed = asyncio.run(roots.scanner.remove_root(str(roots.ext)))
    assert removed == str(roots.ext)
    assert str(roots.ext) not in roots_of(roots.scanner)

    write_gguf(roots.ext / "gamma.gguf")
    asyncio.run(roots.scanner.scan_and_register())
    assert "gamma.gguf" not in rows(), (
        "已经移除的目录还在被扫描 —— 移除按钮只是画了个样子"
    )
    assert asyncio.run(roots.scanner.remove_root(str(roots.ext))) is None


def test_remove_root_matches_regardless_of_spelling(roots):
    """用户从别的入口粘进来的路径大小写和分隔符可能不同，不能因此移不掉。"""
    asyncio.run(roots.scanner.register_paths([str(roots.ext)]))
    variant = str(roots.ext).replace("\\", "/") + os.sep
    assert asyncio.run(roots.scanner.remove_root(variant)) == str(roots.ext)


def test_a_directory_that_is_already_a_config_root_is_not_added_twice(roots):
    """settings.model_dirs 里已经有的目录，_roots() 不能重复走第二遍。"""
    asyncio.run(db_mod.db.add_extra_root(str(roots.cfg)))
    asyncio.run(db_mod.db.add_extra_root(str(roots.cfg).replace("\\", "/")))
    assert asyncio.run(db_mod.db.add_extra_root(str(roots.cfg))) is False, (
        "重复添加必须报告 False，否则 GUI 会告诉用户「记住了一个新目录」"
    )
    assert roots_of(roots.scanner) == [str(roots.cfg)]


# ------------------------------------------------------------------ 同名歧义


def test_same_name_with_both_paths_alive_is_refused(roots):
    """两条路径都在盘上时不许悄悄改指 —— 按文件名索引，选错就是加载另一只模型。"""
    old = write_gguf(roots.cfg / "dup.gguf", "OldCopy")
    asyncio.run(roots.scanner.register_paths([str(old)]))
    new = write_gguf(roots.ext / "dup.gguf", "NewCopy")

    result = asyncio.run(roots.scanner.register_paths([str(new)]))

    assert len(result["rejected"]) == 1
    reason = result["rejected"][0]["reason"]
    assert str(old) in reason and str(new) in reason, (
        "拒绝原因必须把两条路径都摆出来，否则用户不知道要删哪条旧记录"
    )
    assert rows()["dup.gguf"] == str(old), "同名冲突时旧记录的路径被覆盖了"


def test_same_name_with_old_path_gone_repoints_to_the_new_file(roots):
    """反向对照：旧位置确实没了（挪盘）就必须改指，不能和上一条一样拒绝。"""
    old = write_gguf(roots.cfg / "dup.gguf", "OldCopy")
    asyncio.run(roots.scanner.register_paths([str(old)]))
    old.unlink()
    new = write_gguf(roots.ext / "dup.gguf", "NewCopy")

    result = asyncio.run(roots.scanner.register_paths([str(new)]))

    assert result["rejected"] == []
    assert result["moved"] == ["dup.gguf"]
    assert result["registered"] == [], "改指不新增行，报告成「新增」会让用户以为库里有两只"
    assert rows()["dup.gguf"] == str(new)


# ---------------------------------------------------------------- 逐条判定


def test_missing_path_and_unknown_extension_are_rejected_one_by_one(roots):
    """一个坏选择不能带崩整批：另一只好模型照样登记。"""
    good = write_gguf(roots.ext / "keep.gguf")
    txt = roots.ext / "notes.txt"
    txt.write_text("not a model", encoding="utf-8")

    result = asyncio.run(roots.scanner.register_paths(
        [str(roots.ext / "nope.gguf"), str(txt), str(good)]
    ))

    assert result["registered"] == ["keep.gguf"]
    reasons = {Path(r["path"]).name: r["reason"] for r in result["rejected"]}
    assert set(reasons) == {"nope.gguf", "notes.txt"}
    assert "盘没插上" in reasons["nope.gguf"]
    assert "不支持的模型文件类型" in reasons["notes.txt"]
    assert asyncio.run(db_mod.db.list_extra_roots()) == []


# ------------------------------------------------------- 接口/GUI 接线守卫


def test_extra_root_routes_are_declared_before_the_int_id_route():
    """/models/{model_id} 是 int 路径参数；Starlette 先按正则匹配再校验类型。

    若 extra-roots 挂在它下面，GET /api/models/extra-roots 会拿到 422
    （"extra-roots" 不是 int）而不是答案 —— GUI 只会显示空列表，看不出原因。
    """
    src = Path(sc.__file__).parent.parent / "api" / "routes" / "models.py"
    text = src.read_text(encoding="utf-8")
    id_route = text.index('@router.get("/models/{model_id}"')
    for marker in ('@router.post("/models/register-paths")',
                   '@router.get("/models/extra-roots")',
                   '@router.delete("/models/extra-roots")'):
        assert text.index(marker) < id_route, f"{marker} 必须声明在 /models/{{model_id}} 之前"


def test_gui_registers_in_a_thread_and_reenables_both_buttons_on_error():
    """登记一个整盘库要走几秒到几分钟，不能在 GUI 线程里做；失败后按钮必须回来。"""
    src = (Path(sc.__file__).parent.parent / "gui" / "pages" / "models.py").read_text(encoding="utf-8")
    body = src.split("def _register(self", 1)[1].split("\n    def ", 1)[0]
    assert "run_in_thread(do_register" in body
    assert "self.pick_btn.setEnabled(False)" in body
    assert "self.scan_btn.setEnabled(False)" in body, (
        "登记期间不锁「扫描文件」，两个会改同一批记录的请求可能同时跑"
    )
    error_body = body.split("def on_error(err):", 1)[1]
    assert "self.pick_btn.setEnabled(True)" in error_body
    assert "self.scan_btn.setEnabled(True)" in error_body


def test_pick_result_split_is_reported_per_bucket(monkeypatch):
    """新增/改指/已在库/记住目录/拒绝 五桶各报一次数，全空才算"没有变化"。"""
    from lmloader.gui import i18n
    # 断言的是中文文案里的分桶数字；界面语言由 config.yaml 决定，这里钉住 zh
    monkeypatch.setattr(i18n, "_LANGUAGE", "zh")
    page = importlib.import_module("lmloader.gui.pages.models")
    fmt = page.ModelsPage._format_pick_result

    empty = fmt(None, {"registered": [], "existing": [], "moved": [], "roots": [], "rejected": []})
    assert "没有变化" in empty

    line = fmt(None, {
        "registered": ["a.gguf", "b.gguf"],
        "existing": ["c.gguf"],
        "moved": ["d.gguf"],
        "roots": ["E:/library"],
        "rejected": [{"path": "x", "reason": "r"}],
    })
    for fragment in ("新增 2 只", "改指新路径 1 只", "已在库里 1 只", "记住目录 1 个", "拒绝 1 个"):
        assert fragment in line, f"{fragment!r} 丢了 —— 用户没法确认自己那一只到底生效没有"
