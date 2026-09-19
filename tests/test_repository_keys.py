"""PEP 503 等价包名键的合并回归测试。

当同一份快照中出现多个规范化后同名的键（``Foo_Bar``、``foo.bar``、
``FOO-BAR`` 等）时：

* 各键下的候选版本必须**合并**为一个候选集（旧实现以后写覆盖先写，
  会静默丢失候选），并按 PEP 440 降序参与全部端点；
* 跨键的同一规范化版本，依赖（marker 环境过滤后）与元数据完全一致时
  去重；不一致时稳定返回 SnapshotError（API 层为 422
  invalid_snapshot）；
* 合并结果、去重与报错均不得随 JSON 键顺序变化。
"""

from __future__ import annotations

from itertools import permutations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repository import (
    SnapshotError,
    build_repository,
    canonical,
    parse_root_specifier,
)
from app.resolver import Resolver

client = TestClient(app)

DEFAULT_ENV = {
    "python_version": "3.11",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
}


def _candidate(version, deps=(), metadata=None):
    item = {"version": version, "dependencies": [{"name": n} for n in deps]}
    if metadata is not None:
        item["metadata"] = metadata
    return item


def _versions(repo, name):
    return [str(c.version) for c in repo.candidates(name)]


def _resolve(raw_repo, root="foo-bar", env=None):
    repo = build_repository(raw_repo, env or DEFAULT_ENV)
    resolver = Resolver(repo)
    state = resolver.resolve(canonical(root), parse_root_specifier(""), "")
    assert state is not None
    return {pkg: str(ver) for pkg, ver in sorted(state.selection.items())}


def _reordered(mapping, order):
    """按给定键顺序重建字典，模拟不同的 JSON 键顺序。"""
    return {key: mapping[key] for key in order}


# --------------------------------------------------------------------- #
# 1. 跨 PEP 503 等价键合并候选，按 PEP 440 降序
# --------------------------------------------------------------------- #
def test_equivalent_keys_candidates_merged_and_sorted():
    raw = {
        "Foo_Bar": [_candidate("1.0")],
        "foo.bar": [_candidate("2.0"), _candidate("1.5")],
        "FOO-BAR": [_candidate("3.0"), _candidate("1.5")],
    }
    repo = build_repository(raw, DEFAULT_ENV)

    # 三种拼写都命中同一个规范化包。
    assert repo.known("Foo_Bar")
    assert repo.known("foo.bar")
    assert repo.known("FOO-BAR")
    # 跨键候选合并、同版本去重，并按 PEP 440 降序。
    assert _versions(repo, "foo-bar") == ["3.0", "2.0", "1.5", "1.0"]


def test_equivalent_keys_merge_participates_in_resolution():
    # 满足约束的高版本只出现在先写、会被旧实现覆盖掉的等价键下；
    # 覆盖实现会让解析回退到 1.x 甚至无候选，合并后才能选到 3.0。
    raw = {
        "app": [{"version": "1.0",
                 "dependencies": [{"name": "foo_bar", "specifier": ">=2.0"}]}],
        "Foo_Bar": [_candidate("3.0"), _candidate("2.0")],
        "foo.bar": [_candidate("1.0")],
    }
    assert _resolve(raw, root="app") == {
        "app": "1.0", "foo-bar": "3.0",
    }


# --------------------------------------------------------------------- #
# 2. 顺序置换：任意 JSON 键顺序下结果一致
# --------------------------------------------------------------------- #
def test_key_order_permutations_produce_identical_repository():
    raw = {
        "Foo_Bar": [_candidate("1.0")],
        "foo.bar": [_candidate("2.0"), _candidate("1.5")],
        "FOO-BAR": [_candidate("3.0"), _candidate("1.5")],
    }
    baseline = None
    for order in permutations(raw):
        repo = build_repository(_reordered(raw, order), DEFAULT_ENV)
        versions = _versions(repo, "foo-bar")
        assert versions == ["3.0", "2.0", "1.5", "1.0"]
        if baseline is None:
            baseline = versions
        else:
            assert versions == baseline


def test_key_order_permutations_produce_identical_resolution():
    raw = {
        "App": [{"version": "1.0",
                 "dependencies": [{"name": "foo_bar", "specifier": ""}]}],
        "Foo_Bar": [_candidate("1.0"), _candidate("2.0", deps=["shared_util"])],
        "foo.bar": [_candidate("2.0", deps=["shared_util"]),
                    _candidate("3.0", deps=["Shared.Util"])],
        "shared.util": [_candidate("1.0")],
        "SHARED_UTIL": [_candidate("0.9")],
    }
    expected = {"app": "1.0", "foo-bar": "3.0", "shared-util": "1.0"}
    for order in permutations(raw):
        assert _resolve(_reordered(raw, order), root="app") == expected


def test_key_order_permutations_conflict_error_is_stable():
    # 两个规范化包各自存在跨键同版本冲突；报错必须稳定指向同一个包，
    # 不随键顺序翻转。
    def build(order):
        raw = {
            "zeta_pkg": [_candidate("1.0", deps=["a"])],
            "ZETA.PKG": [_candidate("1.0", deps=["b"])],
            "alpha_pkg": [_candidate("1.0", deps=["a"])],
            "ALPHA.PKG": [_candidate("1.0", deps=["b"])],
        }
        with pytest.raises(SnapshotError) as exc:
            build_repository(_reordered(raw, order), DEFAULT_ENV)
        return str(exc.value)

    messages = {build(order) for order in permutations(
        ["zeta_pkg", "ZETA.PKG", "alpha_pkg", "ALPHA.PKG"])}
    assert len(messages) == 1
    assert "alpha-pkg" in messages.pop()


# --------------------------------------------------------------------- #
# 3. 跨键同版本：一致去重 / 依赖冲突 / 元数据冲突
# --------------------------------------------------------------------- #
def test_cross_key_same_version_identical_is_deduplicated():
    raw = {
        "Foo_Bar": [_candidate("1.0", deps=["dep_x"],
                               metadata={"license": "MIT"}),
                    _candidate("2.0")],
        "foo.bar": [_candidate("1.0", deps=["Dep.X"],
                               metadata={"license": "MIT"})],
    }
    repo = build_repository(raw, DEFAULT_ENV)
    # 1.0 跨键去重为一条，2.0 经合并保留（旧覆盖实现会丢失它）。
    assert _versions(repo, "foo-bar") == ["2.0", "1.0"]
    cands = {str(c.version): c for c in repo.candidates("foo-bar")}
    assert len(cands["1.0"].deps) == 1
    assert cands["1.0"].metadata.license == "MIT"


def test_cross_key_same_version_conflicting_deps_raises_regardless_of_order():
    messages = set()
    for first, second in (("Foo_Bar", "foo.bar"), ("foo.bar", "Foo_Bar")):
        raw = {
            first: [_candidate("1.0", deps=["x"])],
            second: [_candidate("1.0", deps=["y"])],
        }
        with pytest.raises(SnapshotError) as exc:
            build_repository(raw, DEFAULT_ENV)
        messages.add(str(exc.value))
    assert len(messages) == 1
    message = messages.pop()
    assert "foo-bar" in message and "1.0" in message


def test_cross_key_same_version_conflicting_metadata_raises():
    for order in (("Foo_Bar", "foo.bar"), ("foo.bar", "Foo_Bar")):
        raw = {
            order[0]: [_candidate("1.0", metadata={"license": "MIT"})],
            order[1]: [_candidate("1.0", metadata={"license": "GPL-2.0"})],
        }
        with pytest.raises(SnapshotError):
            build_repository(raw, DEFAULT_ENV)


def test_cross_key_conflict_http_422_stable_body():
    def post(order):
        repository = {
            order[0]: [{"version": "1.0",
                        "dependencies": [{"name": "x"}]}],
            order[1]: [{"version": "1.0",
                        "dependencies": [{"name": "y"}]}],
        }
        body = {
            "repository": repository,
            "root": order[0],
            "specifier": "",
            "allow_prereleases": False,
            "target_environment": DEFAULT_ENV,
        }
        r = client.post("/resolve", json=body)
        assert r.status_code == 422
        return r.json()

    bodies = [
        post(order)
        for order in (("Foo_Bar", "foo.bar"), ("foo.bar", "Foo_Bar"))
    ]
    assert bodies[0] == bodies[1]
    body = bodies[0]
    assert body["status"] == "invalid_snapshot"
    assert "foo-bar" in body["message"]


# --------------------------------------------------------------------- #
# 4. marker 环境过滤后再比较：过滤后一致去重，过滤后仍冲突报 422
# --------------------------------------------------------------------- #
def test_cross_key_same_version_markers_filtered_before_conflict_check():
    # 两条声明的依赖 marker 不同：linux 下一条激活、一条不激活 ->
    # 过滤后依赖集合不一致 -> 422；换到两边 marker 都不激活的环境，
    # 过滤后均为空 -> 合法去重。
    def raw_with_markers(marker_a, marker_b):
        return {
            "Foo_Bar": [{"version": "1.0", "dependencies": [
                {"name": "platform-extra", "marker": marker_a}]}],
            "foo.bar": [{"version": "1.0", "dependencies": [
                {"name": "platform-extra", "marker": marker_b}]}],
        }

    linux_env = DEFAULT_ENV
    darwin_env = {**DEFAULT_ENV, "sys_platform": "darwin"}

    conflicting = raw_with_markers(
        "sys_platform == 'linux'", "sys_platform == 'win32'"
    )
    with pytest.raises(SnapshotError):
        build_repository(conflicting, linux_env)

    # 同一快照在 darwin 环境：两条 marker 均为假，过滤后依赖都为空 -> 去重。
    repo = build_repository(conflicting, darwin_env)
    cands = repo.candidates("foo-bar")
    assert len(cands) == 1 and cands[0].deps == []

    # marker 相同且同时激活时，过滤后依赖一致 -> 跨键去重合法。
    identical = raw_with_markers(
        "sys_platform == 'linux'", "sys_platform == 'linux'"
    )
    repo = build_repository(identical, linux_env)
    cands = repo.candidates("foo-bar")
    assert len(cands) == 1
    assert [d.name for d in cands[0].deps] == ["platform-extra"]


def test_cross_key_merge_with_markers_end_to_end():
    # 根包的候选拆在等价键下：带 linux extra 的高版本放在先写、会被旧实现
    # 覆盖掉的键里；合并后才能选到 2.0 并正确做 marker 过滤。
    raw = {
        "app": [{"version": "1.0", "dependencies": [
            {"name": "Foo.Bar", "specifier": ""}]}],
        "Foo_Bar": [
            {"version": "2.0", "dependencies": [
                {"name": "linux-extra", "marker": "sys_platform == 'linux'"}]},
        ],
        "foo.bar": [{"version": "1.0", "dependencies": []}],
        "Linux_Extra": [{"version": "1.0", "dependencies": []}],
    }
    locked = _resolve(raw, root="app")
    assert locked == {"app": "1.0", "foo-bar": "2.0", "linux-extra": "1.0"}

    # win32 环境下 marker 不激活，linux-extra 不进入锁定清单。
    win_env = {**DEFAULT_ENV, "sys_platform": "win32"}
    locked_win = _resolve(raw, root="app", env=win_env)
    assert locked_win == {"app": "1.0", "foo-bar": "2.0"}


def test_cross_key_merge_visible_to_sbom_endpoint():
    # 携带完整元数据的高版本候选放在先写、会被旧实现覆盖掉的键里；
    # 合并后审计/SBOM 等端点必须仍能读到 2.0 及其元数据。
    sha = "a" * 64
    body = {
        "repository": {
            "app": [{"version": "1.0",
                     "dependencies": [{"name": "foo_bar"}]}],
            "Foo_Bar": [{"version": "2.0", "dependencies": [],
                         "metadata": {"license": "MIT", "sha256": sha,
                                      "source_url": "https://example.invalid/x"}}],
            "foo.bar": [{"version": "1.0", "dependencies": []}],
        },
        "root": "app",
        "specifier": "",
        "allow_prereleases": False,
        "target_environment": DEFAULT_ENV,
        "current_lock": {"app": "1.0", "foo-bar": "2.0"},
    }
    r = client.post("/sbom", json=body)
    assert r.status_code == 200, r.text
    packages = {p["name"]: p for p in r.json()["packages"]}
    assert set(packages) == {"app", "foo-bar"}
    assert packages["foo-bar"]["version"] == "2.0"
    assert packages["foo-bar"]["metadata"]["license"] == "MIT"
    assert packages["foo-bar"]["metadata"]["sha256"] == sha
