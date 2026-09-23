import pytest

from xtc.utils.tools import (
    get_iree_prefix,
    get_iree_sdk,
    has_iree_runtime,
)

# Discovery helpers (commit 1). Pure Python: no IREE package or shim needed, so
# these run everywhere and give the discovery layer real CI coverage.


def test_get_iree_prefix_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("IREE_RUNTIME_DIR", str(tmp_path))
    assert get_iree_prefix() == tmp_path


def test_get_iree_prefix_explicit_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("IREE_RUNTIME_DIR", str(tmp_path / "ignored"))
    assert get_iree_prefix(tmp_path) == tmp_path


def test_get_iree_prefix_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("IREE_RUNTIME_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError):
        get_iree_prefix()


def test_sdk_detection(monkeypatch, tmp_path):
    monkeypatch.setenv("IREE_RUNTIME_DIR", str(tmp_path))
    # Prefix exists but the SDK (headers + archives) is not installed yet.
    assert has_iree_runtime() is False
    with pytest.raises(RuntimeError):
        get_iree_sdk()
    # Install a fake SDK -> it is now detected.
    include = tmp_path / "include"
    include.mkdir()
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libiree_runtime_unified.a").write_bytes(b"")
    assert has_iree_runtime() is True
    assert get_iree_sdk() == (include, [lib / "libiree_runtime_unified.a"])
