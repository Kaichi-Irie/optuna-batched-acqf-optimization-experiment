# tests/test_utils.py

import sys
from types import ModuleType
import pytest

# テスト対象の関数をインポート
# 注意：この時点では my_app.utils 内の import はまだ実行されない
from tmp_utils import dumps

# --- テストケースのパラメータを定義 ---
# (テストID, importを成功させるか, 期待される出力のプレフィックス)
test_cases = [
    pytest.param(
        "import_succeeds",  # テストID
        True,               # importを成功させる
        "fast_json",        # 期待される出力
        id="when_fast_json_is_available"
    ),
    pytest.param(
        "import_fails",     # テストID
        False,              # importを失敗させる
        "standard_json",    # 期待される出力
        id="when_fast_json_is_not_available"
    ),
]

@pytest.mark.parametrize("test_id, should_succeed, expected_prefix", test_cases)
def test_dumps_behavior_on_import_condition(monkeypatch, test_id, should_succeed, expected_prefix):
    """
    fast_jsonのimport成否によってdumps関数の挙動が変わることをテストする
    """
    target_module = 'fast_json'

    # --- monkeypatchによるimportのシミュレーション ---
    if should_succeed:
        # 成功ケース：偽のモジュールを作成し、sys.modulesに登録
        mock_fast_json = ModuleType(target_module)
        # 偽のdumps関数を定義
        mock_fast_json.dumps = lambda obj: f"mocked_{obj}"
        monkeypatch.setitem(sys.modules, target_module, mock_fast_json)
    else:
        # 失敗ケース：sys.modulesにNoneを設定してimportを失敗させる
        # これにより、次に `import fast_json` が呼ばれたときに ImportError が送出される
        monkeypatch.setitem(sys.modules, target_module, None)

    # --- 実行と検証 ---
    data_to_dump = "data"
    result = dumps(data_to_dump)

    assert result.startswith(expected_prefix)

    # --- クリーンアップ ---
    # monkeypatchがテスト終了後にsys.modulesを自動で元に戻してくれるので、
    # 手動での後処理は不要です。
    # ただし、モジュールがグローバルにキャッシュされることを避けるため、
    # 影響を及ぼす可能性のあるモジュールをアンロードすることもあります。
    if target_module in sys.modules:
        monkeypatch.delitem(sys.modules, target_module)
