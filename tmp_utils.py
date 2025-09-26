# my_app/utils.py

def dumps(obj):
    """
    fast_jsonが利用可能ならそれを使ってJSON文字列に変換し、
    利用できなければ標準のjsonライブラリを使う。
    """
    try:
        import fast_json
        # importが成功した場合
        return f"fast_json: {fast_json.dumps(obj)}"
    except ImportError:
        # importが失敗した場合
        import json
        return f"standard_json: {json.dumps(obj)}"
