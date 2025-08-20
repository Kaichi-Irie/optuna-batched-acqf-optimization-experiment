# optuna-batched-acqf-optimization-experiment
Optunaのサブルーチンである獲得関数をマルチスタートで最適化する部分を並列化（or バッチ化）して、高速化するための実験レポジトリ

# dependencies
```
uv add scipy==1.14
uv add optuna torch numpy pytest
uv add --dev ipykernel # optional
```

# test
```
uv run pytest
```

# プロジェクト構成

- `BatchedSampler`: Optunaの`GPSampler`を継承したクラス。`mode`でバッチ化の方法を切り替えながらバッチ化を実行するSampler
    - `batched_sampler.py`で定義されている。

- `multiprocessing`: GPSamplerのattributeにワーカープールを持たせてMulti-Processingを行う
    - `run_multiprocessing.py`: 実行例
    - `multiprocessing_optim_mixed.py`: Optunaの`optim_mixed.py`をマルチプロセス化したもの。
    - `test_multiprocessing_vs_gpsampler.py`: テスト
- `stacking`: Stackingする
    - `run_stacking.py`: 実行例
    - `stacking_optim_mixed.py`: Optunaの`optim_mixed.py`をスタッキング化したもの。
- `batched_acqf_eval`: 獲得関数の評価をバッチ化する。L-BFGS-Bの各反復はバッチ化されない。（逐次的にfor文を回して行う。）
    - `run_batched_acqf_eval.py`: 実行例
    - `batched_lbfgsb.py`: SciPyのL-BFGS-Bをバッチ化したもの。
        - ソース： https://github.com/nabenabe0928/batched-l-bfgs-b-python/blob/main/batched_lbfgsb.py
    - `test_batched_acqf_eval_vs_original.py`: テスト
- `original`: 元の実装。使い勝手を考えて、`mode`で切り替えられるようにしている。
    - `run_original.py`: 実行例
    - `test_myoriginal_vs_original.py`: テスト
