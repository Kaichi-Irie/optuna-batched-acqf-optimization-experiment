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

# terms
- `multiprocessing`: GPSamplerのattributeにワーカープールを持たせてMulti-Processingを行う
- `stacking`: Stackingする
    - not yet implemented
- `batched_acqf_eval`: 獲得関数の評価をバッチ化する。L-BFGS-Bの各反復はバッチ化されない。（逐次的にfor文を回して行う。）
    - not yet implemented
