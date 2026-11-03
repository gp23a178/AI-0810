# -*- coding: utf-8 -*-
"""公開情報から相手の伏せた手牌(34種の枚数)を推定するベースライン。

使用例: python3 train_hand_predictor_safe.py --mode S --max-train 12000 --max-test 3000
追加ライブラリ: numpy, scikit-learn, joblib

入力: tenhou_data/converted/<S|M|L>/training_samples.csv.gz
出力: tenhou_data/models/<S|M|L>/hand_predictor.joblib,
      metrics.json, feature_names.json, validation_examples.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

RED_TO_BASE = {35: 5, 36: 14, 37: 23}
from convert_tenhou_safe import TRAIN_COLUMNS, TRAIN_META_COLUMNS, TRAIN_LABEL_COLUMNS
META_COLUMNS = set(TRAIN_META_COLUMNS)


def belongs_to_test(game_id: str) -> bool:
    """同じ対局の全観測者と全時点を同じ訓練/評価側に置く。"""
    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100 < 20


def label_from_csv(row: dict) -> list[float]:
    y = [float(row[f"target_hand_{t}"] or 0) for t in range(1, 35)]
    for red, base in RED_TO_BASE.items():
        y[base - 1] += float(row[f"target_hand_{red}"] or 0)
    return y


def numeric_row(row: dict, feature_names: list[str]) -> list[float]:
    result = []
    for name in feature_names:
        v = row.get(name, "")
        try:
            result.append(float(v) if v not in (None, "") else 0.0)
        except (ValueError, TypeError):
            raise ValueError(f"数値に変換できない特徴量: {name}={v!r}")
    return result


def reservoir_add(pool, item, seen: int, cap: int, rng) -> None:
    """メモリに乗る行数を固定した一様サンプリング。"""
    if len(pool) < cap:
        pool.append(item)
        return
    index = rng.randrange(seen)
    if index < cap:
        pool[index] = item


def load_samples(path: Path, max_train: int, max_test: int, seed: int):
    rng = random.Random(seed)
    train = []
    test = []
    n_train = 0
    n_test = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        required = {"game_id", "observer", "target_player"} | {
            f"target_hand_{t}" for t in range(1, 38)
        }
        missing = required - set(columns)
        if missing:
            raise ValueError(f"必要な列がありません: {sorted(missing)}")
        # 許可済みの固定スキーマ以外を入力に通さない（ホワイトリスト方式）。
        if columns != TRAIN_COLUMNS:
            raise ValueError("学習CSVの列構造が想定と一致しません。convert_tenhou_safe.pyで再生成してください")
        feature_names = [c for c in TRAIN_COLUMNS if c not in META_COLUMNS
                         and c not in TRAIN_LABEL_COLUMNS]
        if any(c.startswith("target_hand_") or "ankan_true" in c or
               c.startswith(tuple(f"p{j}_hand_" for j in range(4))) for c in feature_names):
            raise RuntimeError("非公開情報が入力特徴量に混入しました。学習を中断します")
        for i, row in enumerate(reader, 1):
            game_id = row["game_id"]
            x = numeric_row(row, feature_names)
            y = label_from_csv(row)
            record = (game_id, row["observer"], row["target_player"], x, y)
            if belongs_to_test(game_id):
                n_test += 1
                reservoir_add(test, record, n_test, max_test, rng)
            else:
                n_train += 1
                reservoir_add(train, record, n_train, max_train, rng)
            if i % 100000 == 0:
                print(f"読み込み済み: {i:,} 行 / 保持: 訓練{len(train):,}, 評価{len(test):,}")
    return feature_names, train, test, n_train, n_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["S", "M", "L"], default="S")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_train, args.max_test, args.trees) < 1:
        parser.error("max-train, max-test, trees は1以上にしてください")

    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
    except ImportError as exc:
        raise SystemExit(
            "追加パッケージがありません。WSL上で仮想環境を有効化して "
            "python -m pip install numpy scikit-learn joblib を実行してください。"
        ) from exc

    source = Path("tenhou_data") / "converted" / args.mode / "training_samples.csv.gz"
    if not source.is_file():
        raise SystemExit(
            f"データファイルがありません: {source}\n"
            f"先に python3 convert_tenhou_safe.py --mode {args.mode} --max-games 20 "
            "--no-full-reference を実行してください。"
        )
    output_dir = Path("tenhou_data") / "models" / args.mode