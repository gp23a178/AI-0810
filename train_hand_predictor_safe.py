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
META_COLUMNS = set(TRAIN_META_COLUMNS) - {"observer", "target_player"}


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
    # 対局IDだけを先に集め、最低1対局を評価側に確保する。
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        game_ids = {r["game_id"] for r in csv.DictReader(f)}
    if len(game_ids) < 2 or "" in game_ids:
        raise ValueError("学習・評価には異なる正常な対局IDが2件以上必要です")
    ordered = sorted(game_ids, key=lambda gid: hashlib.blake2b(f"{seed}:{gid}".encode(), digest_size=16).digest())
    test_games = set(ordered[:max(1, round(len(ordered)*0.2))])
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
            if game_id in test_games:
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

    source = Path(__file__).resolve().parent / "tenhou_data" / "converted" / args.mode / "training_samples.csv.gz"
    if not source.is_file():
        raise SystemExit(
            f"データファイルがありません: {source}\n"
            f"先に python3 convert_tenhou_safe.py --mode {args.mode} --max-games 20 "
            "--no-full-reference を実行してください。"
        )
    output_dir = Path(__file__).resolve().parent / "tenhou_data" / "models" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'学習データ読み込み: {source.resolve()}', flush=True)
    features, train, test, n_train, n_test = load_samples(source, args.max_train, args.max_test, args.seed)
    if not train or not test:
        raise SystemExit('訓練側または評価側の対局がありません。変換対局数を増やしてください。対局を跨がず行だけで分割する処理は行いません。')
    train_games = {r[0] for r in train}
    test_games = {r[0] for r in test}
    if train_games & test_games:
        raise RuntimeError('対局分割に重複があります')
    x_train = np.asarray([r[3] for r in train], dtype=np.float32)
    x_test = np.asarray([r[3] for r in test], dtype=np.float32)
    y_train = np.asarray([r[4] for r in train], dtype=np.float32)
    y_test = np.asarray([r[4] for r in test], dtype=np.float32)
    print(f'学習開始: 訓練{len(train)}行/{len(train_games)}対局、評価{len(test)}行/{len(test_games)}対局', flush=True)
    model = RandomForestRegressor(n_estimators=args.trees, random_state=args.seed, n_jobs=-1, max_depth=18, min_samples_leaf=5)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = dict(train_rows=len(train), test_rows=len(test), available_train_rows=n_train,
                   available_test_rows=n_test, train_game_ids=sorted(train_games), test_game_ids=sorted(test_games),
                   game_overlap=0, feature_count=len(features),
                   mae=float(mean_absolute_error(y_test, predicted)),
                   mean_baseline_mae=float(mean_absolute_error(y_test, np.tile(y_train.mean(axis=0), (len(test), 1)))),
                   target='concealed hand tile counts, 34 types; reds merged',
                   split='seeded hash ordering of unique game IDs; 20% holdout, minimum 1 game; no row-level split')
    joblib.dump(dict(model=model, feature_names=features, schema_version=2), output_dir/'hand_predictor.joblib')
    (output_dir/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output_dir/'feature_names.json').write_text(json.dumps(features, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output_dir/'validation_examples.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['game_id', 'observer', 'target_player', 'actual_counts', 'predicted_counts'])
        for record, yp in zip(test[:20], predicted[:20]):
            w.writerow([*record[:3], json.dumps(record[4]), json.dumps(yp.tolist())])
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f'保存先: {output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
