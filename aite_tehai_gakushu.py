# -*- coding: utf-8 -*-
"""TensorFlowで公開情報から相手の伏せた手牌34種の枚数を推定。
設定: jikken_settei.py。使用例: python aite_tehai_gakushu.py --mode S
入力: tenhou_data/converted/all/（対局別CSVと一覧）。学習時に人数・達成日で抽出。
出力: tenhou_data/models/S/tehai_yosoku.keras と評価・入力列・学習履歴。
対局単位で分割し、評価データを学習や正規化に使用しません。
"""
from __future__ import annotations

import argparse
import os
import jikken_settei as settei
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from datetime import date

RED_TO_BASE = {35: 5, 36: 14, 37: 23}
from paifu_csv_henkan import TRAIN_COLUMNS, TRAIN_META_COLUMNS, TRAIN_LABEL_COLUMNS, REFERENCE_COLUMNS, REFERENCE_EXTRA_COLUMNS, reference_to_training
META_COLUMNS = set(TRAIN_META_COLUMNS) - {"observer", "target_player"}


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


def select_csvs(directory, mode, seed, players_override=None, max_games_override=None):
    """保存済み一覧から抽出。重複ZIPに属する対局も全所属プレイヤーで判定。"""
    directory = Path(directory).resolve()
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if summary.get('schema_version') != 4 or summary.get('layout') != 'reference451_csv' or summary.get('check_log'):
        raise ValueError('全件CSV形式ではありません。新しい paifu_csv_henkan.py で全件変換してください')
    if summary.get('conversion_errors') or summary['converted_games'] != summary['candidate_games_after_date_filter']:
        raise ValueError('変換失敗または未完了のデータです。conversion_errors.csv を確認してください')
    def table(name):
        with (directory/name).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    config = settei.size_settings(mode)
    players = sorted(table('selected_players.csv'), key=lambda p:p['player'])
    n = config['players'] if players_override is None else (players_override or None)
    if n is not None and n > len(players):
        raise ValueError(f'指定{n}人に対し保存済み人数は{len(players)}人です')
    selected_players = random.Random(seed).sample(players,n) if n is not None else players
    dates = {p['player']:date.fromisoformat(p['achievement_date']) for p in selected_players}
    eligible = set()
    for r in table('logs_manifest.csv'):
        # duplicate_game_idも所属情報として必要。元ZIPを跨いで同じIDを一度だけ採用。
        if r['source_player'] in dates and r['game_id'] and r['game_date']:
            if date.fromisoformat(r['game_date']) > dates[r['source_player']]:
                eligible.add(r['game_id'])
    games = [r for r in table('selected_games.csv') if r['game_id'] in eligible]
    games.sort(key=lambda r:r['game_id'])
    cap = config['max_games'] if max_games_override is None else max_games_override
    if cap is not None and len(games)>cap:
        games = random.Random(seed).sample(games,cap)
    paths=[]
    for r in games:
        p=(directory/r['csv_file']).resolve()
        if not p.is_relative_to(directory) or not p.is_file():
            raise ValueError(f'CSVの場所が不正または存在しません: {p}')
        meta=(directory/r['meta_file']).resolve()
        if not meta.is_relative_to(directory) or not meta.is_file():raise ValueError(f'補助CSVがありません: {meta}')
        paths.append((p,meta,{k:r.get(k,'') for k in ['game_id','game_date','source_player','source_zip']}))
    if len({r['game_id'] for r in games}) < 2:
        raise ValueError(f'選んだ{len(selected_players)}人の達成翌日以降は{len(games)}対局です。学習には2対局以上必要です')
    return paths, selected_players, games


def iter_csv_rows(paths):
    from itertools import zip_longest
    for path,meta_path,info in paths:
        with path.open(encoding='utf-8-sig',newline='') as f, meta_path.open(encoding='utf-8-sig',newline='') as mf:
            reader=csv.DictReader(f);extra=csv.DictReader(mf)
            if reader.fieldnames!=REFERENCE_COLUMNS or extra.fieldnames!=REFERENCE_EXTRA_COLUMNS:
                raise ValueError(f'CSVの列構造が一致しません: {path}')
            for index,(ref,aux) in enumerate(zip_longest(reader,extra)):
                if ref is None or aux is None or int(aux['state_index'])!=index:
                    raise ValueError(f'参照CSVと補助CSVの行が一致しません: {path}')
                yield from reference_to_training(ref,aux,info)


def load_samples(path, max_train: int, max_test: int, seed: int):
    game_ids = {info['game_id'] for _,_,info in path}
    if len(game_ids) < 2 or '' in game_ids:
        raise ValueError('学習・評価には異なる正常な対局IDが2件以上必要です')
    ordered = sorted(game_ids, key=lambda gid: hashlib.blake2b(f"{seed}:{gid}".encode(), digest_size=16).digest())
    test_games = set(ordered[:max(1, round(len(ordered)*0.2))])
    rng = random.Random(seed)
    train = []
    test = []
    n_train = 0
    n_test = 0
    feature_names = [c for c in TRAIN_COLUMNS if c not in META_COLUMNS and c not in TRAIN_LABEL_COLUMNS]
    if any(c.startswith('target_hand_') or 'ankan_true' in c or
           c.startswith(tuple(f'p{j}_hand_' for j in range(4))) for c in feature_names):
        raise RuntimeError('非公開情報が入力に混入しました')
    for i,row in enumerate(iter_csv_rows(path),1):
        game_id=row['game_id']
        record=(game_id,row['observer'],row['target_player'],numeric_row(row,feature_names),label_from_csv(row))
        if game_id in test_games:
            n_test+=1
            reservoir_add(test,record,n_test,max_test,rng)
        else:
            n_train+=1
            reservoir_add(train,record,n_train,max_train,rng)
        if i%100000==0:
            print(f'読み込み済み: {i:,}行',flush=True)
    return feature_names, train, test, n_train, n_test


def build_model(tf, x_train, learning_rate):
    # 正規化の統計量は訓練側だけから計算し、モデルに保存する。
    import numpy as np
    mean = x_train.mean(axis=0)
    variance = np.maximum(x_train.var(axis=0), 1e-6)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Normalization(mean=mean, variance=variance),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(34, activation="sigmoid"),
        tf.keras.layers.Rescaling(4.0),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str.upper, choices=["S", "M", "L"], default=str(settei.MODE).upper())
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--epochs", type=int, default=settei.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settei.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=settei.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=settei.SEED)
    parser.add_argument("--input-dir", type=Path, help="全件CSVの保存フォルダ")
    parser.add_argument("--players", type=int, help="今回の人数を指定。0=全員")
    parser.add_argument("--max-games", type=int, help="今回の学習に選ぶ対局数の上限")
    parser.add_argument("--output-dir", type=Path, help="モデルの保存先を指定")
    args = parser.parse_args()
    config = settei.size_settings(args.mode)
    args.max_train = config['max_train'] if args.max_train is None else args.max_train
    args.max_test = config['max_test'] if args.max_test is None else args.max_test
    if min(args.max_train, args.max_test, args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("行数・epochs・batch-sizeは1以上、learning-rateは正の数です")
    if args.players is not None and args.players < 0:
        parser.error('--playersは0以上です')
    if args.max_games is not None and args.max_games < 2:
        parser.error('--max-gamesは2以上です')
    source = args.input_dir or settei.DATA_DIR/"converted"/"all"
    output = args.output_dir or settei.DATA_DIR/"models"/args.mode
    if not (source/"summary.json").is_file():
        parser.error(f"全件CSVがありません: {source}。先に python paifu_csv_henkan.py で変換してください")
    if (output/'tehai_yosoku.keras').exists():
        parser.error(f"既存モデルがあります。別の --output-dir を指定してください: {output}")
    paths, selected_players, selected_games = select_csvs(source,args.mode,args.seed,args.players,args.max_games)
    print(f'学習対象: {len(selected_players)}人 / 達成翌日以降{len(selected_games)}対局',flush=True)
    os.environ.setdefault("KERAS_HOME", str(settei.BASE_DIR/'.keras'))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow環境を有効にしてください: source ~/WSL/venv/bin/activate") from exc
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__} / 入力: {source.resolve()}", flush=True)
    features, train, test, n_train, n_test = load_samples(paths, args.max_train, args.max_test, args.seed)
    if not train or not test:
        raise SystemExit("訓練または評価の行がありません。変換する対局を増やしてください")
    train_games, test_games = {r[0] for r in train}, {r[0] for r in test}
    if train_games & test_games:
        raise RuntimeError("同一対局が学習・評価の両方に入りました")
    x_train = np.asarray([r[3] for r in train], dtype=np.float32)
    x_test = np.asarray([r[3] for r in test], dtype=np.float32)
    y_train = np.asarray([r[4] for r in train], dtype=np.float32)
    y_test = np.asarray([r[4] for r in test], dtype=np.float32)
    if not all(np.isfinite(a).all() for a in [x_train,x_test,y_train,y_test]):
        raise ValueError("CSVに非有限の数値が含まれています")
    model = build_model(tf, x_train, args.learning_rate)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(train),seed=args.seed).batch(args.batch_size).with_options(options)
    print(f"学習開始: 訓練{len(train)}行/{len(train_games)}対局、評価{len(test)}行/{len(test_games)}対局",flush=True)
    # 評価側はfitに渡さない。epoch選択や学習中の調整にも使用しない。
    history = model.fit(train_data, epochs=args.epochs, verbose=2, shuffle=False)
    predicted = np.concatenate([model(x_test[i:i+args.batch_size],training=False).numpy()
                                for i in range(0,len(x_test),args.batch_size)])
    mae = float(np.abs(y_test-predicted).mean())
    metrics = dict(framework="TensorFlow", tensorflow_version=tf.__version__,
                   train_rows=len(train), test_rows=len(test), available_train_rows=n_train,
                   available_test_rows=n_test, train_game_ids=sorted(train_games),test_game_ids=sorted(test_games),
                   game_overlap=0, feature_count=len(features),mae=mae,
                   mean_baseline_mae=float(np.abs(y_test-y_train.mean(axis=0)).mean()),
                   epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,learning_rate=args.learning_rate,
                   input=str(source.resolve()), selected_players=[p["player"] for p in selected_players], rank_policy="after", normalization="training data only; stored in model",
                   target="concealed hand tile counts, 34 types; reds merged; not a legal hand generator",
                   split="seeded hash ordering of game IDs, 20% holdout (minimum 1); no row-level split")
    output.mkdir(parents=True, exist_ok=True)
    for filename,records in [('gakushu_players.csv',selected_players),('gakushu_games.csv',selected_games)]:
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    model.save(output/'tehai_yosoku.keras')
    # 保存したモデルを読み直して予測値の一致も確認する。
    restored = tf.keras.models.load_model(output/'tehai_yosoku.keras')
    np.testing.assert_allclose(restored(x_test[:5],training=False).numpy(),predicted[:5],rtol=1e-5,atol=1e-6)
    metrics['saved_model_reload_verified'] = True
    for name,value in [('metrics.json',metrics),('feature_names.json',features),('gakushu_rireki.json',history.history)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'validation_examples.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['game_id','observer','target_player','actual_counts','predicted_counts'])
        for record,yp in zip(test[:20],predicted[:20]):
            w.writerow([*record[:3],json.dumps(record[4]),json.dumps(yp.tolist())])
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    print(f"モデル保存先: {output.resolve()}",flush=True)


if __name__ == '__main__':
    main()
