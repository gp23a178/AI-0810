# -*- coding: utf-8 -*-
"""TensorFlowで公開情報から相手の伏せた手牌34種の枚数を推定。
設定: このファイル冒頭。使用例: python aite_tehai_gakushu.py --mode S
入力: 冒頭のINPUT_DIR（既定はtenhou_data/converted/after/）（対局別CSVと一覧）。学習時に人数・達成日で抽出。
出力: 名前付きモデルと学習設定。評価は moderu_jikko.py で別途実行。
対局単位で分割し、評価データを学習や正規化に使用しません。
"""
from __future__ import annotations

import argparse
import os
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from datetime import date

# ==================== 設定（ここを変更して保存） ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "tenhou_data"
MODE = "M"                 # "S", "M", "L"（小文字での指定も可能）
SEED = 42                  # 同じ人数・seedなら同じ人を選択

# players: 選ぶ人数。None はmanifest内の全員。
# max_games: 学習に選ぶ対局数の上限。None は条件を満たす全対局。
# ここで学習条件を変更します。CSV変換の日付条件は --rank-policy で指定。
# max_train / max_test: 学習/評価に保持する行数の上限（人数ではありません）。
SIZE_SETTINGS = {
    "S": {"players": 5,    "max_games": None, "max_train": 12000, "max_test": 3000},
    "M": {"players": 12,   "max_games": None, "max_train": 24000, "max_test": 6000},
    "L": {"players": None, "max_games": None, "max_train": 48000, "max_test": 12000},
}

EPOCHS = 20               # 学習を繰り返す回数
BATCH_SIZE = 128          # 一度に学習する行数
LEARNING_RATE = 0.001


INPUT_DIR = DATA_DIR / "converted" / "after"
MODEL_NAME = "shiken_01"  # 新しく作成するモデル名


def size_settings(mode):
    values = SIZE_SETTINGS[mode.upper()].copy()
    for name in ("players", "max_games"):
        v = values[name]
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise ValueError(f"{mode}.{name} は1以上の整数またはNoneです")
    for name in ("max_train", "max_test"):
        v = values[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{mode}.{name} は1以上の整数です")
    return values


RED_TO_BASE = {35: 5, 36: 14, 37: 23}
from paifu_csv_henkan import TRAIN_COLUMNS, TRAIN_META_COLUMNS, TRAIN_LABEL_COLUMNS, REFERENCE_COLUMNS, REFERENCE_EXTRA_COLUMNS, reference_to_training
META_COLUMNS = set(TRAIN_META_COLUMNS) - {"observer", "target_player"}

LEGACY_FEATURES = [name for name in TRAIN_COLUMNS
                   if name not in META_COLUMNS and name not in TRAIN_LABEL_COLUMNS]
# 副露の個別データと局・親の情報から復元できる集計列を省く。
REDUNDANT_FEATURES = {'game_wind', 'hand_number'}
for seat in range(4):
    REDUNDANT_FEATURES.update(f'p{seat}_meld_{tile}' for tile in range(1, 38))
    REDUNDANT_FEATURES.update(
        f'p{seat}_{kind}_count' for kind in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'nuki')
    )
    REDUNDANT_FEATURES.add(f'p{seat}_seat_wind')
PREDICTION_FEATURES = [name for name in LEGACY_FEATURES if name not in REDUNDANT_FEATURES]


def validate_feature_names(names):
    if names not in (LEGACY_FEATURES, PREDICTION_FEATURES):
        raise ValueError('モデルの入力列が安全な許可リストと一致しません')
    return names



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


def select_csvs(directory, mode, seed, players_override=None, max_games_override=None, rank_policy="after"):
    """保存済み一覧から抽出。重複ZIPに属する対局も全所属プレイヤーで判定。"""
    directory = Path(directory).resolve()
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if summary.get('schema_version') != 4 or summary.get('layout') != 'reference451_csv' or summary.get('check_log'):
        raise ValueError('全件CSV形式ではありません。新しい paifu_csv_henkan.py で全件変換してください')
    if summary.get('conversion_errors') or summary['converted_games'] != summary['candidate_games_after_date_filter']:
        raise ValueError('変換失敗または未完了のデータです。conversion_errors.csv を確認してください')
    if rank_policy == 'all' and summary.get('rank_policy') != 'all':
        raise ValueError('全日付の学習には converted/all の全件変換データを指定してください')
    def table(name):
        with (directory/name).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    config = size_settings(mode)
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
            if rank_policy == 'all' or date.fromisoformat(r['game_date']) > dates[r['source_player']]:
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
        paths.append((p,meta,{k:r.get(k,'') for k in ['game_id','game_date','source_player','source_zip','training_rows']}))
    if len({r['game_id'] for r in games}) < 2:
        raise ValueError(f'選んだ{len(selected_players)}人の条件に合う対局は{len(games)}対局です。学習には2対局以上必要です')
    return paths, selected_players, games


def iter_csv_rows(paths, samples=None, *, action=None):
    from itertools import zip_longest
    for path,meta_path,info in paths:
        positions = samples[info['game_id']] if samples is not None else None
        if positions is not None and not positions:
            continue
        with path.open(encoding='utf-8-sig',newline='') as f, meta_path.open(encoding='utf-8-sig',newline='') as mf:
            reader=csv.DictReader(f);extra=csv.DictReader(mf)
            if reader.fieldnames!=REFERENCE_COLUMNS or extra.fieldnames!=REFERENCE_EXTRA_COLUMNS:
                raise ValueError(f'CSVの列構造が一致しません: {path}')
            max_index = max(positions) if positions else None
            for index,(ref,aux) in enumerate(zip_longest(reader,extra)):
                if ref is None or aux is None or int(aux['state_index'])!=index:
                    raise ValueError(f'参照CSVと補助CSVの行が一致しません: {path}')
                if action is not None and int(ref['action']) != action:
                    continue
                if positions is None:
                    yield from reference_to_training(ref,aux,info)
                elif index in positions:
                    for i, row in enumerate(reference_to_training(ref,aux,info)):
                        if i in positions[index]:
                            yield row
                    if index == max_index:
                        break


def split_game_ids(paths, seed):
    game_ids={info['game_id'] for _,_,info in paths}
    if len(game_ids)<2 or '' in game_ids:
        raise ValueError('学習・評価には異なる正常な対局IDが2件以上必要です')
    ordered=sorted(game_ids,key=lambda gid:hashlib.blake2b(f'{seed}:{gid}'.encode(),digest_size=16).digest())
    test=set(ordered[:max(1,round(len(ordered)*0.2))])
    return game_ids-test,test


def load_samples(path, max_train: int, max_test: int, seed: int, split='both'):
    if split not in ('both','train','test'):raise ValueError('不正なsplitです')
    train_games,test_games=split_game_ids(path,seed)
    # train/test単独では反対側のCSVを開かない。設定変更時も評価サンプルを訓練量から独立させる。
    if split=='train':path=[p for p in path if p[2]['game_id'] in train_games]
    elif split=='test':path=[p for p in path if p[2]['game_id'] in test_games]
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


def load_indexed_samples(paths, cap, seed, split):
    """学習・評価の既定の分割から指定側だけを抽出する。"""
    train_ids, test_ids = split_game_ids(paths, seed)
    selected_ids = train_ids if split == 'train' else test_ids
    subset = [p for p in paths if p[2]['game_id'] in selected_ids]
    return sample_indexed_rows(subset, cap, seed)


def sample_indexed_rows(subset, cap, seed):
    """渡された対局だけから行を重複なしで抽出する。"""
    from bisect import bisect_right
    from itertools import accumulate

    counts = [12 * int(info['training_rows']) for _, _, info in subset]
    if not counts or min(counts) <= 0:
        raise ValueError('保存済みの局面数を確認してください')
    offsets = [0, *accumulate(counts)]
    total = offsets[-1]
    samples = {info['game_id']: {} for _, _, info in subset}
    for offset in random.Random(seed).sample(range(total), min(cap, total)):
        game = bisect_right(offsets, offset) - 1
        state, view = divmod(offset - offsets[game], 12)
        gid = subset[game][2]['game_id']
        samples[gid].setdefault(state, set()).add(view)
    features = [c for c in TRAIN_COLUMNS if c not in META_COLUMNS and c not in TRAIN_LABEL_COLUMNS]
    rows = [(r['game_id'], r['observer'], r['target_player'], numeric_row(r, features), label_from_csv(r))
            for r in iter_csv_rows(subset, samples)]
    if len(rows) != min(cap, total):
        raise ValueError('CSVの局面数が保存済み一覧と一致しません')
    return features, rows, total


def remaining_tiles(inputs, feature_names, ops):
    """見えている牌を引き、赤牌を通常牌と同じ34種にまとめる。"""
    visible_indices = [feature_names.index(f'visible_{i}') for i in range(1, 38)]
    visible = ops.take(inputs, visible_indices, axis=-1)
    red_map = [[float(RED_TO_BASE[red] == tile) for tile in range(1, 35)]
               for red in range(35, 38)]
    reds = ops.matmul(visible[..., 34:], ops.convert_to_tensor(red_map, dtype='float32'))
    return ops.clip(4.0 - visible[..., :34] - reds, 0.0, 4.0)


def build_model(tf, x_train, feature_names, learning_rate):
    import numpy as np
    keras = tf.keras
    ops = keras.ops
    inputs = keras.Input(shape=(x_train.shape[1],))
    remaining = remaining_tiles(inputs, feature_names, ops)
    # 副露数から相手の伏せ牌の標準枚数を見積もり、均等配分を初期予測にする。
    target = ops.cast(inputs[:, feature_names.index('target_player')], 'int32')
    meld_indices = [feature_names.index(f'p{p}_meld_count') for p in range(4)]
    meld_counts = ops.take(inputs, meld_indices, axis=-1)
    target_melds = ops.sum(meld_counts * ops.one_hot(target, 4), axis=-1, keepdims=True)
    hand_size = 13.0 - 3.0 * target_melds
    actor = inputs[:, feature_names.index('actor')]
    action = inputs[:, feature_names.index('action')]
    drew = ops.logical_and(ops.equal(actor, ops.cast(target, 'float32')), ops.equal(action, 1.0))
    hand_size = hand_size + ops.expand_dims(ops.cast(drew, 'float32'), -1)
    probability = ops.clip(hand_size / ops.maximum(ops.sum(remaining, axis=-1, keepdims=True), 1.0),
                           0.001, 0.999)
    prior_logits = ops.log(probability) - ops.log1p(-probability)

    # 訓練側の統計量だけで正規化。小さなネットで初期予測を補正する。
    x = keras.layers.Normalization(mean=x_train.mean(axis=0),
                                    variance=np.maximum(x_train.var(axis=0), 1e-6))(inputs)
    for width in (64, 32):
        x = keras.layers.Dense(width, activation='relu',
                               kernel_regularizer=keras.regularizers.L2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)
    correction = keras.layers.Dense(34, kernel_initializer='zeros', bias_initializer='zeros')(x)
    outputs = remaining * ops.sigmoid(prior_logits + correction)
    model = keras.Model(inputs, outputs, name='visible_tile_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


def format_evaluation(metrics):
    """端末の最後に表示する評価側の数値。負の改善率もそのまま表示。"""
    mae=metrics['mae']
    baseline=metrics['mean_baseline_mae']
    improvement=baseline-mae
    rate=improvement/baseline*100 if baseline>0 else None
    lines=['', '========== 評価結果（未学習の対局） ==========',
           f'モデル MAE       : {mae:.6f}',
           f'平均予測 MAE     : {baseline:.6f}',
           f'MAE 改善幅       : {improvement:+.6f}',
           f'誤差削減率       : {rate:+.2f}%' if rate is not None else '誤差削減率       : 算出不可（平均予測MAEが0）']
    if 'mse' in metrics:lines.append(f'モデル MSE       : {metrics["mse"]:.6f}')
    if 'rmse' in metrics:lines.append(f'モデル RMSE      : {metrics["rmse"]:.6f}')
    for key,label in (
        ('count_accuracy', '枚数一致率'),
        ('presence_precision', '保有ありの適合率'),
        ('presence_recall', '保有ありの再現率'),
        ('presence_f1', '保有ありのF1'),
    ):
        if key in metrics:
            value = metrics[key]
            lines.append(f'{label}: {value:.2%}' if value is not None else f'{label}: 算出不可')
    lines+=['============================================']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str.upper, choices=["S", "M", "L"], default=str(MODE).upper())
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rank-policy",choices=["after","all"],default="after",help="学習対象の日付条件")
    parser.add_argument("--input-dir", type=Path, help="全件CSVの保存フォルダ")
    parser.add_argument("--players", type=int, help="今回の人数を指定。0=全員")
    parser.add_argument("--max-games", type=int, help="今回の学習に選ぶ対局数の上限")
    parser.add_argument("--name", help="モデル名。例: shiken_01")
    parser.add_argument("--output-dir", type=Path, help="モデルの保存先を指定")
    args = parser.parse_args()
    config = size_settings(args.mode)
    args.max_train = config['max_train'] if args.max_train is None else args.max_train
    args.max_test = config['max_test'] if args.max_test is None else args.max_test
    if min(args.max_train, args.max_test, args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("行数・epochs・batch-sizeは1以上、learning-rateは正の数です")
    if args.players is not None and args.players < 0:
        parser.error('--playersは0以上です')
    if args.max_games is not None and args.max_games < 2:
        parser.error('--max-gamesは2以上です')
    source = args.input_dir or (DATA_DIR/"converted"/"all" if args.rank_policy=="all" else INPUT_DIR)
    if args.name and (Path(args.name).name!=args.name or args.name in ('.','..') or '\\' in args.name):
        parser.error('--nameにはフォルダを含まない名前を指定してください')
    if args.name and args.output_dir:parser.error('--nameと--output-dirはどちらか一方を指定してください')
    output = args.output_dir or DATA_DIR/"models"/(args.name or MODEL_NAME)
    if not (source/"summary.json").is_file():
        parser.error(f"全件CSVがありません: {source}。先に python paifu_csv_henkan.py で変換してください")
    if (output/'tehai_yosoku.keras').exists():
        parser.error(f"既存モデルがあります。別の --output-dir を指定してください: {output}")
    paths, selected_players, selected_games = select_csvs(source,args.mode,args.seed,args.players,args.max_games,args.rank_policy)
    print(f'学習対象: {len(selected_players)}人 / {len(selected_games)}対局（{args.rank_policy}）',flush=True)
    os.environ.setdefault("KERAS_HOME", str(BASE_DIR/'.keras'))
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
    features, train, n_train = load_indexed_samples(paths, args.max_train, args.seed, 'train')
    if not train:raise SystemExit('訓練行がありません')
    reserved_train,reserved_test=split_game_ids(paths,args.seed)
    train_games={r[0] for r in train}
    if train_games & reserved_test:raise RuntimeError('対局分割の重複があります')
    columns = [features.index(name) for name in PREDICTION_FEATURES]
    x_train = np.asarray([r[3] for r in train],dtype=np.float32)[:, columns]
    features = PREDICTION_FEATURES
    y_train = np.asarray([r[4] for r in train],dtype=np.float32)
    if not all(np.isfinite(a).all() for a in [x_train,y_train]):raise ValueError('非有限の数値が含まれています')
    upper = remaining_tiles(x_train, features, tf.keras.ops).numpy()
    if (y_train > upper).any():
        raise ValueError('正解の枚数が公開情報から求めた上限を超えています。変換データを確認してください')
    model = build_model(tf, x_train, features, args.learning_rate)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(train),seed=args.seed).batch(args.batch_size).with_options(options)
    print(f"学習開始: 訓練{len(train)}行/{len(train_games)}対局、評価用に{len(reserved_test)}対局を確保（今回は評価しません）",flush=True)
    # 評価側はfitに渡さない。epoch選択や学習中の調整にも使用しない。
    history = model.fit(train_data, epochs=args.epochs, verbose=2, shuffle=False)
    input_path = source.resolve()
    saved_input = input_path.relative_to(BASE_DIR).as_posix() if input_path.is_relative_to(BASE_DIR) else str(input_path)
    settings=dict(format_version=1, sampler_version=3, model_name=output.name, architecture='visible_tile_model',
                  framework='TensorFlow',tensorflow_version=tf.__version__,mode=args.mode,
                  input=saved_input,seed=args.seed,max_train=args.max_train,max_test=args.max_test,
                  epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
                  requested_players=config['players'] if args.players is None else (args.players or None),
                  max_games=config['max_games'] if args.max_games is None else args.max_games,
                  train_rows=len(train),available_train_rows=n_train,train_game_ids=sorted(train_games),
                  reserved_train_game_ids=sorted(reserved_train),test_game_ids=sorted(reserved_test),
                  selected_players=[p['player'] for p in selected_players],feature_count=len(features),
                  train_label_mean=y_train.mean(axis=0).tolist(),rank_policy=args.rank_policy,game_overlap=0,
                  normalization='training data only; stored in model',
                  target='concealed hand tile counts, 34 types; reds merged',
                  split='seeded hash ordering of game IDs, 20% holdout; independent partition sampling')
    output.mkdir(parents=True, exist_ok=True)
    for filename,records in [('gakushu_players.csv',selected_players),('gakushu_games.csv',selected_games)]:
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    model.save(output/'tehai_yosoku.keras')
    # 保存したモデルを読み直して予測値の一致も確認する。
    restored = tf.keras.models.load_model(output/'tehai_yosoku.keras')
    np.testing.assert_allclose(restored(x_train[:5],training=False).numpy(),model(x_train[:5],training=False).numpy(),rtol=1e-5,atol=1e-6)
    settings['saved_model_reload_verified'] = True
    for name,value in [('gakushu_settei.json',settings),('feature_names.json',features),('gakushu_rireki.json',history.history)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'モデル作成完了: {output.resolve()}',flush=True)
    print('次に評価するコマンド:',flush=True)
    print(f'python moderu_jikko.py --action hyoka --model-dir "{output.resolve()}"',flush=True)



if __name__ == '__main__':
    main()
