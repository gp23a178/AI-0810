"""天鳳牌譜から、PIFWの公開局面に対応する4人麻雀モデルを学習する。"""
import argparse
import csv
from itertools import islice, zip_longest
import json
import os
from pathlib import Path
import random
import tempfile
from urllib.parse import unquote

from aite_tehai_gakushu import BASE_DIR, DATA_DIR, split_game_ids, select_csvs
from paifu_csv_henkan import (
    ACTION_INITIAL, ACTION_DRAW, ACTION_DISCARD, ACTION_MELD, ACTION_REACH, ACTION_DORA,
    REFERENCE_COLUMNS, REFERENCE_EXTRA_COLUMNS, cached_zip, convert_one, read_xml,
)
from pifw_tehai_yosoku import (
    PRE_DISCARD_SCHEMA, SCHEMA, FEATURE_PROFILES, MSE_BASELINE, MSE_BASELINE_SOURCE,
    HandPredictor, assign_hands, encode_observation, feature_names, model_path, tile_counts,
)

RED_BASE = {35: 5, 36: 14, 37: 23}
LOSS_MODES = ('mse', 'weighted_negative', 'hard_negative')
COMPARISON_MODELS = (MSE_BASELINE, 'pifw_weighted_negative', 'pifw_hard_negative')
DEFAULT_NEGATIVE_WEIGHT = 0.25
DEFAULT_HARD_K = 5
EVENT_NAMES = {
    ACTION_INITIAL: 'initial', ACTION_DRAW: 'draw', ACTION_DISCARD: 'discard',
    ACTION_MELD: 'meld', ACTION_REACH: 'riichi', ACTION_DORA: 'dora',
}


def tenhou_tiles(counts):
    return [RED_BASE.get(tile, tile) for tile in range(1, 38) for _ in range(int(counts(tile)))]


def examples(paths, profile, schema):
    """1局面を各観測席から見た3人分の予測に展開する。"""
    for path, meta, info in paths:
        game = info['game_id']
        with path.open(encoding='utf-8-sig', newline='') as file, meta.open(encoding='utf-8-sig', newline='') as extra:
            rows, aux = csv.DictReader(file), csv.DictReader(extra)
            if rows.fieldnames != REFERENCE_COLUMNS or aux.fieldnames != REFERENCE_EXTRA_COLUMNS:
                raise ValueError(f'変換済みCSVの形式が違います: {path}')
            for index, (row, public) in enumerate(zip_longest(rows, aux)):
                if row is None or public is None or int(public['state_index']) != index:
                    raise ValueError(f'CSVと補助CSVの局面が一致しません: {game}/{index}')
                action = int(row['action'])
                actor = int(row['actor'])
                if schema == PRE_DISCARD_SCHEMA and action not in (ACTION_DRAW, ACTION_MELD):
                    continue
                if action not in EVENT_NAMES:
                    raise ValueError(f'未対応の局面の種類: {game}/{index}')
                meld_counts = [int(public[f'p{seat}_meld_count']) for seat in range(4)]
                hands = [tenhou_tiles(lambda tile: row[f'p{seat}_hand_{tile}']) for seat in range(4)]
                concealed = [len(hand) for hand in hands]
                if schema == PRE_DISCARD_SCHEMA and action == ACTION_MELD:
                    if concealed[actor] != 14 - 3 * meld_counts[actor]:
                        continue
                rivers = []
                for seat in range(4):
                    river = []
                    for n in range(1, 25):
                        tile = abs(int(row[f'p{seat}_disc_{n}']))
                        if tile and not int(public[f'p{seat}_disc_called_{n}']):
                            river.append(RED_BASE.get(tile, tile))
                    rivers.append(river)
                public_observation = dict(
                    schema=schema, rivers=rivers, meld_counts=meld_counts,
                    meld_tiles=[tenhou_tiles(lambda tile: row[f'p{seat}_meld_{tile}']) for seat in range(4)],
                    riichi=[int(row[f'p{seat}_is_reach']) for seat in range(4)],
                    dora_indicators=[RED_BASE.get(tile, tile) for n in range(1, 6)
                                     if (tile := int(row[f'dora_{n}']))],
                )
                if schema == SCHEMA:
                    public_observation['concealed_sizes'] = concealed
                hand_counts = [tile_counts(hand) for hand in hands]
                observers = range(4) if schema == SCHEMA else (actor,)
                for observer in observers:
                    # 他家について渡すのは、対局者から見える牌の枚数だけ。
                    observation = dict(public_observation, observer=observer, own_hand=hands[observer])
                    features, targets, remaining, sizes = encode_observation(observation, profile)
                    labels = [hand_counts[seat] for seat in targets]
                    if [sum(label) for label in labels] != sizes or any(
                        sum(counts) > cap for counts, cap in zip(zip(*labels), remaining)
                    ):
                        raise ValueError(f'教師ラベルと公開情報の枚数が一致しません: {game}/{index}')
                    yield features, labels, action


def hand_count_loss(tf, mode, weight, hard_k):
    if mode == 'mse':
        return 'mse'

    def loss(actual, expected):
        actual = tf.cast(actual, expected.dtype)
        squared = tf.square(expected - actual)
        # The output cap forces publicly exhausted tiles to exactly zero.
        negative = (actual == 0) & (expected > 0)
        penalty = tf.where(negative, squared, tf.zeros_like(squared))
        count = tf.reduce_sum(tf.cast(negative, expected.dtype), axis=-1)
        if mode == 'hard_negative':
            penalty = tf.math.top_k(penalty, k=hard_k, sorted=False).values
            count = tf.minimum(count, tf.cast(hard_k, expected.dtype))
        extra = tf.math.divide_no_nan(tf.reduce_sum(penalty, axis=-1), count)
        return tf.reduce_mean(squared, axis=-1) + weight * extra

    return loss


def build_model(tf, train, width, learning_rate, loss='mse'):
    """残り牌比例の基準値を、公開局面から学習するネットワークで補正する。"""
    import numpy as np

    keras = tf.keras
    ops = keras.ops
    features = keras.Input(shape=(train.shape[-1],), name='features')
    remaining = features[:, -35:-1]
    size = features[:, -1:]
    ratio = ops.clip(size / ops.maximum(ops.sum(remaining, axis=-1, keepdims=True), 1), 0.001, 0.999)
    prior = ops.log(ratio) - ops.log1p(-ratio)
    mean = train.mean(axis=0, dtype=np.float32)
    variance = np.maximum(train.var(axis=0, dtype=np.float32), 1e-6)
    x = keras.layers.Normalization(mean=mean, variance=variance)(features)
    for units in (width, max(8, width // 2)):
        x = keras.layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.L2(0.001))(x)
        x = keras.layers.Dropout(0.15)(x)
    adjustment = keras.layers.Dense(34, kernel_initializer='zeros', bias_initializer='zeros')(x)
    model = keras.Model(features, remaining * ops.sigmoid(prior + adjustment))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss, metrics=['mae'])
    return model


def load_dataset(paths, ids, profile, schema=SCHEMA):
    import numpy as np

    selected = [path for path in paths if path[2]['game_id'] in ids]
    if set(ids) != {path[2]['game_id'] for path in selected}:
        raise ValueError('対象の対局CSVが揃っていません')
    rows = examples(selected, profile, schema)
    xs, ys, events = [], [], []
    while batch := list(islice(rows, 1024)):
        x, y, event = zip(*batch)
        xs.append(np.asarray(x, dtype=np.uint8))
        ys.append(np.asarray(y, dtype=np.uint8))
        events.append(np.asarray(event, dtype=np.uint8))
    if not xs:
        raise ValueError('学習または評価に使用できる局面がありません')
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(events)


def count_metrics(predictions, labels, remaining):
    import numpy as np

    present = labels > 0
    positive = predictions >= 0.5
    true_positive = int((present & positive).sum())
    predicted_positive = int(positive.sum())
    known_positive = int(present.sum())
    feasible_negative = (labels == 0) & (remaining[:, None, :] > 0)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / known_positive if known_positive else 0.0
    return {
        'model_mse': float(np.square(predictions - labels).mean()),
        'expected_total_per_opponent': float(predictions.sum(axis=-1).mean()),
        'actual_total_per_opponent': float(labels.sum(axis=-1).mean()),
        'feasible_negative_mean_expected': float(predictions[feasible_negative].mean())
        if feasible_negative.any() else 0.0,
        'presence': {
            'threshold_expected_count': 0.5,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'false_positives': predicted_positive - true_positive,
            'false_negatives': known_positive - true_positive,
        },
    }


def evaluate(predict, features, labels, seed, events=None, *, detailed=False):
    import numpy as np

    x = features.reshape(-1, features.shape[-1])
    predictions = np.concatenate([predict(x[i:i+256]) for i in range(0, len(x), 256)])
    predictions = predictions.reshape(labels.shape)
    remaining, sizes = features[:, 0, -35:-1], features[:, :, -1]
    baseline = sizes[:, :, None] * remaining[:, None, :] / remaining.sum(axis=1)[:, None, None]
    if (not np.isfinite(predictions).all() or (predictions < 0).any()
            or (predictions > remaining[:, None, :] + 1e-6).any()):
        raise ValueError('予測枚数が公開情報の制約を満たしません')
    indices = random.Random(seed).sample(range(len(features)), min(256, len(features)))
    matched = {}
    for strategy in ('uniform', 'proportional', 'joint'):
        total = 0
        for i in indices:
            hands, _ = assign_hands(predictions[i], remaining[i].astype(int).tolist(),
                                    sizes[i].astype(int).tolist(), strategy, seed + i)
            total += int(np.minimum(hands, labels[i]).sum())
        matched[strategy] = total / (3 * len(indices))
    errors = np.abs(predictions - labels)
    baseline_errors = np.abs(baseline - labels)
    metrics = {'model_mae': float(errors.mean()), 'uniform_expected_mae': float(baseline_errors.mean()),
               'allocation_cases': len(indices), 'mean_matched_tiles_per_opponent': matched}
    if detailed:
        metrics.update(count_metrics(predictions, labels, remaining))
    if events is not None:
        metrics['by_event'] = {
            name: {'positions': int(mask.sum()) // 4, 'model_mae': float(errors[mask].mean()),
                   'uniform_expected_mae': float(baseline_errors[mask].mean())}
            for event, name in EVENT_NAMES.items() if (mask := events == event).any()
        }
    return metrics


def achievement_day_games(directory, players, excluded):
    """達成日当日に、対象者が天鳳位として開始した四麻だけを選ぶ。"""
    by_name = {player['player']: player for player in players}
    selected = {}
    with (directory / 'logs_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            player = by_name.get(row['source_player'])
            game_id = row['game_id']
            if (player is None or not game_id or game_id in excluded or game_id in selected
                    or row['game_date'] != player['achievement_date']):
                continue
            archive = DATA_DIR / 'archives' / Path(player['zip_filename']).name
            root = read_xml(cached_zip(str(archive)).read(row['member']))
            info = next((tag for tag in root.iter('UN') if 'dan' in tag.attrib), None)
            if info is None:
                continue
            ranks = info.attrib['dan'].split(',')
            seats = [seat for seat in range(4) if unquote(info.get(f'n{seat}', '')) == player['player']]
            if len(seats) == 1 and len(ranks) == 4 and ranks[seats[0]] == '20':
                selected[game_id] = {**row, 'source_zip': str(archive)}
    if not selected:
        raise ValueError('達成日当日に天鳳位で開始した追加対局が見つかりません')
    return list(selected.values())


def convert_extra_games(games, directory):
    (directory / 'games').mkdir()
    paths = []
    for game in games:
        result, error = convert_one((game, str(directory), False))
        if error:
            raise ValueError(f"{error['game_id']}: {error['error']}")
        paths.append((directory / result['csv_file'], directory / result['meta_file'], result))
    return paths


def evaluation_ids(manifest):
    train_ids = set(manifest['train_game_ids'])
    test_ids = set(manifest['test_game_ids'])
    if not test_ids or train_ids & test_ids:
        raise ValueError('評価対局が空、または学習対局と重複しています')
    return test_ids


def train(args, output):
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    paths, players, _ = select_csvs(args.input_dir, 'L', args.seed)
    train_ids, test_ids = split_game_ids(paths, args.seed)
    extra_ids = set()
    if args.include_achievement_day:
        games = achievement_day_games(args.input_dir, players, train_ids | test_ids)
        extra_ids = {game['game_id'] for game in games}
        train_ids |= extra_ids
        print(f'達成日当日の {len(extra_ids)} 対局を学習側だけに追加します', flush=True)
    print(f'公開局面を抽出中: 学習 {len(train_ids)}対局 / 評価 {len(test_ids)}対局', flush=True)
    if extra_ids:
        with tempfile.TemporaryDirectory(prefix='pifw-achievement-day-') as tmp:
            extra_paths = convert_extra_games(games, Path(tmp))
            x_train, y_train, _ = load_dataset(paths + extra_paths, train_ids, args.features)
    else:
        x_train, y_train, _ = load_dataset(paths, train_ids, args.features)
    x_test, y_test, test_events = load_dataset(paths, test_ids, args.features)
    order = np.random.default_rng(args.seed).permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]
    names = feature_names(args.features)
    flat_x, flat_y = x_train.reshape(-1, len(names)), y_train.reshape(-1, 34)
    print(f'学習 {len(x_train)//4}局面・{len(flat_x)}行 / 評価 {len(x_test)//4}局面 / 特徴量 {len(names)}',
          flush=True)

    loss = hand_count_loss(tf, args.loss, args.negative_weight, args.hard_k)
    model = build_model(tf, flat_x, args.width, args.learning_rate, loss)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    data = tf.data.Dataset.from_tensor_slices((flat_x, flat_y)).shuffle(
        min(len(flat_x), 32768), seed=args.seed).batch(args.batch_size).with_options(options)
    history = model.fit(data, epochs=args.epochs, verbose=2, shuffle=False)
    metrics = evaluate(lambda x: model(x, training=False).numpy(), x_test, y_test, args.seed, test_events)

    export = tf.Module()
    export.model = model

    @tf.function(input_signature=[tf.TensorSpec([None, len(names)], tf.float32, name='features')])
    def serve(features):
        return {'expected_counts': export.model(features, training=False)}

    manifest = {
        'model_name': args.name, 'observation_schema': SCHEMA, 'feature_profile': args.features,
        'feature_names': names, 'dtype': 'float32', 'input': 'features', 'output': 'expected_counts',
        'output_meaning': 'expected concealed counts for tile IDs 1 to 34', 'tensorflow': tf.__version__,
        'training_data': 'Tenhou four-player logs', 'seed': args.seed, 'epochs': args.epochs,
        'width': args.width, 'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
        'loss': args.loss, 'negative_weight': args.negative_weight, 'hard_k': args.hard_k,
        'train_game_ids': sorted(train_ids), 'test_game_ids': sorted(test_ids),
        'achievement_day_train_game_ids': sorted(extra_ids),
        'train_positions': len(x_train) // 4, 'test_positions': len(x_test) // 4,
        'observers_per_position': 4, 'train_target_rows': len(flat_x), 'test_target_rows': len(x_test) * 3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pifw-export-', dir=output.parent) as tmp:
        staging = Path(tmp) / args.name
        staging.mkdir()
        keras_path = staging / 'tehai_yosoku.keras'
        # Export the custom-loss models for inference without serializing the loss.
        keras_model = model if args.loss == 'mse' else tf.keras.Model(model.inputs, model.outputs)
        keras_model.save(keras_path)
        tf.saved_model.save(export, str(staging / 'saved_model'), signatures={'serving_default': serve})
        signature = tf.saved_model.load(str(staging / 'saved_model')).signatures['serving_default']
        sample = tf.convert_to_tensor(x_test[0], dtype=tf.float32)
        np.testing.assert_allclose(signature(features=sample)['expected_counts'].numpy(),
                                   model(sample, training=False).numpy(), rtol=1e-5, atol=1e-6)
        if args.loss != 'mse':
            restored = tf.keras.models.load_model(keras_path)
            np.testing.assert_allclose(restored(sample, training=False).numpy(),
                                       model(sample, training=False).numpy(), rtol=1e-5, atol=1e-6)
        for name, value in [('model.json', manifest), ('metrics.json', metrics), ('history.json', history.history)]:
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f'モデル作成完了: {output}', flush=True)


def evaluate_saved(args, output):
    predictor = HandPredictor(output)
    manifest = predictor.manifest
    test_ids = evaluation_ids(manifest)
    paths, _, _ = select_csvs(args.input_dir, 'L', manifest['seed'])
    x, y, events = load_dataset(paths, test_ids, predictor.profile, predictor.schema)
    metrics = evaluate(predictor.predict_features, x, y, manifest['seed'],
                       events if predictor.schema == SCHEMA else None)
    metrics.update(test_games=len(manifest['test_game_ids']),
                   test_positions=len(x) // manifest.get('observers_per_position', 1))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def compare_saved(args):
    paths = [model_path(name).resolve() for name in args.names]
    if len(set(paths)) != len(paths):
        raise ValueError('同じ保存済みモデルが重複しています')
    manifests = [json.loads((path / 'model.json').read_text(encoding='utf-8')) for path in paths]
    first = manifests[0]
    for manifest in manifests:
        evaluation_ids(manifest)
    comparable = ('observation_schema', 'feature_names', 'test_game_ids',
                  'seed', 'epochs', 'width', 'learning_rate', 'batch_size')
    if not args.allow_different_training_games:
        comparable += ('train_game_ids',)
    for path, manifest in zip(paths[1:], manifests[1:]):
        differences = [key for key in comparable if manifest[key] != first[key]]
        if differences:
            raise ValueError(f'{path.name} の学習・評価条件が異なります: {", ".join(differences)}')
    files, _, _ = select_csvs(args.input_dir, 'L', first['seed'])
    schema = first['observation_schema']
    features, labels, events = load_dataset(files, first['test_game_ids'], first['feature_profile'], schema)
    result = {}
    for name, path, manifest in zip(args.names, paths, manifests):
        predictor = HandPredictor(path)
        metrics = evaluate(predictor.predict_features, features, labels, first['seed'],
                           events if schema == SCHEMA else None, detailed=True)
        result[name] = {
            'saved_as': path.name, 'loss': manifest.get('loss', 'mse'),
            'train_games': len(manifest['train_game_ids']),
            'achievement_day_train_games': len(manifest.get('achievement_day_train_game_ids', [])),
            **metrics,
        }
    print(json.dumps({
        'test_games': len(first['test_game_ids']),
        'test_positions': len(features) // first.get('observers_per_position', 1),
        'models': result,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    training = actions.add_parser('train', help='学習、評価、モデルの保存')
    scoring = actions.add_parser('evaluate', help='保存済みモデルと同じ評価対局で再評価')
    comparison = actions.add_parser('compare', help='同じ特徴量と学習条件のモデルを共通局面で比較')
    for action in (training, scoring):
        action.add_argument('--name', default=MSE_BASELINE_SOURCE)
    for action in (training, scoring, comparison):
        action.add_argument('--input-dir', type=Path, default=DATA_DIR / 'converted/after')
    comparison.add_argument('--names', nargs='+', default=COMPARISON_MODELS)
    comparison.add_argument('--allow-different-training-games', action='store_true',
                            help='学習対局数を変えたモデルを同じ評価対局で比較する')
    training.add_argument('--include-achievement-day', action='store_true',
                          help='達成当日に天鳳位として開始した牌譜を学習側に追加する')
    training.add_argument('--features', choices=FEATURE_PROFILES, default='counts')
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--width', type=int, default=64)
    training.add_argument('--batch-size', type=int, default=128)
    training.add_argument('--learning-rate', type=float, default=0.001)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--loss', choices=LOSS_MODES, default='mse')
    training.add_argument('--negative-weight', type=float)
    training.add_argument('--hard-k', type=int)
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    try:
        if args.action == 'compare':
            if len(args.names) < 2:
                comparison.error('比較には2つ以上のモデル名が必要です')
            compare_saved(args)
        elif args.action == 'evaluate':
            evaluate_saved(args, model_path(args.name))
        else:
            output = model_path(args.name)
            if min(args.epochs, args.width, args.batch_size) < 1 or not 0 < args.learning_rate < float('inf'):
                training.error('学習条件は正の数で指定してください')
            if args.negative_weight is not None:
                if args.loss == 'mse' or not 0 < args.negative_weight < float('inf'):
                    training.error('負例を使う損失に正の --negative-weight を指定してください')
            if args.hard_k is not None and (args.loss != 'hard_negative' or not 1 <= args.hard_k <= 34):
                training.error('--hard-k は hard_negative で1から34の範囲です')
            args.negative_weight = 0.0 if args.loss == 'mse' else (args.negative_weight or DEFAULT_NEGATIVE_WEIGHT)
            args.hard_k = (args.hard_k or DEFAULT_HARD_K) if args.loss == 'hard_negative' else None
            if output.exists() or args.name == MSE_BASELINE:
                training.error(f'同名または予約済みのモデル名です: {args.name}')
            train(args, output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
