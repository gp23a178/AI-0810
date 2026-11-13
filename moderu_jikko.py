# -*- coding: utf-8 -*-
"""名前を指定して保存済みモデルを実行。既定は評価、--action yosokuで1局面予測。"""
import argparse
import csv
import json
import os
import random
import secrets
from collections import deque
from itertools import groupby
import zipfile
from pathlib import Path
from aite_tehai_gakushu import (BASE_DIR,DATA_DIR,load_samples,sample_indexed_rows,split_game_ids,
                                format_evaluation,META_COLUMNS,iter_csv_rows,numeric_row,remaining_tiles,label_from_csv,reservoir_add,validate_feature_names)
from paifu_csv_henkan import TRAIN_COLUMNS,TRAIN_LABEL_COLUMNS

# ==================== 実行設定 ====================
MODEL_NAME = "shiken_01"  # 実行したいモデル名だけを変更できます
ACTION = "sannin"         # 既定は他家3人の手牌予測

def training_settings_path(directory):
    path = directory/'gakushu_settei.json'
    return path if path.is_file() else directory/'gakushu_json'


def resolve_csv_root(saved_input, override=None):
    """モデルを別の端末へコピーした場合は、プロジェクト内の相対位置を使う。"""
    if override is not None:
        candidates = [Path(override).resolve()]
    else:
        saved = Path(saved_input)
        candidates = []
        # WindowsとWSLで保存された既存の絶対パスを、clone先へ置き換える。
        parts = saved_input.replace(chr(92), '/').split('/')
        if 'tenhou_data' in parts:
            start = parts.index('tenhou_data')
            candidates.append(BASE_DIR.joinpath(*parts[start:]).resolve())
        candidates.append((BASE_DIR / saved).resolve())
    for path in candidates:
        if (path/'selected_games.csv').is_file():
            return path
    checked = ', '.join(str(path) for path in candidates)
    raise ValueError(f'変換済みCSVの一覧が見つかりません。--input-dir で指定してください。確認先: {checked}')


def saved_paths(root,records):
    root=Path(root).resolve()
    paths=[]
    for r in records:
        pair=[]
        for key in ['csv_file','meta_file']:
            p=(root/r[key]).resolve()
            if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'CSVがありません: {p}')
            pair.append(p)
        paths.append((*pair,{k:r.get(k,'') for k in ['game_id','game_date','source_player','source_zip','training_rows']}))
    return paths


def choose_evaluation_games(args, paths):
    """モデルに保存済みの未学習対局から評価対象を選ぶ。"""
    seed = args.eval_seed if args.eval_seed is not None else secrets.randbits(63)
    ordered = sorted(paths, key=lambda p: p[2]['game_id'])
    if not ordered:
        raise ValueError('評価用の対局がありません')
    if args.test_games is not None:
        if args.test_games > len(ordered):
            raise ValueError(f'評価用の対局は{len(ordered)}件です。--test-gamesを減らしてください')
        ordered = sorted(random.Random(seed).sample(ordered, args.test_games),
                         key=lambda p: p[2]['game_id'])
    return ordered, seed


def count_metrics(actual, predicted):
    """四捨五入した枚数から牌種単位の一致率とマイクロF1を計算。"""
    import numpy as np

    rounded = np.floor(predicted + 0.5)
    actual_has_tile = actual > 0
    predicted_has_tile = rounded > 0
    true_positive = int(np.count_nonzero(actual_has_tile & predicted_has_tile))
    actual_positive = int(np.count_nonzero(actual_has_tile))
    predicted_positive = int(np.count_nonzero(predicted_has_tile))
    return {
        'count_accuracy': float((rounded == actual).mean()),
        'presence_precision': true_positive / predicted_positive if predicted_positive else None,
        'presence_recall': true_positive / actual_positive if actual_positive else None,
        'presence_f1': (2 * true_positive / (actual_positive + predicted_positive)
                        if actual_positive + predicted_positive else None),
    }


def evaluate(args):
    directory=args.model_dir.resolve()
    config_path=training_settings_path(directory)
    legacy=not config_path.exists()
    config=json.loads((directory/'metrics.json' if legacy else config_path).read_text())
    names=json.loads((directory/'feature_names.json').read_text())
    validate_feature_names(names)
    with (directory/'gakushu_games.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    paths=saved_paths(resolve_csv_root(config['input'], args.input_dir),records)
    train_ids,test_ids=split_game_ids(paths,config['seed'])
    if set(config['train_game_ids']) & test_ids or test_ids!=set(config['test_game_ids']):raise ValueError('保存時の対局分割と一致しません')
    saved_cap = config['test_rows'] if legacy else config['max_test']
    cap_test = args.test_rows if args.test_rows is not None else saved_cap
    held_out = [path for path in paths if path[2]['game_id'] in test_ids]
    selected, eval_seed = choose_evaluation_games(args, held_out)
    features,test,nv = sample_indexed_rows(selected,cap_test,eval_seed)
    # 旧モデルの比較基準だけ、当時の訓練行抽出で再現する。
    train = []
    if legacy:
        _,train,_,_,_ = load_samples(paths,config['train_rows'],saved_cap,config['seed'],split='both')
    used_ids = {row[0] for row in test}
    if not test:raise ValueError('評価行がありません')
    os.environ.setdefault('KERAS_HOME',str(BASE_DIR/'.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import numpy as np
    import tensorflow as tf
    model=tf.keras.models.load_model(directory/'tehai_yosoku.keras',compile=False)
    full_x=np.asarray([r[3] for r in test],dtype=np.float32)
    x=full_x[:, [features.index(name) for name in names]]
    y=np.asarray([r[4] for r in test],dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('評価データに非有限値があります')
    batch=config['batch_size']
    predicted=np.concatenate([model(x[i:i+batch],training=False).numpy() for i in range(0,len(x),batch)])
    if predicted.shape!=y.shape or not np.isfinite(predicted).all():raise ValueError('予測結果が不正です')
    baseline=np.asarray([r[4] for r in train],dtype=np.float32).mean(axis=0) if legacy else np.asarray(config['train_label_mean'],dtype=np.float32)
    mae=float(np.abs(y-predicted).mean())
    base=float(np.abs(y-baseline).mean())
    mse=float(np.square(y-predicted).mean())
    metrics=dict(model_name=directory.name,model_dir=str(directory),mae=mae,mean_baseline_mae=base,
                 mae_improvement=base-mae,mae_reduction_percent=(base-mae)/base*100 if base>0 else None,
                 mse=mse,rmse=float(np.sqrt(mse)),train_rows=config['train_rows'],test_rows=len(test),
                 available_test_rows=nv,train_game_ids=config['train_game_ids'],test_game_ids=sorted(used_ids),
                 eligible_test_game_ids=sorted(test_ids), selected_test_game_ids=[p[2]['game_id'] for p in selected],
                 eval_seed=eval_seed, requested_test_rows=cap_test, requested_test_games=args.test_games,
                 game_overlap=0,seed=config['seed'],input=config['input'],
                 evaluated_saved_model=True,legacy_sampling_reproduced=legacy)
    metrics.update(count_metrics(y, predicted))
    if config.get('architecture')=='visible_tile_model':
        upper=remaining_tiles(x,names,tf.keras.ops).numpy()
        if (y>upper).any() or (predicted>upper+1e-6).any() or (predicted<0).any():
            raise ValueError('見えている牌の枚数から計算した上限と矛盾があります')
        metrics['tile_bounds_verified']=True
    if args.compare:
        other=DATA_DIR/'models'/args.compare
        other_config,other_metrics=saved_info(other)
        other_train_ids = other_config.get('train_game_ids',other_metrics.get('train_game_ids'))
        if other_train_ids is None:
            raise ValueError('比較モデルの学習対局一覧がありません')
        if set(other_train_ids) & used_ids:
            raise ValueError('比較モデルの訓練対局が今回の評価対局に含まれています')
        other_names=validate_feature_names(json.loads((other/'feature_names.json').read_text()))
        other_x=full_x[:, [features.index(name) for name in other_names]]
        old=tf.keras.models.load_model(other/'tehai_yosoku.keras',compile=False)
        previous=np.concatenate([old(other_x[i:i+batch],training=False).numpy() for i in range(0,len(other_x),batch)])
        if previous.shape!=y.shape or not np.isfinite(previous).all():
            raise ValueError('比較モデルの予測結果が不正です')
        old_mae=float(np.abs(y-previous).mean())
        metrics.update(comparison_model=args.compare, comparison_mae=old_mae,
                       comparison_mae_improvement=old_mae-mae)
        print(f'同一評価行の旧モデル {args.compare} MAE: {old_mae:.6f}')
        print(f'旧モデルからのMAE改善幅: {old_mae-mae:+.6f}')
    out=args.output_dir or directory/'hyoka'
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'validation_examples.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['game_id','observer','target_player','actual_counts','predicted_counts'])
        for record,yp in zip(test[:20],predicted[:20]):w.writerow([*record[:3],json.dumps(record[4]),json.dumps(yp.tolist())])
    print(f'モデル: {directory.name} / 評価: {len(test)}行・{len(used_ids)}対局',flush=True)
    print(f'評価seed: {eval_seed}')
    if len(test)==nv:
        print('選択した対局の全行が評価対象です。抽出し直すには --test-rows を減らしてください。')
    print(f'評価結果: {(out/"metrics.json").resolve()}',flush=True)
    print(format_evaluation(metrics),flush=True)


def allocate_hands(scores, remaining, sizes):
    """最小費用流で、枚数制約内かつ予測値に最も近い整数の手牌を求める。"""
    import numpy as np

    player_count, tile_count = scores.shape
    source = 0
    first_tile = 1 + player_count
    sink = first_tile + tile_count
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity, cost):
        forward = [end, len(graph[end]), capacity, cost]
        graph[end].append([start, len(graph[start]), 0, -cost])
        graph[start].append(forward)
        return forward

    tile_edges = {}
    for seat, size in enumerate(sizes):
        add_edge(source, seat + 1, int(size), 0)
        for tile, cap in enumerate(remaining):
            # n枚目を足したときの二乗誤差の増分。整数化して同点時の挙動を安定させる。
            tile_edges[seat, tile] = [
                add_edge(seat + 1, first_tile + tile, 1,
                         round(10000 * (2 * n - 1 - 2 * scores[seat, tile])))
                for n in range(1, min(int(cap), int(size)) + 1)
            ]
    for tile, cap in enumerate(remaining):
        add_edge(first_tile + tile, sink, int(cap), 0)

    for _ in range(int(sum(sizes))):
        distance = [float('inf')] * len(graph)
        previous = [None] * len(graph)
        distance[source] = 0
        queue = deque([source])
        queued = {source}
        while queue:
            start = queue.popleft()
            queued.remove(start)
            for i, (end, reverse, capacity, cost) in enumerate(graph[start]):
                if capacity and distance[start] + cost < distance[end]:
                    distance[end] = distance[start] + cost
                    previous[end] = (start, i)
                    if end not in queued:
                        queue.append(end)
                        queued.add(end)
        if previous[sink] is None:
            raise ValueError('公開情報から求めた残り牌では、必要な手牌枚数を満たせません')
        end = sink
        while end != source:
            start, i = previous[end]
            edge = graph[start][i]
            edge[2] -= 1
            graph[end][edge[1]][2] += 1
            end = start

    result = np.zeros_like(scores, dtype=np.int32)
    for (seat, tile), edges in tile_edges.items():
        result[seat, tile] = sum(edge[2] == 0 for edge in edges)
    if not np.array_equal(result.sum(axis=1), sizes) or (result.sum(axis=0) > remaining).any():
        raise RuntimeError('牌の配分が枚数制約を満たしていません')
    return result


def evaluate_joint(args):
    """固定済みの評価対局から、打牌直後の同一局面で3方式を比較する。"""
    directory = args.model_dir.resolve()
    config, metrics = saved_info(directory)
    config = config or metrics
    names = json.loads((directory/'feature_names.json').read_text(encoding='utf-8'))
    validate_feature_names(names)
    test_ids = set(config['test_game_ids'])
    if test_ids & set(config['train_game_ids']):
        raise ValueError('学習対局と評価対局が重複しています')
    with (directory/'gakushu_games.csv').open(encoding='utf-8-sig', newline='') as file:
        records = [row for row in csv.DictReader(file) if row['game_id'] in test_ids]
    if {r['game_id'] for r in records} != test_ids:
        raise ValueError('保存済みの評価対局が揃っていません')
    paths, eval_seed = choose_evaluation_games(args, saved_paths(resolve_csv_root(config['input'], args.input_dir), records))

    cases = []
    rng = random.Random(eval_seed)
    seen = 0
    def case_key(row):
        return row['game_id'], row['state_index'], row['observer']

    for key, group in groupby(iter_csv_rows(paths, action=2), key=case_key):
        rows = list(group)
        if len(rows) != 3 or {r['target_player'] for r in rows} != (set(range(4)) - {key[2]}):
            raise ValueError(f'他家3席が揃っていません: {key}')
        seen += 1
        reservoir_add(cases, (key, rows), seen, args.cases, rng)
    if not cases:
        raise ValueError('評価対局に打牌直後の局面がありません')

    os.environ.setdefault('KERAS_HOME', str(BASE_DIR/'.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    model = tf.keras.models.load_model(directory/'tehai_yosoku.keras', compile=False)
    x = np.asarray([numeric_row(row, names) for _, rows in cases for row in rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('評価入力に非有限値があります')
    predicted = model.predict(x, batch_size=128, verbose=0)
    if predicted.shape != (len(x), 34) or not np.isfinite(predicted).all():
        raise ValueError('予測結果が不正です')
    predicted = predicted.reshape(-1, 3, 34)
    limits = remaining_tiles(x[::3], names, tf.keras.ops).numpy().astype(np.int32)
    methods = ('remaining_only', 'independent', 'joint')
    per_case = []
    for i, (key, rows) in enumerate(cases):
        remaining = limits[i]
        # 打牌直後の伏せ牌数は公開された副露・槓の数で決まる。
        sizes = np.array([13 - 3 * int(r[f"p{r['target_player']}_meld_count"]) for r in rows])
        actual = np.asarray([label_from_csv(r) for r in rows], dtype=np.int32)
        if (sizes < 1).any() or not np.array_equal(actual.sum(axis=1), sizes) or (actual.sum(axis=0) > remaining).any():
            raise ValueError(f'牌譜の正解と公開情報の枚数が一致しません: {key}')

        uniform = sizes[:, None] * remaining[None, :] / remaining.sum()
        independent = np.concatenate([
            allocate_hands(predicted[i, j:j+1], remaining, sizes[j:j+1]) for j in range(3)
        ])
        estimates = (allocate_hands(uniform, remaining, sizes),
                     independent,
                     allocate_hands(predicted[i], remaining, sizes))
        result = dict(game_id=key[0], state_index=key[1], observer=key[2], actual_tiles=int(sizes.sum()))
        for method, estimate in zip(methods, estimates):
            result[f'{method}_matches'] = int(np.minimum(estimate, actual).sum())
            result[f'{method}_mae'] = float(np.abs(estimate - actual).mean())
            result[f'{method}_violation'] = int((estimate.sum(axis=0) > remaining).any())
        per_case.append(result)

    def summarize(rows):
        result = {'cases': len(rows)}
        for method in methods:
            matches = sum(row[f'{method}_matches'] for row in rows)
            result[method] = {
                'matched_tiles_per_hand': matches / (3 * len(rows)),
                'matched_tile_rate': matches / sum(row['actual_tiles'] for row in rows),
                'mae': sum(row[f'{method}_mae'] for row in rows) / len(rows),
                'violation_cases': sum(row[f'{method}_violation'] for row in rows),
            }
        result['joint_gain_per_hand'] = (result['joint']['matched_tiles_per_hand']
                                         - result['independent']['matched_tiles_per_hand'])
        return result

    report = summarize(per_case)
    report['test_game_ids'] = sorted(test_ids)
    report['available_cases'] = seen
    report['eval_seed'] = eval_seed
    report['selected_test_game_ids'] = [p[2]['game_id'] for p in paths]
    report['per_game'] = {gid: summarize([row for row in per_case if row['game_id'] == gid])
                          for gid in sorted({row['game_id'] for row in per_case})}
    out = args.output_dir or directory/'douji_hyoka'
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out/'cases.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    print(f'評価: {len(per_case)}局面・{len(report["per_game"])}対局')
    for method, label in zip(methods, ('残り牌のみ', '個別推定', '3人同時推定')):
        scores = report[method]
        print(f'{label}: 一致 {scores["matched_tiles_per_hand"]:.3f}枚/人、'
              f'MAE {scores["mae"]:.6f}、共有上限違反 {scores["violation_cases"]}局面')
    print(f'同時推定の増減: {report["joint_gain_per_hand"]:+.3f}枚/人')
    print(f'評価seed: {eval_seed}')
    print(f'評価結果: {out.resolve()}')


def prediction_rows(args):
    model_dir = args.model_dir.resolve()
    config_file = training_settings_path(model_dir)
    config = json.loads((config_file if config_file.exists() else model_dir/'metrics.json').read_text())
    names = json.loads((model_dir/'feature_names.json').read_text())
    validate_feature_names(names)
    root = resolve_csv_root(config['input'], args.input_dir)
    if args.game_id or args.input_dir:
        index = root/'selected_games.csv'
        gid = args.game_id
    else:
        index = model_dir/'gakushu_games.csv'
        gid = None if args.action == 'sannin' else config['test_game_ids'][0]
    with index.open(encoding='utf-8-sig', newline='') as file:
        records = csv.DictReader(file)
        if gid is None and args.action == 'sannin':
            eligible = set(config['test_game_ids'])
            candidates = [r for r in records if r['game_id'] in eligible]
            record = secrets.choice(candidates) if candidates else None
        else:
            record = next((r for r in records if gid is None or r['game_id'] == gid), None)
    if record is None:
        raise ValueError(f'指定した対局が一覧にありません: {gid}')
    gid = record['game_id']
    paths = saved_paths(root, [record])
    # 1局面を観測席ごとに3行へ展開する既存の並びを利用する。
    if args.state_index is None:
        with paths[0][0].open(encoding='utf-8-sig', newline='') as file:
            discards = [i for i, row in enumerate(csv.DictReader(file)) if int(row['action']) == 2]
        if not discards:
            raise ValueError(f'打牌直後の局面がありません: {gid}')
        args.state_index = secrets.choice(discards)
    start = 3 * args.observer
    positions = {gid: {args.state_index: set(range(start, start + 3))}}
    rows = list(iter_csv_rows(paths, positions))
    if len(rows) != 3 or {r['target_player'] for r in rows} != set(range(4)) - {args.observer}:
        raise ValueError('指定した局面の他家3席が揃っていません。--state-index を確認してください')
    return rows, names, gid


def prediction_row(args):
    rows, names, gid = prediction_rows(args)
    return next(row for row in rows if row['target_player'] == args.target), names, gid


def load_predictions(model_dir, values):
    os.environ.setdefault('KERAS_HOME', str(BASE_DIR/'.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('入力値が不正です')
    model = tf.keras.models.load_model(model_dir/'tehai_yosoku.keras', compile=False)
    result = model(x, training=False).numpy()
    if result.shape != (len(x), 34) or not np.isfinite(result).all():
        raise ValueError('予測結果が不正です')
    return result, x


TILE_NAMES = [f'{i}{suit}' for suit in ('萬', '筒', '索') for i in range(1, 10)] + [
    '東', '南', '西', '北', '白', '發', '中',
]


def predict_three(args):
    if (args.model_dir/'sample.json').is_file():
        raise ValueError('任意の局面で3人を予測するには、対局CSVと補助CSVが必要です')
    import numpy as np

    rows, names, gid = prediction_rows(args)
    scores, x = load_predictions(args.model_dir, [numeric_row(row, names) for row in rows])
    import tensorflow as tf
    remaining = remaining_tiles(x[:1], names, tf.keras.ops).numpy()[0].astype(np.int32)
    # 旧モデルの出力も、公開情報で不可能と分かる牌は候補から外す。
    scores = np.clip(scores, 0, remaining)
    allocation = None
    if rows[0]['action'] == 2:
        sizes = np.array([13 - 3 * int(r[f"p{r['target_player']}_meld_count"]) for r in rows])
        allocation = allocate_hands(scores, remaining, sizes)
    action_names = {0: '配牌', 1: 'ツモ', 2: '打牌', 3: '鳴き', 4: 'リーチ', 5: 'ドラ追加'}
    event = action_names.get(rows[0]['action'], str(rows[0]['action']))
    print(f'対局 {gid} ／ 局面 {args.state_index}（{event}） ／ 自分の席 {args.observer}')
    print(f'相手ごとに上位{args.top}牌種を表示します。')
    actual_hands = np.asarray([label_from_csv(row) for row in rows]) if args.show_answer else None
    estimates = allocation if allocation is not None else scores
    if actual_hands is not None:
        print('誤差は上位候補だけでなく、全34牌種の枚数で計算します。')
    for i, row in enumerate(rows):
        print(f"\n相手の席 {row['target_player']}")
        actual = actual_hands[i] if actual_hands is not None else None
        counts = estimates[i]
        candidates = [t for t in np.argsort(-scores[i], kind='stable') if counts[t] > 0][:args.top]
        for t in candidates:
            answer = f' ／ 実際 {actual[t]:.0f}枚' if actual is not None else ''
            count = str(int(counts[t])) if allocation is not None else f'{counts[t]:.3f}'
            print(f'  {TILE_NAMES[t]}  推定 {count}枚{answer}')

        if actual is not None:
            answer_hand = ' '.join(f'{tile}×{int(n)}' for tile, n in zip(TILE_NAMES, actual) if n)
            print(f'  正解の伏せ牌（{int(actual.sum())}枚）  {answer_hand}')
            if allocation is not None:
                estimated_hand = ' '.join(f'{tile}×{n}' for tile, n in zip(TILE_NAMES, counts) if n)
                matches = int(np.minimum(counts, actual).sum())
                print(f'  推定の伏せ牌（{counts.sum()}枚）  {estimated_hand}')
                print(f'  一致枚数 {matches}/{int(actual.sum())}枚')
            hits = sum(actual[t] > 0 for t in candidates)
            if candidates:
                print(f'  上位候補の的中 {hits}/{len(candidates)}牌種')
            error = counts - actual
            print(f'  MAE {np.abs(error).mean():.3f}枚 ／ RMSE {np.sqrt(np.square(error).mean()):.3f}枚')

    if actual_hands is not None:
        error = estimates - actual_hands
        print(f'\n3人合計の評価（この局面）')
        if allocation is not None:
            matches = int(np.minimum(estimates, actual_hands).sum())
            print(f'  一致枚数 {matches}/{int(actual_hands.sum())}枚')
        print(f'  MAE {np.abs(error).mean():.3f}枚 ／ RMSE {np.sqrt(np.square(error).mean()):.3f}枚')


def saved_info(directory):
    config_path=training_settings_path(directory)
    metrics_path=directory/'hyoka'/'metrics.json'
    if not metrics_path.is_file():
        metrics_path=directory/'metrics.json'
    config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    metrics=json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.is_file() else {}
    return config,metrics


def show_info(args):
    config,metrics=saved_info(args.model_dir)
    values={**metrics,**config}
    print(f'モデル: {args.model_dir.name}')
    for key,label in [('tensorflow_version','学習時TensorFlow'),('mode','モード'),('epochs','学習回数'),
                      ('batch_size','バッチサイズ'),('learning_rate','学習率'),('seed','seed'),
                      ('train_rows','訓練行数'),('test_rows','評価行数'),('feature_count','入力列数')]:
        if key in values:
            print(f'{label}: {values[key]}')
    if 'mae' in metrics:
        print('保存済みの評価結果')
        print(format_evaluation(metrics))
    else:
        print('評価結果は未保存です。学内で評価してから再度書き出してください。')


def export_model(args):
    directory=args.model_dir.resolve()
    config,metrics=saved_info(directory)
    row,names,_=prediction_row(args)
    dest=args.output_dir or BASE_DIR/'mochikaeri'
    dest.mkdir(parents=True,exist_ok=True)
    output=dest/f'{directory.name}.zip'
    prefix=f'tenhou_data/models/{directory.name}/'
    settings_keys=('tensorflow_version','mode','epochs','batch_size','learning_rate','seed',
                   'train_rows','test_rows','feature_count','max_train','max_test','max_games','requested_players')
    error_keys=('mae','mean_baseline_mae','mse','rmse','mae_improvement','mae_reduction_percent','test_rows',
                'count_accuracy','presence_precision','presence_recall','presence_f1','eval_seed')
    values={**metrics,**config}
    portable_settings={k:values[k] for k in settings_keys if k in values}
    portable_metrics={k:metrics[k] for k in error_keys if k in metrics}
    sample={'x':numeric_row(row,names),'state_index':args.state_index,
            'observer':args.observer,'target_player':args.target}
    instructions=[
        'ZIPを展開し、展開先でターミナルを開きます。',
        '',
        '保存済みの評価結果と学習設定を表示（Pythonのみ必要）',
        f'python moderu_jikko.py --name {directory.name} --action info',
        '',
        'サンプル予測（初回のみライブラリをインストール）',
        'python -m pip install -r requirements.txt',
        f'python moderu_jikko.py --name {directory.name} --action yosoku',
        '',
        '学習や対局全体の再評価には変換済みデータが必要です。',
        '付属サンプルにはモデルの入力値だけを収録しています。',
    ]
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ['moderu_jikko.py','aite_tehai_gakushu.py','paifu_csv_henkan.py','requirements.txt']:
            bundle.write(BASE_DIR/name,name)
        bundle.write(directory/'tehai_yosoku.keras',prefix+'tehai_yosoku.keras')
        for name,value in [('feature_names.json',names),('gakushu_settei.json',portable_settings),
                           ('metrics.json',portable_metrics),('sample.json',sample)]:
            bundle.writestr(prefix+name,json.dumps(value,ensure_ascii=False,indent=2))
        bundle.writestr('jitaku_gaido.txt','\n'.join(instructions))
    print(f'持ち帰り用ZIP: {output.resolve()} ({output.stat().st_size/1024**2:.2f} MiB)')


def predict(args):
    model_dir=args.model_dir.resolve()
    portable=(model_dir/'sample.json').is_file()
    if portable:
        if args.show_answer or args.game_id:
            raise ValueError('持ち帰り用には固定の予測サンプルだけが含まれます')
        sample=json.loads((model_dir/'sample.json').read_text(encoding='utf-8'))
        names=json.loads((model_dir/'feature_names.json').read_text(encoding='utf-8'))
        validate_feature_names(names)
        if len(sample['x'])!=len(names):
            raise ValueError('サンプルの入力列が一致しません')
        values=sample['x']
    else:
        row,names,gid=prediction_row(args)
        values=numeric_row(row,names)
    import numpy as np
    predictions, _ = load_predictions(model_dir, [values])
    prediction = predictions[0]
    if portable:
        print(f"付属サンプル: 局面{sample['state_index']} / 自分の席{sample['observer']} / 相手の席{sample['target_player']}")
    else:
        print(f'対局: {gid}')
        print(f'局面: {args.state_index} / 自分の席: {args.observer} / 相手の席: {args.target}')
    print(f'予測枚数が多い牌（上位{args.top}種）')
    if args.show_answer:
        actual=label_from_csv(row)
    for i in np.argsort(-prediction)[:args.top]:
        extra=f' / 実際: {actual[i]:.0f}枚' if args.show_answer else ''
        print(f'{TILE_NAMES[i]:>3}: {prediction[i]:.3f}枚{extra}')
    print(f'予測枚数の合計: {prediction.sum():.3f}枚（合法な手牌枚数への制約なし）')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',help='実行するモデル名。省略時は冒頭のMODEL_NAME')
    parser.add_argument('--compare',help='同じ評価行で比較するモデル名')
    parser.add_argument('--model-dir',type=Path,help='別の場所にあるモデルを指定')
    parser.add_argument('--action',choices=['hyoka','yosoku','info','export','douji','sannin'],default=ACTION)
    parser.add_argument('--output-dir',type=Path,help='評価結果の保存先。既定はモデル内のhyoka')
    parser.add_argument('--eval-seed',type=int,help='評価抽出のseed。省略時は毎回生成')
    parser.add_argument('--test-games',type=int,help='未学習対局から抽出する対局数')
    parser.add_argument('--test-rows',type=int,help='通常評価で抽出する行数')
    parser.add_argument('--cases',type=int,default=256,help='同時推定の評価局面数')
    parser.add_argument('--state-index',type=int,help='局面番号。3人予測では省略時に打牌直後から抽出')
    parser.add_argument('--observer',type=int,choices=range(4),default=0)
    parser.add_argument('--target',type=int,choices=range(4),default=1)
    parser.add_argument('--game-id',help='予測する対局ID')
    parser.add_argument('--input-dir',type=Path,help='評価・予測に使う変換先フォルダ')
    parser.add_argument('--top',type=int,choices=range(1,35),default=10,metavar='1..34')
    parser.add_argument('--show-answer', action=argparse.BooleanOptionalAction,
                        help='正解と誤差を表示。3人予測では既定で表示')
    args=parser.parse_args()
    name=args.name or MODEL_NAME
    if Path(name).name!=name or name in ('.','..') or '\\' in name:
        parser.error('モデル名にはフォルダを含めないでください')
    if args.name and args.model_dir:parser.error('--nameと--model-dirはどちらか一方を指定してください')
    if args.compare and (Path(args.compare).name!=args.compare or args.compare in ('.','..') or '\\' in args.compare):
        parser.error('比較モデル名にはフォルダを含めないでください')
    args.model_dir=args.model_dir or DATA_DIR/'models'/name
    if not (args.model_dir/'tehai_yosoku.keras').is_file():
        parser.error(f'モデルがありません: {args.model_dir}')
    if args.test_rows is not None and args.test_rows<1:parser.error('--test-rowsは1以上です')
    if args.test_games is not None and args.test_games<1:parser.error('--test-gamesは1以上です')
    if args.state_index is not None and args.state_index < 0:
        parser.error('state-indexは0以上です')
    if args.action == 'sannin' and (args.model_dir/'sample.json').is_file():
        args.action = 'yosoku'
    if args.show_answer is None:
        args.show_answer = args.action == 'sannin'
    if args.state_index is None and args.action != 'sannin':
        args.state_index = 30
    if args.action=='info' or (args.action=='hyoka' and (args.model_dir/'sample.json').is_file()):
        show_info(args)
    elif args.action=='hyoka':
        evaluate(args)
    elif args.action=='douji':
        if args.cases<1:parser.error('--casesは1以上です')
        evaluate_joint(args)
    elif args.action=='export':
        if args.observer==args.target:parser.error('observerとtargetは異なる席を指定してください')
        export_model(args)
    elif args.action=='sannin':
        predict_three(args)
    else:
        if args.observer==args.target:parser.error('observerとtargetは異なる席を指定してください')
        predict(args)


if __name__=='__main__':
    main()
