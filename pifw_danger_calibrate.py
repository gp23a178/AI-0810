"""Fit aggregate ron risk and residual error from saved Tenhou games."""
import argparse
from collections import defaultdict
import csv
from itertools import zip_longest
import json
import os
import random
import tempfile

from aite_tehai_gakushu import DATA_DIR, select_csvs
from paifu_csv_henkan import ACTION_INITIAL, ACTION_DRAW, ACTION_DISCARD, ACTION_MELD
from pifw_danger import CALIBRATION_VERSION, DangerCalibration, allowed_ron, contexts_from_public, group_key, tile_bucket
from pifw_live import live_result
from pifw_moderu_seisei import RED_BASE, tenhou_tiles
from pifw_tehai_yosoku import DEFAULT_MODEL, SCHEMA, HandPredictor, encode_observation, tile_counts

# The mark and length use the same format as PIFW's public meld packet.
MELD_FLAGS = {1: (6, 3), 2: (5, 3), 3: (4, 4), 4: (8, 4), 5: (7, 4)}


def rows_for_game(files, limit, seed):
    path, meta, info = files
    candidates = []
    with path.open(encoding='utf-8-sig', newline='') as f, meta.open(encoding='utf-8-sig', newline='') as m:
        for i, (row, aux) in enumerate(zip_longest(csv.DictReader(f), csv.DictReader(m))):
            if row is None or aux is None or int(aux['state_index']) != i:
                raise ValueError(f'CSV/補助CSVの対応が不正です: {path}')
            action, actor = int(row['action']), int(row['actor'])
            riichi = [int(row[f'p{p}_is_reach']) for p in range(4)]
            if action == ACTION_INITIAL:
                passed = [set() for _ in range(4)]
                riichi_passed = [set() for _ in range(4)]
            elif action == ACTION_DISCARD:
                tile = abs(int(row['target_tile']))
                tile = RED_BASE.get(tile, tile)
                for seat in range(4):
                    if seat != actor:
                        passed[seat].add(tile)
                        if riichi[seat]:
                            riichi_passed[seat].add(tile)
            elif action in (ACTION_DRAW, ACTION_MELD):
                passed[actor].clear()
            if action != ACTION_DRAW:
                continue

            hands = [tenhou_tiles(lambda tile: row[f'p{seat}_hand_{tile}']) for seat in range(4)]
            rivers, histories, melds, flags = [], [], [], []
            for seat in range(4):
                river, history, meld_tiles, meld_flags = [], [], [], []
                for n in range(1, 25):
                    tile = abs(int(row[f'p{seat}_disc_{n}']))
                    if tile:
                        tile = RED_BASE.get(tile, tile)
                        history.append(tile)
                        if not int(aux[f'p{seat}_disc_called_{n}']):
                            river.append(tile)
                for slot in range(1, 5):
                    prefix = f'p{seat}_meld{slot}'
                    kind = int(aux[f'{prefix}_type'])
                    if kind:
                        tag, length = MELD_FLAGS[kind]
                        for n in range(1, length+1):
                            tile = int(aux[f'{prefix}_tile{n}'])
                            meld_tiles.append(RED_BASE.get(tile, tile))
                            meld_flags.append(tag if n == 1 else 0)
                rivers.append(river)
                histories.append(history)
                melds.append(meld_tiles)
                flags.append(meld_flags)
            obs = dict(schema=SCHEMA, observer=actor, own_hand=hands[actor],
                       concealed_sizes=[len(hand) for hand in hands], rivers=rivers, meld_tiles=melds,
                       meld_counts=[int(aux[f'p{p}_meld_count']) for p in range(4)], riichi=riichi,
                       dora_indicators=[RED_BASE.get(tile, tile) for n in range(1, 6)
                                        if (tile := int(row[f'dora_{n}']))])
            bits = sum(riichi[p] << p for p in range(4))
            bits |= sum(int(aux[f'p{p}_ippatsu']) << (4+p) for p in range(4))
            bits |= sum(int(aux[f'p{p}_double_riichi']) << (8+p) for p in range(4))
            public = dict(meld_flags=flags, discard_history=histories,
                          passed=[sorted(a | b) for a, b in zip(passed, riichi_passed)],
                          riichi_bits=bits, dealer=int(aux['oya']),
                          round_wind=int(aux['game_wind'])-1, wall_left=int(row['remain_tsumo']))
            candidates.append((dict(observation=obs, public_context=public), [tile_counts(hand) for hand in hands]))
    if not candidates:
        raise ValueError(f'ツモ局面がありません: {path}')
    return random.Random(f"{seed}:{info['game_id']}").sample(candidates, min(len(candidates), limit))


def ground_truth(request, hands):
    obs, public = request['observation'], request['public_context']
    contexts = contexts_from_public(obs, public)
    candidates = frozenset(obs['own_hand'])
    discards = sum(len(river) for river in public['discard_history'])
    for offset in range(1, 4):
        seat = (obs['observer'] + offset) % 4
        context = contexts[seat]
        key = group_key(discards, context.riichi, obs['meld_counts'][seat])
        if sum(hands[seat]) + 3 * len(context.melds) != 13:
            continue
        possible = candidates - context.discarded
        yield offset-1, seat, key, possible, allowed_ron(hands[seat], context, possible)


def fit(predictor, games):
    import numpy as np

    residual = defaultdict(lambda: [0.0, 0])
    by_tile = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for positions in games:
        rows = [row for request, _ in positions
                for row in encode_observation(request['observation'], predictor.profile, predictor.schema)[0]]
        predicted = predictor.predict_features(rows).reshape(len(positions), 3, 34)
        for (request, actual), expected in zip(positions, predicted):
            for index, target, key, possible, hits in ground_truth(request, actual):
                mse = np.square(expected[index] - actual[target]).mean()
                for group in ('all', key):
                    residual[group][0] += float(mse)
                    residual[group][1] += 1
                    for tile in possible:
                        counter = by_tile[group][tile_bucket(tile)]
                        counter[0] += 1
                        counter[1] += tile in hits
    return {'version': CALIBRATION_VERSION,
            'residual_rmse': {key: (total / n) ** 0.5 for key, (total, n) in residual.items()},
            'by_tile': by_tile}


def evaluate(predictor, calibration, games):
    baseline = corrected = observed = predicted = 0.0
    total = positives = 0
    for positions in games:
        for request, truth in positions:
            result = live_result(predictor, request, offline_calibration=calibration)
            for i, _, key, possible, hits in ground_truth(request, truth):
                predicted_risk = {r['tile']: r['probability'] for r in result['opponents'][i]['ron_risks']}
                for tile in possible:
                    actual = int(tile in hits)
                    base = calibration.prior(key, tile)
                    p = predicted_risk[tile]
                    baseline += (base - actual)**2
                    corrected += (p - actual)**2
                    predicted += p
                    positives += actual
                    total += 1
    if not total:
        raise ValueError('危険度の評価候補がありません')
    return dict(tile_checks=total, ron_possible=positives, brier_prior=baseline/total,
                brier_blended=corrected/total, mean_risk=predicted/total,
                observed_rate=positives/total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-positions-per-game', type=int, default=160)
    parser.add_argument('--test-positions-per-game', type=int, default=24)
    args = parser.parse_args()
    if min(args.fit_positions_per_game, args.test_positions_per_game) < 1:
        parser.error('局面数は1以上です')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    predictor = HandPredictor(DEFAULT_MODEL)
    files, _, _ = select_csvs(DATA_DIR / 'converted/after', 'L', predictor.manifest['seed'])
    test_ids = sorted(predictor.manifest['test_game_ids'])
    if len(test_ids) < 4 or predictor.schema != SCHEMA:
        raise ValueError('手牌モデルの独立した評価対局が必要です')
    lookup = {game[2]['game_id']: game for game in files}
    random.Random(49).shuffle(test_ids)
    first, second = test_ids[:len(test_ids)//2], test_ids[len(test_ids)//2:]
    print(f'危険度の補正 {len(first)}対局、検証 {len(second)}対局', flush=True)
    a = [rows_for_game(lookup[key], args.fit_positions_per_game, 49) for key in first]
    b = [rows_for_game(lookup[key], args.test_positions_per_game, 49) for key in second]
    data = fit(predictor, a)
    data.update(model_name=predictor.manifest['model_name'], calibration_games=sorted(first),
                evaluation_games=sorted(second), fit_positions=sum(map(len, a)), evaluation_positions=sum(map(len, b)))
    target = DEFAULT_MODEL / 'danger_calibration.json'
    with tempfile.TemporaryDirectory(prefix='.danger-', dir=target.parent) as d:
        temporary = target.__class__(d) / target.name
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        calibration = DangerCalibration(temporary)
        data['evaluation'] = evaluate(predictor, calibration, b)
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(target)
    print(json.dumps(data['evaluation'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
