"""PIFWの公開局面を入力し、他家3人の伏せ牌を推定する。"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from moderu_jikko import allocate_hands

PRE_DISCARD_SCHEMA = 'pifw-mahjong-observation-v1'
SCHEMA = 'pifw-mahjong-observation-v2'
TILE_IDS = range(1, 35)
FEATURE_PROFILES = ('counts', 'recent')
DEFAULT_MODEL = Path(__file__).resolve().parent / 'tenhou_data/models/pifw_02'


def feature_names(profile, schema=SCHEMA):
    if schema not in (PRE_DISCARD_SCHEMA, SCHEMA):
        raise ValueError(f'未対応の公開局面スキーマ: {schema}')
    if profile not in FEATURE_PROFILES:
        raise ValueError(f'未対応の特徴量セット: {profile}')
    names = [f'{group}_{tile}' for group in ('own', 'dora') for tile in TILE_IDS]
    for seat in range(4):
        names.extend(f'seat{seat}_{group}_{tile}' for group in ('river', 'meld') for tile in TILE_IDS)
        names.extend((f'seat{seat}_meld_count', f'seat{seat}_riichi'))
        if schema == SCHEMA:
            names.append(f'seat{seat}_concealed_size')
        if profile == 'recent':
            names.extend(f'seat{seat}_last{back}_{tile}' for back in range(1, 4) for tile in TILE_IDS)
    names.extend(f'target_relative_{seat}' for seat in range(1, 4))
    names.extend(f'remaining_{tile}' for tile in TILE_IDS)
    return names + ['target_concealed_size']


def tile_counts(tiles):
    if not isinstance(tiles, list):
        raise ValueError('牌列は牌IDの配列で指定してください')
    counts = [0] * 34
    for tile in tiles:
        if type(tile) is not int or tile not in TILE_IDS:
            raise ValueError(f'牌IDは1〜34です: {tile!r}')
        counts[tile - 1] += 1
    return counts


def encode_observation(observation, profile, feature_schema=None):
    """他家の手牌の牌種や山を受け取らず、入力と配分制約を作る。"""
    fields = {'schema', 'observer', 'own_hand', 'rivers', 'meld_tiles', 'meld_counts', 'riichi', 'dora_indicators'}
    schema = observation.get('schema') if isinstance(observation, dict) else None
    if schema not in (PRE_DISCARD_SCHEMA, SCHEMA):
        raise ValueError('公開局面JSONのスキーマが一致しません')
    if schema == SCHEMA:
        fields.add('concealed_sizes')
    if set(observation) != fields:
        raise ValueError('公開局面JSONの項目が一致しません')
    feature_schema = feature_schema or schema
    if profile not in FEATURE_PROFILES:
        raise ValueError(f'未対応の特徴量セット: {profile}')
    observer = observation['observer']
    if type(observer) is not int or observer not in range(4):
        raise ValueError('観測席は0〜3です')
    rivers, melds = observation['rivers'], observation['meld_tiles']
    meld_counts, riichi = observation['meld_counts'], observation['riichi']
    if any(not isinstance(values, list) or len(values) != 4 for values in (rivers, melds, meld_counts, riichi)):
        raise ValueError('河・副露・立直は4席分を指定してください')
    if any(type(count) is not int or count not in range(5) for count in meld_counts):
        raise ValueError('副露数は0〜4です')
    if any(type(flag) is not int or flag not in (0, 1) for flag in riichi):
        raise ValueError('立直は0か1です')
    own = tile_counts(observation['own_hand'])
    dora = tile_counts(observation['dora_indicators'])
    pre_discard_sizes = [13 - 3 * count + int(seat == observer)
                         for seat, count in enumerate(meld_counts)]
    concealed = observation['concealed_sizes'] if schema == SCHEMA else pre_discard_sizes
    if (not isinstance(concealed, list) or len(concealed) != 4
            or any(type(size) is not int for size in concealed)
            or concealed[observer] != sum(own)):
        raise ValueError('伏せ牌の枚数が自分の手牌または座席数と一致しません')
    effective = [size + 3 * melds for size, melds in zip(concealed, meld_counts)]
    if any(size not in (13, 14) for size in effective) or effective.count(14) > 1:
        raise ValueError('副露を面子に換算した手牌は各席13枚または14枚です')
    if feature_schema == PRE_DISCARD_SCHEMA and concealed != pre_discard_sizes:
        raise ValueError('旧モデルは自分のツモ後または鳴き後の打牌前だけ予測できます')
    if not 1 <= sum(dora) <= 5:
        raise ValueError('公開ドラ表示牌は1〜5枚です')

    river_counts = [tile_counts(river) for river in rivers]
    open_counts = [tile_counts(meld) for meld in melds]
    if any(not 3 * count <= sum(tiles) <= 4 * count for tiles, count in zip(open_counts, meld_counts)):
        raise ValueError('副露数と副露の現物枚数が一致しません')
    remaining = [4 - sum(counts) for counts in zip(own, dora, *river_counts, *open_counts)]
    if min(remaining) < 0:
        raise ValueError('同種の可視牌が4枚を超えています。鳴かれた牌の二重計上を確認してください')

    common = own + dora
    for offset in range(4):
        seat = (observer + offset) % 4
        common += river_counts[seat] + open_counts[seat] + [meld_counts[seat], riichi[seat]]
        if feature_schema == SCHEMA:
            common.append(concealed[seat])
        if profile == 'recent':
            recent = rivers[seat][-3:][::-1]
            for index in range(3):
                tile = recent[index] if index < len(recent) else 0
                common.extend(int(tile == kind) for kind in TILE_IDS)
    seats = [(observer + offset) % 4 for offset in range(1, 4)]
    sizes = [concealed[seat] for seat in seats]
    if sum(sizes) > sum(remaining):
        raise ValueError('見えていない牌が3人分の伏せ牌を満たしません')
    rows = [common + [int(index == target) for index in range(3)] + remaining + [size]
            for target, size in enumerate(sizes)]
    return rows, seats, remaining, sizes


def assign_hands(expected, remaining, sizes, strategy, seed=None):
    """予測とは独立に、整数の仮手牌と未割当の牌を作る。"""
    import numpy as np
    import random

    if strategy in ('joint', 'proportional'):
        scores = expected
        if strategy == 'proportional':
            scores = np.asarray(sizes)[:, None] * np.asarray(remaining)[None, :] / sum(remaining)
        counts = allocate_hands(scores, remaining, sizes)
    elif strategy == 'uniform':
        pool = [tile for tile, count in enumerate(remaining) for _ in range(count)]
        random.Random(seed).shuffle(pool)
        counts = np.zeros((3, 34), dtype=np.int32)
        start = 0
        for i, size in enumerate(sizes):
            counts[i] = np.bincount(pool[start:start + size], minlength=34)
            start += size
    else:
        raise ValueError(f'未対応の配分方法: {strategy}')
    unassigned = np.asarray(remaining, dtype=np.int32) - counts.sum(axis=0)
    if (unassigned < 0).any() or not np.array_equal(counts.sum(axis=1), sizes):
        raise RuntimeError('配分結果が枚数制約を満たしません')
    return counts, unassigned


class HandPredictor:
    """SavedModelを一度読み込み、複数の対局局面で再利用する。"""

    def __init__(self, model_dir):
        import tensorflow as tf

        self.manifest = json.loads((model_dir / 'model.json').read_text(encoding='utf-8'))
        self.profile = self.manifest['feature_profile']
        self.schema = self.manifest['observation_schema']
        if self.manifest['feature_names'] != feature_names(self.profile, self.schema):
            raise ValueError('モデルの入力形式と特徴量定義が一致しません')
        self.model = tf.saved_model.load(str(model_dir / 'saved_model'))
        self.serving = self.model.signatures['serving_default']

    def predict_features(self, rows):
        import numpy as np

        return self.serving(features=np.asarray(rows, dtype=np.float32))['expected_counts'].numpy()

    def predict(self, observation, strategy='joint', seed=None):
        import numpy as np

        rows, seats, remaining, sizes = encode_observation(observation, self.profile, self.schema)
        expected = self.predict_features(rows)
        if (expected.shape != (3, 34) or not np.isfinite(expected).all()
                or (expected < 0).any() or (expected > np.asarray(remaining) + 1e-6).any()):
            raise ValueError('モデル出力の形または牌数が不正です')

        result = {'observer': observation['observer'], 'tile_ids': list(TILE_IDS), 'remaining': remaining,
                  'opponents': [{'seat': seat, 'concealed_size': size, 'expected_counts': row.tolist()}
                                for seat, size, row in zip(seats, sizes, expected)]}
        if strategy != 'none':
            hands, unknown = assign_hands(expected, remaining, sizes, strategy, seed)
            for player, hand in zip(result['opponents'], hands):
                player['assigned_counts'] = hand.tolist()
            result.update(allocation=strategy, unassigned_counts=unknown.tolist())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--input', type=Path, help='公開局面JSON。省略時は標準入力')
    parser.add_argument('--stream', action='store_true', help='1行1局面のJSONを受け取り、順に予測')
    parser.add_argument('--allocation', choices=('joint', 'proportional', 'uniform', 'none'), default='joint')
    parser.add_argument('--seed', type=int, help='無作為配分を再現するseed')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    try:
        predictor = HandPredictor(args.model_dir)
        source = nullcontext(sys.stdin) if args.input is None else args.input.open(encoding='utf-8')
        with source as stream:
            observations = (json.loads(line) for line in stream if line.strip()) if args.stream else (json.load(stream),)
            for observation in observations:
                result = predictor.predict(observation, args.allocation, args.seed)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'予測できません: {exc}\n')


if __name__ == '__main__':
    main()
