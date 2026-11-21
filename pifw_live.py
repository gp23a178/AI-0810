"""Unity対局画面用。公開局面だけを受け取り、1行ずつ推論する。"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
import sys

from pifw_danger import DangerCalibration, allowed_ron, contexts_from_public, group_key, shape_waits
from pifw_tehai_yosoku import DEFAULT_MODEL, HandPredictor, assign_hands

DISPLAY_THRESHOLD = 0.35
SEARCH_SAMPLES = 16
TERMINALS_AND_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


@lru_cache(maxsize=65536)
def complete_groups(counts, suited):
    """1色または字牌を、刻子と順子だけに分けられるか判定する。"""
    first = next((i for i, n in enumerate(counts) if n), None)
    if first is None:
        return True
    work = list(counts)
    if work[first] >= 3:
        work[first] -= 3
        if complete_groups(tuple(work), suited):
            return True
        work[first] += 3
    if suited and first < 7 and work[first + 1] and work[first + 2]:
        for i in range(first, first + 3):
            work[i] -= 1
        return complete_groups(tuple(work), suited)
    return False


def complete_hand(counts, melds):
    if melds == 0:
        if sum(n == 2 for n in counts) == 7:
            return True
        if all(counts[tile] for tile in TERMINALS_AND_HONORS) and sum(
            counts[tile] for tile in TERMINALS_AND_HONORS
        ) == 14:
            return True
    groups = [list(counts[:9]), list(counts[9:18]), list(counts[18:27]), list(counts[27:])]
    pair_group = [i for i, group in enumerate(groups) if sum(group) % 3 == 2]
    if len(pair_group) != 1:
        return False
    pair_index = pair_group[0]
    for i, group in enumerate(groups):
        if i != pair_index and (sum(group) % 3 or not complete_groups(tuple(group), i < 3)):
            return False
    group = groups[pair_index]
    for i, n in enumerate(group):
        if n >= 2:
            group[i] -= 2
            valid = complete_groups(tuple(group), pair_index < 3)
            group[i] += 2
            if valid:
                return True
    return False


def tenpai_shape(counts, melds, upper):
    """和了形まであと1枚の形。役やフリテン、待ちの残枚数は判定しない。"""
    hand = list(counts)
    if sum(hand) + 3 * melds == 14:
        # ツモ後なら、どれか1枚を切って聴牌になる形かを見る。
        for tile, copies in enumerate(counts):
            if copies:
                hand[tile] -= 1
                if tenpai_shape(hand, melds, upper):
                    return True
                hand[tile] += 1
        return False
    for tile, cap in enumerate(upper):
        if hand[tile] < cap:
            hand[tile] += 1
            valid = complete_hand(hand, melds)
            hand[tile] -= 1
            if valid:
                return True
    return False


def live_result(predictor, request, calibration=None):
    import numpy as np

    enhanced = isinstance(request, dict) and set(request) == {'observation', 'public_context'}
    observation = request['observation'] if enhanced else request
    prediction = predictor.predict(observation)
    opponents = prediction['opponents']
    remaining = prediction['remaining']
    sizes = [p['concealed_size'] for p in opponents]
    expected = np.asarray([p['expected_counts'] for p in opponents])
    public_melds = observation['meld_counts']
    riichi = observation['riichi']
    seed_bytes = json.dumps(request, sort_keys=True).encode()
    seed = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), 'big')
    rng = np.random.default_rng(seed)
    tenpai = [0, 0, 0]
    upper = [[4 - observation['meld_tiles'][p['seat']].count(tile) for tile in range(1, 35)]
             for p in opponents]
    if enhanced:
        if calibration is None:
            raise ValueError('危険度の補正ファイルがありません')
        contexts = contexts_from_public(observation, request['public_context'])
        count = sum(len(river) for river in request['public_context']['discard_history'])
        groups = [group_key(count, riichi[p['seat']], public_melds[p['seat']]) for p in opponents]
        sigma = np.asarray([[calibration.error(key)] for key in groups])
        candidates = frozenset(observation['own_hand'])
        hits = [Counter() for _ in opponents]
        causes = [dict() for _ in opponents]
        trials = [0] * len(opponents)
    else:
        sigma = 0.40

    for _ in range(SEARCH_SAMPLES):
        scores = expected + rng.normal(0, sigma, size=expected.shape)
        hands, _ = assign_hands(scores, remaining, sizes, 'joint')
        for index, (player, counts) in enumerate(zip(opponents, hands.tolist())):
            seat = player['seat']
            if enhanced and sizes[index] + 3 * public_melds[seat] == 13:
                waits = shape_waits(tuple(counts), tuple(upper[index]))
                # A live riichi declaration is public proof of a tenpai hand.
                # Rejected non-tenpai hypotheses carry no weight in the riichi denominator.
                if not riichi[seat] or waits:
                    trials[index] += 1
                    for tile, yaku in allowed_ron(counts, contexts[seat], candidates).items():
                        hits[index][tile] += 1
                        causes[index].setdefault(tile, Counter()).update(yaku)
                tenpai[index] += bool(waits)
            elif not riichi[seat] and tenpai_shape(counts, public_melds[seat], upper[index]):
                tenpai[index] += 1
    for index, player in enumerate(opponents):
        seat = player['seat']
        player['display_counts'] = [
            assigned if value >= DISPLAY_THRESHOLD else 0
            for assigned, value in zip(player['assigned_counts'], player['expected_counts'])
        ]
        player['riichi'] = bool(riichi[seat])
        player['tenpai_index'] = 100 if riichi[seat] else round(100 * tenpai[index] / SEARCH_SAMPLES)
        if not enhanced:
            continue
        player['risk_available'] = sizes[index] + 3 * public_melds[seat] == 13
        if not player['risk_available']:
            continue
        group = groups[index]
        player['hand_error_rmse'] = calibration.error(group)
        player['risk_trials'] = trials[index]
        player['ron_risks'] = []
        for tile in sorted(candidates):
            probability = (0.0 if tile in contexts[seat].discarded
                           else calibration.probability(group, tile, hits[index][tile], trials[index]))
            player['ron_risks'].append(dict(
                tile=tile, probability=probability, hypothesis_hits=hits[index][tile],
                yaku=[name for name, _ in causes[index].get(tile, Counter()).most_common(3)],
            ))
        player['ron_risks'].sort(key=lambda r: (-r['probability'], r['tile']))
    return dict(prediction, display_threshold=DISPLAY_THRESHOLD, search_samples=SEARCH_SAMPLES)


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    predictor = HandPredictor(DEFAULT_MODEL)
    path = DEFAULT_MODEL / 'danger_calibration.json'
    calibration = DangerCalibration(path) if path.is_file() else None
    for line in sys.stdin:
        if line.strip():
            try:
                result = live_result(predictor, json.loads(line), calibration)
            except (ValueError, TypeError, KeyError) as error:
                result = {'error': str(error)}
            print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
