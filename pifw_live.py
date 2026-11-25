"""Stream hand and danger estimates derived from public observations."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from statistics import median
import sys

from mahjong.shanten import Shanten

from pifw_danger import allowed_ron, contexts_from_public, group_key, shape_waits
from pifw_tehai_yosoku import (
    DEFAULT_MODEL_NAME, DISPLAY_THRESHOLD, HandPredictor, assign_hands, displayed_tile_counts, model_path,
)

SEARCH_SAMPLES = 16
TERMINALS_AND_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
CLOSED_ROUTES = ('七対子', '国士無双')


@lru_cache(maxsize=32768)
def shanten_for(hand, closed):
    return Shanten.calculate_shanten(hand, use_chiitoitsu=closed, use_kokushi=closed)


def hand_routes(hand, meld_tiles, melds, shanten, context):
    if not melds and shanten <= 3:
        for distance, label in zip((Shanten.calculate_shanten_for_chiitoitsu_hand(hand),
                                    Shanten.calculate_shanten_for_kokushi_hand(hand)), CLOSED_ROUTES):
            if distance == shanten:
                yield label
    full = list(hand)
    for tile in meld_tiles:
        full[tile - 1] += 1
    suits = [sum(full[start:start + 9]) for start in (0, 9, 18)]
    suit = max(range(3), key=suits.__getitem__)
    outside = sum(suits) - suits[suit]
    if suits[suit] >= 7 and outside <= 2 and all(
            tile >= 28 or (tile - 1) // 9 == suit for tile in meld_tiles):
        name = '混一色' if sum(full[27:]) >= 2 else '清一色'
        yield name + '（' + '萬筒索'[suit] + '）'
    if shanten <= 3 and sum(full[tile] for tile in TERMINALS_AND_HONORS) <= 3 and all(
            tile - 1 not in TERMINALS_AND_HONORS for tile in meld_tiles):
        yield '断么九'
    value_honors = {31, 32, 33}
    if context is not None:
        value_honors.update((context.player_wind, context.round_wind))
    if any(full[tile] >= 3 for tile in value_honors):
        yield '役牌'


def effective_copies(hand, pool, closed, shanten):
    counts = list(hand)
    copies = 0
    for tile, available in enumerate(pool):
        if available and counts[tile] < 4:
            counts[tile] += 1
            if shanten_for(tuple(counts), closed) < shanten:
                copies += available
            counts[tile] -= 1
    return copies


def sample_hands(expected, remaining, sizes, rng):
    # Draw physical copies without replacement. Randomize seat order to avoid a fixed first choice.
    import numpy as np

    remaining = np.asarray(remaining)
    pool = np.repeat(np.arange(34), remaining)
    weights = np.maximum(expected / np.maximum(remaining, 1), 0.03)
    hands = np.zeros((3, 34), dtype=int)
    for seat in rng.permutation(3):
        relative = weights[seat, pool]
        picked = rng.choice(len(pool), size=sizes[seat], replace=False, p=relative / relative.sum())
        hands[seat] = np.bincount(pool[picked], minlength=34)
        pool = np.delete(pool, picked)
    return hands, np.bincount(pool, minlength=34)


def live_result(predictor, request, *, offline_calibration=None):
    import numpy as np

    enhanced = isinstance(request, dict) and set(request) in (
        {'observation', 'public_context'}, {'observation', 'public_context', 'round_id'})
    observation = request['observation'] if enhanced else request
    prediction = predictor.predict(observation)
    opponents = prediction['opponents']
    remaining = prediction['remaining']
    sizes = [p['concealed_size'] for p in opponents]
    expected = np.asarray([p['expected_counts'] for p in opponents])
    melds, riichi = observation['meld_counts'], observation['riichi']
    histories = request['public_context']['discard_history'] if enhanced else observation['rivers']
    contexts = contexts_from_public(observation, request['public_context']) if enhanced else (None,) * 4
    opening = not any(histories) and not any(melds) and not any(riichi)
    wall_left = request['public_context']['wall_left'] if enhanced else None
    turns_left = (wall_left + 3) // 4 if wall_left is not None else None
    seed = int.from_bytes(hashlib.blake2b(json.dumps(observation, sort_keys=True).encode(),
                                         digest_size=8).digest(), 'big')
    rng = np.random.default_rng(seed)
    groups = [group_key(sum(map(len, histories)), riichi[p['seat']], melds[p['seat']]) for p in opponents]
    # Historical residuals are used only by the offline calibration experiment.
    sigma = (None if offline_calibration is None
             else np.asarray([[offline_calibration.error(key)] for key in groups]))
    frequencies = np.zeros((3, 34, 5), dtype=int)
    count_indicators = np.eye(5, dtype=int)
    distances = [[] for _ in opponents]
    estimates = [[] for _ in opponents]
    strengths = [[] for _ in opponents]
    routes = [Counter() for _ in opponents]
    hits = [Counter() for _ in opponents]
    causes = [dict() for _ in opponents]
    trials = [0] * len(opponents)
    candidates = frozenset(observation['own_hand'])
    for _ in range(SEARCH_SAMPLES):
        if sigma is None:
            hands, unknown = sample_hands(expected, remaining, sizes, rng)
        else:
            scores = expected + rng.normal(0, sigma, size=expected.shape)
            hands, unknown = assign_hands(scores, remaining, sizes, 'joint')
        pool = unknown.tolist()
        pool_size = sum(pool)
        frequencies += count_indicators[hands]
        for index, (player, counts) in enumerate(zip(opponents, hands.tolist())):
            seat = player['seat']
            hand = tuple(counts)
            closed = melds[seat] == 0
            distance = shanten_for(hand, closed)
            distances[index].append(distance)
            ready_for_ron = sizes[index] + 3 * melds[seat] == 13
            context = contexts[seat]
            # Only need to test draws for reasonably close hands.
            outs = (effective_copies(hand, pool, closed, distance)
                    if ready_for_ron and 0 <= distance <= 2 and not opening else 0)
            turns = (distance + 1) * (pool_size + 1) / (outs + 1) if outs else None
            if turns is not None:
                estimates[index].append(turns)
            if distance <= 0:
                strength = 58 + min(20, 2 * outs) if outs or not ready_for_ron else 24
            else:
                strength = {1: 38, 2: 23, 3: 10}.get(distance, 4)
            if turns is not None and turns_left is not None:
                strength += 4 if turns <= turns_left else -4
            strengths[index].append(strength)
            if not opening:
                routes[index].update(hand_routes(hand, observation['meld_tiles'][seat],
                                                 melds[seat], distance, context))
            if context is not None and ready_for_ron and (
                    not riichi[seat] or shape_waits(hand, context.upper)):
                trials[index] += 1
                if distance == 0:
                    ron = allowed_ron(hand, context, candidates)
                    for tile, yaku in ron.items():
                        hits[index][tile] += 1
                        causes[index].setdefault(tile, Counter()).update(yaku)

    for index, player in enumerate(opponents):
        seat = player['seat']
        player['display_counts'] = displayed_tile_counts(player['assigned_counts'], player['expected_counts'])
        count_hypotheses = frequencies[index] / SEARCH_SAMPLES
        player['count_hypotheses'] = count_hypotheses.tolist()
        player['tile_support'] = (1 - count_hypotheses[:, 0]).tolist()
        player['riichi'] = bool(riichi[seat])
        player['tenpai_index'] = 100 if riichi[seat] else round(
            100 * sum(distance <= 0 for distance in distances[index]) / SEARCH_SAMPLES)
        player['minimum_draws'] = 1 if riichi[seat] else round(median(distances[index])) + 1
        player['turn_estimate'] = (round(median(estimates[index]), 1)
                                   if len(estimates[index]) >= SEARCH_SAMPLES // 2 else 0)
        player['turns_beyond_round'] = turns_left is not None and player['turn_estimate'] > turns_left
        player['routes'] = (['立直'] if riichi[seat] else
                            [label for label, count in routes[index].most_common(2)
                             if count >= SEARCH_SAMPLES // 4])
        player['risk_trials'] = trials[index]
        player['ron_risks'] = []
        player['risk_available'] = contexts[seat] is not None and sizes[index] + 3 * melds[seat] == 13
        if player['risk_available']:
            for tile in sorted(candidates):
                probability = hits[index][tile] / trials[index] if trials[index] else 0.0
                if offline_calibration is not None and tile not in contexts[seat].discarded:
                    probability = offline_calibration.probability(groups[index], tile, hits[index][tile], trials[index])
                player['ron_risks'].append(dict(
                    tile=tile, probability=probability, hypothesis_hits=hits[index][tile],
                    yaku=[name for name, _ in causes[index].get(tile, Counter()).most_common(3)]))
            player['ron_risks'].sort(key=lambda item: (-item['probability'], item['tile']))
        instant_ron = max(hits[index].values(), default=0) / max(1, trials[index])
        player['danger_index'] = 0 if opening else round(max(
            sum(strengths[index]) / SEARCH_SAMPLES, 51 if riichi[seat] else 0, 100 * instant_ron))
    return dict(prediction, display_threshold=DISPLAY_THRESHOLD, search_samples=SEARCH_SAMPLES)


class PublicTrend:
    """Track unexpected public discards for the current round."""
    def __init__(self):
        self.previous = None
        self.uncertainty = [0] * 4

    def update(self, request, result):
        if 'public_context' not in request:
            self.previous = None
            self.uncertainty = [0] * 4
            return
        history = request['public_context']['discard_history']
        identity = (request.get('round_id'), result['observer'])
        previous = self.previous
        self.previous = (identity, history, result['remaining'], result['opponents'])
        if previous is None or previous[0] != identity or any(
                history[seat][:len(old)] != old for seat, old in enumerate(previous[1])):
            self.uncertainty = [0] * 4
            for player in result['opponents']:
                player['danger_index'] = 0
            return
        _, old_history, remaining, old_players = previous
        pool_size = max(1, sum(remaining))
        for player, old in zip(result['opponents'], old_players):
            seat = player['seat']
            new_tiles = history[seat][len(old_history[seat]):]
            if len(new_tiles) == 1:
                tile = new_tiles[0] - 1
                held = old['tile_support'][tile]
                plausible = held + (1 - held) * remaining[tile] / pool_size
                if plausible < 0.20:
                    self.uncertainty[seat] = min(40, self.uncertainty[seat] + 6)
                else:
                    self.uncertainty[seat] = max(0, self.uncertainty[seat] - 6)
            elif new_tiles:
                self.uncertainty[seat] = 0
            # Public riichi and clear threats take precedence over uncertainty.
            if not player['riichi'] and player['danger_index'] <= 30 and self.uncertainty[seat] > 0:
                player['danger_index'] = -self.uncertainty[seat]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    predictor = HandPredictor(model_path(args.name))
    trend = PublicTrend()
    for line in sys.stdin:
        if line.strip():
            try:
                request = json.loads(line)
                result = live_result(predictor, request)
                trend.update(request, result)
            except (ValueError, TypeError, KeyError) as error:
                result = {'error': str(error)}
            print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
