"""Estimate ron eligibility from predicted hands and public information."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json

from mahjong.agari import Agari
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

RULES = OptionalRules(has_open_tanyao=True)
MELD_TYPES = {4: (Meld.KAN, 4, True), 5: (Meld.PON, 3, True),
              6: (Meld.CHI, 3, True), 7: (Meld.SHOUMINKAN, 4, True),
              8: (Meld.KAN, 4, False)}
CALIBRATION_VERSION = 1
YAKU_NAMES = {'Riichi': '立直',
 'DaburuRiichi': 'ダブル立直',
 'Ippatsu': '一発',
 'Tsumo': '門前清自摸和',
 'Pinfu': '平和',
 'Tanyao': '断么九',
 'Iipeiko': '一盃口',
 'Chankan': '搶槓',
 'Rinshan': '嶺上開花',
 'Haitei': '海底摸月',
 'Houtei': '河底撈魚',
 'NagashiMangan': '流し満貫',
 'Haku': '役牌・白',
 'Hatsu': '役牌・發',
 'Chun': '役牌・中',
 'SeatWindEast': '自風・東',
 'SeatWindSouth': '自風・南',
 'SeatWindWest': '自風・西',
 'SeatWindNorth': '自風・北',
 'RoundWindEast': '場風・東',
 'RoundWindSouth': '場風・南',
 'RoundWindWest': '場風・西',
 'RoundWindNorth': '場風・北',
 'Sanshoku': '三色同順',
 'Ittsu': '一気通貫',
 'Chantai': '混全帯么九',
 'Honroto': '混老頭',
 'Toitoi': '対々和',
 'Sanankou': '三暗刻',
 'SanKantsu': '三槓子',
 'SanshokuDoukou': '三色同刻',
 'Chiitoitsu': '七対子',
 'Shosangen': '小三元',
 'Honitsu': '混一色',
 'Junchan': '純全帯么九',
 'Ryanpeikou': '二盃口',
 'Chinitsu': '清一色',
 'KokushiMusou': '国士無双',
 'ChuurenPoutou': '九蓮宝燈',
 'Suuankou': '四暗刻',
 'Daisangen': '大三元',
 'Shousuushii': '小四喜',
 'Ryuuiisou': '緑一色',
 'SuuKantsu': '四槓子',
 'Tsuuiisou': '字一色',
 'Chinroutou': '清老頭',
 'DaiSuushii': '大四喜',
 'DaburuKokushiMusou': '国士無双十三面待ち',
 'SuuankouTanki': '四暗刻単騎',
 'DaburuChuurenPoutou': '純正九蓮宝燈',
 'Tenhou': '天和',
 'Chiihou': '地和',
 'Dora': 'ドラ',
 'AkaDora': '赤ドラ',
 'UraDora': '裏ドラ'}


def decode_melds(tiles, flags):
    """Decode the PIFW wire format, before constructing library-specific tile IDs."""
    if len(tiles) != len(flags):
        raise ValueError('副露牌と状態の数が一致しません')
    groups = []
    start = 0
    while start < len(tiles):
        tag = next((value & 15 for value in flags[start:start+4] if value & 15), None)
        if tag not in MELD_TYPES:
            raise ValueError('副露種別が不明です')
        kind, length, opened = MELD_TYPES[tag]
        group = tuple(tiles[start:start+length])
        marks = flags[start:start+length]
        if len(group) != length or sum(bool(value & 15) for value in marks) != 1:
            raise ValueError('副露の区切りが不正です')
        groups.append((kind, opened, group))
        start += length
    return tuple(groups)


@dataclass(frozen=True)
class RonContext:
    melds: tuple
    discarded: frozenset
    passed: frozenset
    riichi: bool
    double_riichi: bool
    ippatsu: bool
    player_wind: int
    round_wind: int
    houtei: bool
    dora: tuple

    @property
    def upper(self):
        seen = Counter(tile for _, _, meld in self.melds for tile in meld)
        return tuple(4 - seen[tile] for tile in range(1, 35))


@lru_cache(maxsize=8192)
def shape_waits(hand, upper):
    counts = list(hand)
    waits = []
    for index, cap in enumerate(upper):
        if counts[index] < cap:
            counts[index] += 1
            if Agari.is_agari(counts):
                waits.append(index + 1)
            counts[index] -= 1
    return frozenset(waits)


def allowed_ron(hand, context, candidates):
    """Return legal normal-discard ron tiles and their yaku in one hypothetical hand."""
    if sum(hand) + 3 * len(context.melds) != 13:
        return {}
    waits = shape_waits(tuple(hand), context.upper)
    playable = waits & candidates
    # All waits are checked, including no-yaku waits, for discard furiten.
    if not playable or waits & context.discarded:
        return {}

    used = [0] * 34

    def physical(tile):
        index = tile - 1
        identity = index * 4 + used[index]
        used[index] += 1
        return identity

    concealed = [physical(tile) for tile, n in enumerate(hand, 1) for _ in range(n)]
    melds = [Meld(kind, [physical(tile) for tile in tiles], opened=opened)
             for kind, opened, tiles in context.melds]
    fixed = concealed + [tile for meld in melds for tile in meld.tiles]

    def score(tile, latest):
        index = tile - 1
        winning = index * 4 + used[index]
        conf = HandConfig(
            is_riichi=context.riichi and not context.double_riichi,
            is_daburu_riichi=context.double_riichi,
            is_ippatsu=latest and context.ippatsu,
            is_houtei=latest and context.houtei,
            player_wind=context.player_wind, round_wind=context.round_wind, options=RULES,
        )
        return HandCalculator.estimate_hand_value(
            fixed + [winning], winning, melds=melds,
            dora_indicators=[(tile - 1) * 4 for tile in context.dora], config=conf,
        )

    # Whether a player had a legal offer is inferred for this hypothetical hand alone.
    if any(score(tile, False).error is None for tile in waits & context.passed):
        return {}
    result = {}
    for tile in playable:
        value = score(tile, True)
        if value.error is None:
            result[tile] = [YAKU_NAMES.get(type(yaku).__name__, yaku.name) for yaku in value.yaku]
    return result


def contexts_from_public(observation, public):
    fields = {'meld_flags', 'discard_history', 'passed', 'riichi_bits', 'dealer', 'round_wind', 'wall_left'}
    if not isinstance(public, dict) or set(public) != fields:
        raise ValueError('危険度用の公開情報が不正です')
    if any(type(public[key]) is not int for key in ('dealer', 'round_wind', 'wall_left', 'riichi_bits')):
        raise ValueError('Expected integer values for public round information')
    if not 0 <= public['dealer'] < 4 or not 0 <= public['round_wind'] < 4 or not 0 <= public['wall_left'] <= 70:
        raise ValueError('公開された風または山の枚数が不正です')
    for name in ('meld_flags', 'discard_history', 'passed'):
        if not isinstance(public[name], list) or len(public[name]) != 4:
            raise ValueError('危険度用の座席数が不正です')
    for name in ('discard_history', 'passed'):
        if any(type(x) is not int or x not in range(1, 35) for row in public[name] for x in row):
            raise ValueError('打牌履歴の牌IDが不正です')
    result = []
    mask = public['riichi_bits']
    for seat in range(4):
        melds = decode_melds(observation['meld_tiles'][seat], public['meld_flags'][seat])
        if len(melds) != observation['meld_counts'][seat]:
            raise ValueError('副露数が一致しません')
        result.append(RonContext(
            melds, frozenset(observation['rivers'][seat]) | frozenset(public['discard_history'][seat]),
            frozenset(public['passed'][seat]), bool(observation['riichi'][seat]),
            bool(observation['riichi'][seat] and mask & (256 << seat)),
            bool(observation['riichi'][seat] and mask & (16 << seat)),
            27 + (seat - public['dealer']) % 4, 27 + public['round_wind'],
            public['wall_left'] == 0, tuple(observation['dora_indicators']),
        ))
    return result


def group_key(discard_count, riichi, meld_count):
    return f'{min(discard_count // 24, 2)}:{int(riichi)}:{min(meld_count, 3)}'


def tile_bucket(tile):
    return 'honor' if tile >= 28 else 'terminal' if (tile - 1) % 9 in (0, 8) else 'simple'


class DangerCalibration:
    def __init__(self, path):
        self.data = json.loads(path.read_text(encoding='utf-8'))
        if self.data['version'] != CALIBRATION_VERSION:
            raise ValueError('危険度補正のバージョンが一致しません')

    def error(self, key):
        return self.data['residual_rmse'].get(key, self.data['residual_rmse']['all'])

    def prior(self, key, tile):
        bucket = tile_bucket(tile)
        total, hits = self.data['by_tile']['all'][bucket]
        fallback = (hits + 1) / (total + 2)
        stats = self.data['by_tile'].get(key, {}).get(bucket)
        if not stats:
            return fallback
        count, hits = stats
        return (hits + 32 * fallback) / (count + 32)

    def probability(self, key, tile, hits, trials):
        prior = self.prior(key, tile)
        mass = 16 * (0.5 + self.error(key) ** 2 / 0.10)
        return (hits + mass * prior) / (trials + mass)
