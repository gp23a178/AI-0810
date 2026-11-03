# -*- coding: utf-8 -*-
"""天鳳牌譜ZIPから局面CSVと、観測者に見える情報だけの学習データを作る。

Python標準ライブラリのみを使用します。

出力:
  tenhou_data/converted/<S|M|L>/states_full.csv.gz
      添付CSVと同じ主要451列 + メタデータと鳴き構造情報。検証用。
      4人全員の実際の手牌を含むため、モデルの入力には直接使わないでください。
  tenhou_data/converted/<S|M|L>/training_samples.csv.gz
      1局面×自分視点×相手1人の教師データ。特徴量には自分の手牌だけを含め、
      予測対象相手の門前部分（副露と暗槓を除いた伏せた手牌）は
      target_hand_1..37 の正解ラベルとして分離します。
  tenhou_data/converted/<S|M|L>/selected_players.csv
      ランダム抽出されたプレイヤー一覧。
  tenhou_data/converted/<S|M|L>/selected_games.csv
      選択プレイヤーのZIPから採用したゲーム一覧。
  tenhou_data/converted/<S|M|L>/logs_manifest.csv
      発見したログと、天鳳位達成日フィルターによる採用/除外理由。
  tenhou_data/converted/<S|M|L>/conversion_errors.csv
      ZIP/XML/牌譜解析で発生したエラー。

天鳳のXML牌譜書式は変更される場合があります。初回はSまたはMで結果を検証してください。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ========================== 設定（ここを変更） ==========================
BASE_DIR = Path("tenhou_data")
SAMPLE_MODE = "S"        # 初回は S（5人）。実行時 --mode M または --mode L で変更
RANDOM_SEED = 42          # 再現可能な抽出。毎回変えたい場合は None
EXCLUDE_ON_OR_BEFORE_RANK_DATE = True  # 達成日当日も安全側に除外
WRITE_FULL_REFERENCE = True            # 添付CSVに近い4人全員の状態CSVも出力
MAX_DISCARDS_PER_PLAYER = 24
MAX_MELD_SLOTS = 4
DOWNLOAD_MANIFEST = BASE_DIR / "download_manifest.csv"
OUTPUT_BASE = BASE_DIR / "converted"

SAMPLE_PLAYER_COUNTS = {"S": 5, "M": 12, "L": None}
RED_RAW_TO_CODE = {16: 35, 52: 36, 88: 37}  # 赤5萬/赤5筒/赤5索を通常5と別カテゴリにする
GAME_ID_RE = re.compile(r"(?P<id>(?P<date>20\d{6})\d{2}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4,}-[0-9a-fA-F]{8})")

ACTION_INITIAL = 0
ACTION_DRAW = 1
ACTION_DISCARD = 2
ACTION_MELD = 3
ACTION_REACH = 4
ACTION_DORA = 5
MELD_TYPE = {"none": 0, "chi": 1, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 5, "nuki": 6}
MELD_TYPE_NAME = {v: k for k, v in MELD_TYPE.items()}


# ========================== 列定義 ==========================
BASE_COLUMNS = [
    "round", "honba", "kyotaku", "remain_tsumo", "step", "actor",
    "action", "target_tile", "tsumogiri",
]
BASE_COLUMNS += [f"dora_{i}" for i in range(1, 6)]
BASE_COLUMNS += [f"p{i}_score" for i in range(4)]
BASE_COLUMNS += [f"visible_{i}" for i in range(1, 38)]

REFERENCE_COLUMNS = BASE_COLUMNS[:]
for _p in range(4):
    REFERENCE_COLUMNS.append(f"p{_p}_is_reach")
    REFERENCE_COLUMNS += [f"p{_p}_hand_{i}" for i in range(1, 38)]
    REFERENCE_COLUMNS += [f"p{_p}_meld_{i}" for i in range(1, 38)]
    REFERENCE_COLUMNS += [f"p{_p}_disc_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
assert len(REFERENCE_COLUMNS) == 451, f"参照列数が451列ではありません: {len(REFERENCE_COLUMNS)}"

MELD_EXTRA_COLUMNS = []
for _p in range(4):
    MELD_EXTRA_COLUMNS += [
        f"p{_p}_meld_count", f"p{_p}_chi_count", f"p{_p}_pon_count",
        f"p{_p}_daiminkan_count", f"p{_p}_ankan_count", f"p{_p}_kakan_count",
        f"p{_p}_nuki_count",
    ]
    for _slot in range(1, MAX_MELD_SLOTS + 1):
        MELD_EXTRA_COLUMNS += [
            f"p{_p}_meld{_slot}_type", f"p{_p}_meld{_slot}_from_rel",
            f"p{_p}_meld{_slot}_tile1", f"p{_p}_meld{_slot}_tile2",
            f"p{_p}_meld{_slot}_tile3", f"p{_p}_meld{_slot}_tile4",
        ]
RIVER_EXTRA_COLUMNS = []
PUBLIC_STATUS_COLUMNS = []
ANKAN_TRUE_COLUMNS = []
for _p in range(4):
    RIVER_EXTRA_COLUMNS += [f"p{_p}_disc_called_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
    PUBLIC_STATUS_COLUMNS += [f"p{_p}_ippatsu", f"p{_p}_double_riichi", f"p{_p}_seat_wind"]
    # Oracle-only columns: actual tile type in a concealed kan is not public to opponents.
    ANKAN_TRUE_COLUMNS += [f"p{_p}_ankan_true_{i}" for i in range(1, 38)]
META_COLUMNS = ["game_id", "game_date", "source_player", "source_zip", "state_index", "oya", "game_wind", "hand_number"]
FULL_COLUMNS = REFERENCE_COLUMNS + META_COLUMNS + MELD_EXTRA_COLUMNS + RIVER_EXTRA_COLUMNS + PUBLIC_STATUS_COLUMNS + ANKAN_TRUE_COLUMNS

TRAIN_META_COLUMNS = ["game_id", "game_date", "source_player", "source_zip", "state_index", "observer", "target_player"]
TRAIN_BASE_COLUMNS = [
    "round", "honba", "kyotaku", "remain_tsumo", "step", "actor", "action",
    "target_tile", "target_tile_is_hidden", "tsumogiri",
]
TRAIN_BASE_COLUMNS += [f"dora_{i}" for i in range(1, 6)]
TRAIN_BASE_COLUMNS += [f"p{i}_score" for i in range(4)]
TRAIN_BASE_COLUMNS += [f"visible_{i}" for i in range(1, 38)]
TRAIN_PUBLIC_COLUMNS = []
for _p in range(4):
    TRAIN_PUBLIC_COLUMNS.append(f"p{_p}_is_reach")
    TRAIN_PUBLIC_COLUMNS += [f"p{_p}_meld_{i}" for i in range(1, 38)]
    TRAIN_PUBLIC_COLUMNS += [f"p{_p}_disc_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
TRAIN_PUBLIC_COLUMNS += ["oya", "game_wind", "hand_number"] + MELD_EXTRA_COLUMNS + RIVER_EXTRA_COLUMNS + PUBLIC_STATUS_COLUMNS
TRAIN_LABEL_COLUMNS = [f"target_hand_{i}" for i in range(1, 38)]
TRAIN_COLUMNS = TRAIN_META_COLUMNS + TRAIN_BASE_COLUMNS + TRAIN_PUBLIC_COLUMNS + [f"observer_hand_{i}" for i in range(1, 38)] + TRAIN_LABEL_COLUMNS
