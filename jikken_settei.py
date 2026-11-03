# -*- coding: utf-8 -*-
"""このファイルの数値を変更するだけでS/M/Lの規模を設定できます。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "tenhou_data"
MODE = "S"                 # "S", "M", "L"（小文字での指定も可能）
SEED = 42                  # 同じ人数・seedなら同じ人を選択

# players: 選ぶ人数。None はmanifest内の全員。
# max_games: 学習に選ぶ対局数の上限。None は条件を満たす全対局。
# この設定ファイルは学習にのみ使用。CSV変換は常に全人数・全日付です。
# max_train / max_test: 学習/評価に保持する行数の上限（人数ではありません）。
SIZE_SETTINGS = {
    "S": {"players": 5,    "max_games": None, "max_train": 12000, "max_test": 3000},
    "M": {"players": 12,   "max_games": None, "max_train": 24000, "max_test": 6000},
    "L": {"players": None, "max_games": None, "max_train": 48000, "max_test": 12000},
}

EPOCHS = 20               # 学習を繰り返す回数
BATCH_SIZE = 128          # 一度に学習する行数
LEARNING_RATE = 0.001


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
