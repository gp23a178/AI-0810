# -*- coding: utf-8 -*-
"""機械学習(104次元_完全版) - ローカル環境用"""

import tensorflow as tf
import numpy as np
import os
import re
import zipfile
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# 牌のテキストをIDに変換する辞書
tile_to_id = {
    "一萬":0,"二萬":1,"三萬":2,"四萬":3,"五萬":4,
    "六萬":5,"七萬":6,"八萬":7,"九萬":8,

    "一筒":9,"二筒":10,"三筒":11,"四筒":12,"五筒":13,
    "六筒":14,"七筒":15,"八筒":16,"九筒":17,

    "一索":18,"二索":19,"三索":20,"四索":21,"五索":22,
    "六索":23,"七索":24,"八索":25,"九索":26,

    "東":27,"南":28,"西":29,"北":30,
    "白":31,"發":32,"中":33
}

def load_data_from_zips(zip_paths):
    X_list = []
    y_list = []

    for zip_path in zip_paths:
        print(f"読み込み中: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                files = [f for f in z.namelist() if f.endswith(".txt")]
                for idx, filename in enumerate(files):
                    try:
                        text = z.read(filename).decode("utf-8")
                        lines = text.splitlines()

                        current_hand = None
                        field_tiles = []
                        dora_tiles = []
                        self_reach = 0
                        other_reach = 0

                        for line in lines:
                            if line.startswith("[配牌]"):
                                current_hand = line.replace("[配牌]", "").strip().split()
                                field_tiles = []
                                dora_tiles = []
                                self_reach = 0
                                other_reach = 0
                                continue
                            if current_hand is None: continue

                            if line.startswith("[ドラ]"):
                                dora = line.replace("[ドラ]", "").strip()
                                if dora in tile_to_id: dora_tiles.append(dora)
                            elif line.startswith("[自リーチ]"): self_reach = 1
                            elif line.startswith("[他家リーチ]"): other_reach += 1
                            elif line.startswith("[ツモ]"):
                                current_hand.append(line.replace("[ツモ]", "").strip())
                            elif line.startswith("[他家打牌]"):
                                t = line.replace("[他家打牌]", "").strip()
                                if t in tile_to_id: field_tiles.append(t)
                            elif line.startswith("[他家鳴き]"):
                                for t in line.replace("[他家鳴き]", "").strip().split():
                                    if t in tile_to_id: field_tiles.append(t)
                            elif line.startswith("[手牌から消費]"):
                                for t in line.replace("[手牌から消費]", "").strip().split():
                                    if t in tile_to_id:
                                        field_tiles.append(t)
                                        if t in current_hand: current_hand.remove(t)

                            elif line.startswith("[打牌]"):
                                discard = line.replace("[打牌]", "").strip()
                                if discard not in tile_to_id or discard not in current_hand: continue

                                # 4つの情報を数値ベクトル化
                                hand_vec = np.zeros(34, dtype=np.int8)
                                for t in current_hand:
                                    if t in tile_to_id: hand_vec[tile_to_id[t]] += 1

                                field_vec = np.zeros(34, dtype=np.int8)
                                for t in field_tiles:
                                    if t in tile_to_id: field_vec[tile_to_id[t]] += 1

                                dora_vec = np.zeros(34, dtype=np.int8)
                                for t in dora_tiles:
                                    if t in tile_to_id: dora_vec[tile_to_id[t]] += 1

                                reach_vec = np.array([self_reach, other_reach], dtype=np.int8)

                                # 全情報を結合 (34 + 34 + 34 + 2 = 104次元)
                                X_list.append(np.concatenate([hand_vec, field_vec, dora_vec, reach_vec]))
                                y_list.append(tile_to_id[discard])

                                current_hand.remove(discard)
                                field_tiles.append(discard)

                    except Exception as e: pass
        except Exception as e: print(f"エラー: {zip_path} ({e})")

    return np.array(X_list), np.array(y_list)

def main():
    # 学習用ZIPファイルのパス
    train_zip_paths = [
        "しょぼんn_AI学習用.zip",
        "CLS_AI学習用.zip"
    ]

    # テスト用ZIPファイルのパス
    test_zip_paths = [
        "Son_AI学習用.zip",
        "Nisimura_AI学習用.zip"
    ]

    print("=== 学習データの読み込み ===")
    X_train, y_train = load_data_from_zips(train_zip_paths)

    print("\n=== テストデータの読み込み ===")
    X_test, y_test = load_data_from_zips(test_zip_paths)

    print("\nデータ件数の確認:")
    print("学習用 データ (X_train):", X_train.shape, " / 正解ラベル (y_train):", y_train.shape)
    print("テスト用 データ (X_test):", X_test.shape, " / 正解ラベル (y_test):", y_test.shape)

    # データが空の場合はエラーになるためストップ
    if len(X_train) == 0 or len(X_test) == 0:
        print("エラー: 学習データまたはテストデータが読み込めませんでした。ZIPファイルが存在するか確認してください。")
        return

    # === ここからAIモデルの構築と学習 ===
    print("\n=== モデルの構築 ===")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(104,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(34, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\n=== 学習の開始 ===")
    # 学習を実行する
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 最終的な正解率を確認
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nテストデータでの予測精度: {acc * 100:.2f}%")

    # 学習曲線のプロット
    plt.figure(figsize=(10,5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Test Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# プログラムの実行トリガー（これが無いとmain関数が動かない）
if __name__ == "__main__":
    main()