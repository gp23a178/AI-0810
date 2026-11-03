# ファイル名: run_csv_predict.py
import pandas as pd
import tensorflow as tf
import zipfile
import numpy as np
import os

TRAIN_ZIP_PATHS = ["Syobon_c.zip", "CLS_c.zip"]
TEST_ZIP_PATHS = ["Son_c.zip", "Nisimura_c.zip"]
MODEL_NAME = "model_csv_predict.keras"

INPUT_DIM = 4 + 4 + 136 + 136 + 136

def create_feature_vector(rnd, honba, step, actor, scores, hand_ids, discard_ids, dora_ids):
    base_feat = np.array([rnd, honba, step, actor], dtype=np.float32)
    score_feat = np.array(scores, dtype=np.float32) / 10000.0
    
    hand_feat = np.zeros(136, dtype=np.float32)
    for tid in hand_ids:
        if 0 <= tid < 136: hand_feat[tid] = 1.0
            
    discard_feat = np.zeros(136, dtype=np.float32)
    for tid in discard_ids:
        if 0 <= tid < 136: discard_feat[tid] = 1.0
            
    dora_feat = np.zeros(136, dtype=np.float32)
    for tid in dora_ids:
        if 0 <= tid < 136: dora_feat[tid] = 1.0
            
    return np.concatenate([base_feat, score_feat, hand_feat, discard_feat, dora_feat])

def load_csv_data(zip_paths):
    X_list, y_list = [], []
    
    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            print(f"\n[警告] {zip_path} が見つかりません。")
            continue
            
        print(f"\n>> {zip_path} を処理中...")
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            for i, file_name in enumerate(csv_files):
                with z.open(file_name) as f:
                    df = pd.read_csv(f)
                    
                    current_round = 0
                    current_honba = 0
                    hands = {0: [], 1: [], 2: [], 3: []}
                    current_discards = []
                    current_dora = []
                    
                    for row in df.itertuples(index=False):
                        act = int(row.action)
                        step = int(row.step)
                        actor = int(row.actor)
                        tile = int(row.tile)
                        
                        if act == 0:
                            current_round = int(row.round)
                            current_honba = int(row.honba)
                            current_discards = []
                            current_dora = [tile] if tile >= 0 else []
                            for p in range(4):
                                hands[p] = [int(getattr(row, f"p{p}_hp_{idx}")) for idx in range(14) if int(getattr(row, f"p{p}_hp_{idx}")) >= 0]
                                
                        elif act == 1:
                            if tile >= 0 and actor in hands:
                                hands[actor].append(tile)
                                
                        elif act == 2:
                            if tile >= 0 and actor in hands:
                                current_hand = hands[actor].copy()
                                X_feat = create_feature_vector(
                                    current_round, current_honba, step, actor,
                                    [0, 0, 0, 0],
                                    current_hand, current_discards, current_dora
                                )
                                X_list.append(X_feat)
                                y_list.append(tile)
                                
                                if tile in hands[actor]:
                                    hands[actor].remove(tile)
                                current_discards.append(tile)
                                
                print(f"\r  進捗: {i+1}/{len(csv_files)} ファイル完了", end="")
        print()
        
    if not X_list: return None, None
    X_data = tf.convert_to_tensor(np.array(X_list), dtype=tf.float32)
    y_data = tf.convert_to_tensor(np.array(y_list), dtype=tf.int32)
    return X_data, y_data

def main():
    print("【CSV版: 打牌予測モデル】")
    X_train, y_train = load_csv_data(TRAIN_ZIP_PATHS)
    if X_train is None: return

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(136, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n学習を開始します...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
    model.save(MODEL_NAME)

    print("\nテストデータで評価を行います...")
    X_test, y_test = load_csv_data(TEST_ZIP_PATHS)
    if X_test is not None:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print("\n" + "="*40)
        print("【CSV版: 最終テスト結果】")
        print(f"正解率 (Accuracy): {accuracy * 100:.2f} %")
        print("="*40)

if __name__ == "__main__":
    main()