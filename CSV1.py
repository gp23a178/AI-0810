import zipfile
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# 1. カラム定義 (Dual-1)
target_cols = [f'p1_hand_{i}' for i in range(1, 38)]
input_cols = ['round', 'honba'] + [f'dora_{i}' for i in range(1, 6)]
input_cols += [f'p0_hand_{i}' for i in range(1, 38)]
for p in range(4):
    input_cols += [f'p{p}_is_reach']
    input_cols += [f'p{p}_meld_{i}' for i in range(1, 38)]
    input_cols += [f'p{p}_disc_{i}' for i in range(1, 24)]

# 2. モデル構築
model_v1 = Sequential([
    BatchNormalization(input_shape=(len(input_cols),)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(37, activation='relu') 
])
model_v1.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. メモリ節約型データ読み込み関数
def get_csv_list(zip_path):
    if not os.path.exists(zip_path):
        return []
    with zipfile.ZipFile(zip_path, 'r') as z:
        return [f for f in z.namelist() if f.endswith('.csv')]

def load_chunk(zip_path, file_list):
    use_cols = input_cols + target_cols 
    X_list = []
    Y_list = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for f_name in file_list:
            with z.open(f_name) as f:
                try:
                    df = pd.read_csv(f, dtype=np.float32, usecols=use_cols)
                    X_list.append(df[input_cols].values)
                    Y_list.append(df[target_cols].values)
                    del df
                except Exception:
                    continue
                    
    if not X_list:
        return None, None
        
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y

# 4. 学習の実行設定
CHUNK_SIZE = 500     
EPOCHS = 5          # ★ 5から3に短縮（全体の学習時間を40%カット）
EVAL_FILES = 20     

train_zips = ["CLS_c.zip", "Syobon_c.zip"]
test_zips = ["Son_c.zip", "Nisimura_c.zip"]

print("--- Dual-1 評価用データの準備 (Son, Nisimura) ---")
X_val_list, Y_val_list = [], []
for tz in test_zips:
    csv_list = get_csv_list(tz)[:EVAL_FILES]
    if csv_list:
        x, y = load_chunk(tz, csv_list)
        if x is not None:
            X_val_list.append(x)
            Y_val_list.append(y)

X_val = np.vstack(X_val_list) if X_val_list else np.array([])
Y_val = np.vstack(Y_val_list) if Y_val_list else np.array([])

print("\n--- Dual-1 メモリ節約型フルパワー学習を開始 (CLS, Syobon) ---")
for epoch in range(EPOCHS):
    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
    for train_zip in train_zips:
        csv_list = get_csv_list(train_zip)
        total_files = len(csv_list)
        if total_files == 0: continue
            
        print(f"[{train_zip}] を学習中... (全 {total_files} ファイル)")
        for i in range(0, total_files, CHUNK_SIZE):
            chunk_files = csv_list[i:i+CHUNK_SIZE]
            X_train, Y_train = load_chunk(train_zip, chunk_files)
            
            if X_train is not None:
                # ★ バッチサイズを256から2048に増加（処理速度を劇的アップ）
                model_v1.fit(X_train, Y_train, batch_size=2048, epochs=1, verbose=0)
                del X_train, Y_train
                gc.collect()
            
            print(f"  - 進捗: {min(i+CHUNK_SIZE, total_files)} / {total_files} ファイル完了", end='\r')
        print() 
        
    if X_val.size > 0:
        loss, mae = model_v1.evaluate(X_val, Y_val, verbose=0)
        print(f"--> Epoch {epoch+1} 終了時のテストデータMAE: {mae:.4f}")

# ★ 学習完了後の保存処理
print("\n学習が完了しました！モデルを保存しています...")
model_v1.save("mahjong_dual1_model.h5")
print("✅ 保存完了: mahjong_dual1_model.h5")