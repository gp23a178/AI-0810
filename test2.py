import os
import zipfile
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# 1. カラム定義
target_cols = [f'p1_hand_{i}' for i in range(1, 38)]
input_cols = ['round', 'honba', 'kyotaku', 'remain_tsumo'] 
input_cols += [f'dora_{i}' for i in range(1, 6)]
input_cols += [f'p{i}_score' for i in range(4)]
input_cols += [f'visible_{i}' for i in range(1, 38)]
input_cols += [f'p0_hand_{i}' for i in range(1, 38)]
for p in range(4):
    input_cols += [f'p{p}_is_reach']
    input_cols += [f'p{p}_meld_{i}' for i in range(1, 38)]
    input_cols += [f'p{p}_disc_{i}' for i in range(1, 25)]

# 牌のインデックス(1〜37)を名前に変換する関数
def get_tile_name(idx):
    if 1 <= idx <= 9: return f"{idx}萬"
    elif idx == 10: return "赤5萬"
    elif 11 <= idx <= 19: return f"{idx-10}筒"
    elif idx == 20: return "赤5筒"
    elif 21 <= idx <= 29: return f"{idx-20}索"
    elif idx == 30: return "赤5索"
    elif 31 <= idx <= 37:
        jihai = ["東", "南", "西", "北", "白", "發", "中"]
        return jihai[idx - 31]
    return f"不明({idx})"

def load_test_data(zip_path, max_files=5):
    if not os.path.exists(zip_path):
        print(f"エラー: {zip_path} が見つかりません。")
        return None, None
        
    use_cols = input_cols + target_cols
    X_list, Y_list = [], []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')][:max_files]
        for f_name in csv_files:
            with z.open(f_name) as f:
                try:
                    df = pd.read_csv(f, dtype=np.float32, usecols=use_cols)
                    X_list.append(df[input_cols].values)
                    Y_list.append(df[target_cols].values)
                except Exception:
                    continue
                    
    if not X_list: return None, None
    return np.vstack(X_list), np.vstack(Y_list)

print("--- Dual-2 モデルのテストを開始します ---")
model_path = "mahjong_dual2_model.h5"
if not os.path.exists(model_path):
    print(f"エラー: {model_path} が見つかりません。")
    exit()

print("モデルを読み込み中...")
model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✅ 読み込み完了！\n")

print("テストデータを準備中...")
X_test, Y_test = load_test_data("Son_c_v2.zip", max_files=3) 

if X_test is not None:
    print("\n【1】全体テスト")
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    print(f"平均絶対誤差 (MAE): {mae:.4f}")
    
    print("\n【2】直感テスト (実際の予測結果を見てみる)")
    indices = np.random.choice(len(X_test), 3, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n--- テスト {i+1} ---")
        x_sample = X_test[idx].reshape(1, -1)
        y_true = Y_test[idx]
        
        y_pred = model.predict(x_sample, verbose=0)[0]
        
        # 牌の名前に変換して、数値も小数点第1位までにスッキリさせる
        true_tiles = [f"{get_tile_name(j+1)}:{int(y_true[j])}枚" for j in range(37) if y_true[j] > 0]
        pred_tiles = [f"{get_tile_name(j+1)}:{y_pred[j]:.1f}枚" for j in range(37) if y_pred[j] > 0.5]
        
        print(f"🎯 正解の手牌 : {', '.join(true_tiles) if true_tiles else 'なし'}")
        print(f"🤖 AIの予測  : {', '.join(pred_tiles) if pred_tiles else 'なし'}")