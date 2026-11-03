import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os

def convert_to_vectors(hands_array):
    """ (N行, 14列) の牌ID配列を、(N行, 34次元) の枚数ベクトル配列に変換 """
    N = hands_array.shape[0]
    vectors = np.zeros((N, 34), dtype=np.float32)
    for i in range(N):
        for tid in hands_array[i]:
            if tid != -1:
                tile_type = tid // 4
                vectors[i, tile_type] += 1.0
    return vectors

def load_csv_data_from_zips(zip_paths):
    X_list, y_list = [], []
    p0_cols = [f"p0_{i}" for i in range(14)]
    p1_cols = [f"p1_{i}" for i in range(14)]
    
    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            print(f"警告: {zip_path} が見つかりません。スキップします。")
            continue
            
        print(f"CSV読み込み中: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            for csv_name in csv_files:
                with z.open(csv_name) as f:
                    try:
                        df = pd.read_csv(f)
                    except Exception:
                        continue
                    if not all(col in df.columns for col in p0_cols + p1_cols):
                        continue
                        
                    p0_data = df[p0_cols].fillna(-1).values.astype(int)
                    p1_data = df[p1_cols].fillna(-1).values.astype(int)
                    
                    X_list.append(convert_to_vectors(p0_data))
                    y_list.append(convert_to_vectors(p1_data))

    if not X_list:
        return np.array([]), np.array([])
    return np.vstack(X_list), np.vstack(y_list)

if __name__ == "__main__":
    TRAIN_ZIP_PATHS = ["Syobon_c.zip", "CLS_c.zip"]
    TEST_ZIP_PATHS = ["Son_c.zip", "Nisimura_c.zip"]

    print("\n=== 学習データの構築 (CSV) ===")
    X_train, y_train = load_csv_data_from_zips(TRAIN_ZIP_PATHS)
    if len(X_train) > 0:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(64).prefetch(tf.data.AUTOTUNE)
        print(f"完了! X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("\n=== テストデータの構築 (CSV) ===")
    X_test, y_test = load_csv_data_from_zips(TEST_ZIP_PATHS)
    if len(X_test) > 0:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
        print(f"完了! X_test: {X_test.shape}, y_test: {y_test.shape}")

        # === CSV_3.py の一番最後（print文の下など）に追記 ===
    print("\n=== データのファイル保存を開始します ===")
    
    # データをNumpy形式(.npy)でローカルに保存
    np.save("X_train_csv.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test_csv.npy", X_test)
    np.save("y_test.npy", y_test)
    
    print("CSVデータの保存が完全に完了しました！")