import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_csv_model():
    """ CSVの数値データ(34次元)だけを使って予測するシンプルなモデル """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(34,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(34, activation='relu') # 34次元の予測を出力
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("=== 保存されたデータの読み込みを開始します ===")
    
    # 1. CSVデータの読み込み（数秒で終わります）
    X_train_csv = np.load("X_train_csv.npy")
    y_train = np.load("y_train.npy")
    X_test_csv = np.load("X_test_csv.npy")
    y_test = np.load("y_test.npy")
    
    print(f"読み込み完了! 学習データ: {X_train_csv.shape[0]:,}件")

    # 2. モデルの構築と学習
    print("\n=== モデルの構築と学習開始 ===")
    model = build_csv_model()
    
    # コールバック（保存設定）
    checkpoint = callbacks.ModelCheckpoint('mahjong_csv_model.keras', save_best_only=True)
    
    # 学習の実行
    model.fit(
        X_train_csv, y_train,
        validation_data=(X_test_csv, y_test),
        batch_size=2048,
        epochs=10,
        callbacks=[checkpoint]
    )
    print("学習が完了しました！")