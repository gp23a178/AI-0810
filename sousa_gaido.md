# 天鳳AI 操作ガイド（Pythonは4本）

| ファイル | 役割 |
|---|---|
| paifu_ikkatsu_hozon.py | 天鳳から牌譜ZIPを一括保存 |
| paifu_csv_henkan.py | MJLOGを451列の通常CSVへ変換。並列・途中再開に対応 |
| aite_tehai_gakushu.py | 冒頭でパラメータを設定し、名前付きTensorFlowモデルを構築 |
| moderu_jikko.py | モデル名を指定して評価、または1局面の予測を実行 |

## 準備

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
```

## 1．元牌譜のダウンロード（既に保存済みなら不要）

```bash
python paifu_ikkatsu_hozon.py
```

保存先はtenhou_data/archives、対応一覧はtenhou_data/download_manifest.csvです。

## 2．CSVへの変換（既存afterのCSVは再変換不要）

モデル作成を優先し、達成翌日以降だけ変換する場合：

```bash
python paifu_csv_henkan.py --rank-policy after
```

保存先はtenhou_data/converted/after/gamesです。
中断後は同じコマンドに --resume を追加します。
全日付の変換は `python paifu_csv_henkan.py`（converted/allへ保存）。
.csvと.meta.csvと.done.json、一覧ファイルは一式で保持してください。

## 3．学習ファイル冒頭の設定を変える

`aite_tehai_gakushu.py` の「設定（ここを変更して保存）」を編集し、Ctrl+Sで保存します。
旧jikken_settei.pyの設定はこのファイルに統合しました。

| 設定名 | 意味 |
|---|---|
| MODE | S/M/L。全員を使う場合はLのplayers=Noneを使用 |
| SIZE_SETTINGSのplayers | 選ぶ人数。None=全員 |
| SIZE_SETTINGSのmax_games | 学習に選ぶ対局数の上限。None=対象全対局 |
| SIZE_SETTINGSのmax_train / max_test | 訓練／評価の最大行数 |
| EPOCHS | 学習回数 |
| BATCH_SIZE | 1回の更新で使う行数 |
| LEARNING_RATE | 学習率 |
| SEED | 抽出・分割・初期化の再現用の値 |
| INPUT_DIR | 読み込むCSV一式の場所。既定はconverted/after |
| MODEL_NAME | 作成するモデル名。既定はshiken_01 |

コマンド引数で指定した値は、冒頭の設定より優先します。

## 4．モデルに名前を付けて構築する

```bash
python aite_tehai_gakushu.py --mode l --name shiken_01
```

Lの設定を使用します。ほかの数値・CSV保存先は冒頭の設定を使います。
名前もMODEも冒頭で設定した場合は `python aite_tehai_gakushu.py` だけで実行できます。

保存先：tenhou_data/models/shiken_01/
モデル本体はtehai_yosoku.kerasです。設定・対局分割・入力列なども同じフォルダに保存します。
この段階では訓練だけを行い、評価対局のCSVは読みません。
別の設定で作るときはshiken_02など別名にします。既存モデルは上書きしません。

## 5．そのモデルを実行して評価する

```bash
python moderu_jikko.py --name shiken_01
```

モデル名を変えるだけで別モデルを評価できます。
例：既存のjikken_02は `python moderu_jikko.py --name jikken_02`。
冒頭のMODEL_NAMEを変えてから `python moderu_jikko.py` だけで実行することもできます。

最後にMAE、平均予測MAE、改善幅、削減率、MSE、RMSEを表示します。
評価結果はモデル内のhyoka/metrics.jsonに保存します。
評価は保存時に確保した未学習の対局を使い、再学習はしません。
後で学習ファイルの数値を変えても、そのモデルの保存済み評価条件は変わりません。

## 6．1局面の手牌予測を見たい場合（同じ実行ファイルを使う）

```bash
python moderu_jikko.py --name shiken_01 --action yosoku --show-answer
```

局面を変える例：

```bash
python moderu_jikko.py --name shiken_01 --action yosoku --state-index 100 --show-answer
```

評価用牌譜を使い、指定局面の予測と正解を表示します。
正解の手牌は比較表示にだけ使用し、モデル入力に混ぜません。
実際のライブ対局に接続する機能は含まれていません。

## 整理について

研究用Pythonは上記4本に統合し、ガイドはこのファイル1本にまとめました。
元ZIP・manifest・CSV・学習済みモデル・Python環境は保持しています。
設定専用・評価専用・予測専用・テスト専用の旧Python、旧研究コード、重複ガイドは削除しました。
旧モデルhajimete_01/jikken_02も読み込めます。
統合後の回帰テスト12件と既存モデルの予測を確認済みです。新しいモデルの再学習は実行していません。
