# 設定・コマンド対応表（全件CSV保存方式）

基本手順・設定行番号・保存先は `mjlog_kara_model_gaido.md` を参照してください。

## 全件CSV変換：paifu_csv_henkan.py

| 指定 | 意味 |
|---|---|
| 引数なし | 全員・全日付の牌譜を451列の通常CSVとしてconverted/all/gamesへ保存 |
| --base-dir | archivesとdownload_manifest.csvを置いたデータフォルダ |
| --output-dir | 全件CSVの保存先。既定はtenhou_data/converted/all |
| --workers 4 | 並列処理数（既定は最大4） |
| --resume | 完成済みCSVを再利用して続行 |
| --check-log | 単一MJLOGの診断専用。正式な全件学習データには使用不可 |

## 学習：aite_tehai_gakushu.py

| 引数 | 意味 |
|---|---|
| --mode s/m/l | 使用する設定の選択 |
| --players 3 | 今回だけ3人。0なら全員 |
| --max-games 20 | 今回の学習で選ぶ最大対局数（2以上） |
| --max-train 10000 | 訓練行数上限 |
| --max-test 2000 | 評価行数上限 |
| --epochs 30 | 学習回数 |
| --batch-size 64 | バッチサイズ |
| --learning-rate 0.0001 | 学習率 |
| --seed 123 | 人数・対局抽出・分割などのseed |
| --input-dir | 全件CSVと一覧の保存フォルダ |
| --output-dir | モデルの保存先 |

コマンド引数はjikken_settei.pyの値より優先します。
対局を絞ってから対局単位で分割し、その後に訓練・評価側でそれぞれ行をサンプリングします。
評価用データを学習や正規化に使用しません。
モデル名はtehai_yosoku.kerasで固定し、実験の違いは保存フォルダ名で管理する方法が簡単です。

合成データでの回帰テスト：`python joho_more_kakunin.py`。
全件変換・学習の動作確認はこの方式ではまだ実行していません。

通常CSVと.meta.csvと.done.jsonを一式で管理してください。
添付と同じ451列の全手牌を含むCSVから、学習時に自分視点のXと相手手牌のyへ安全に展開します。
