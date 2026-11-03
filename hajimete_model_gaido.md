# まずモデルを完成させる手順

全件CSV変換は後回しにし、全員の牌譜から「天鳳位達成翌日以降」の少量データだけを変換します。
今回のZIP名監査では、重複除去後の候補は40対局でした。
実際の変換・学習はまだ実行していません。以下は自分で実行するコマンドです。

## 1．準備

全件変換が動いている場合は、先にそのターミナルでCtrl+Cで停止します。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
```

## 2．達成翌日以降だけCSV化

```bash
python paifu_csv_henkan.py --rank-policy after --output-dir tenhou_data/converted/after
```

全15万件を復元せず、ZIP内の名前で日付を選んでから変換します。
1対局1行ではなく、1局面1行の451列CSVです。補助ファイルも一緒に残してください。

```bash
cat tenhou_data/converted/after/summary.json
```

converted_gamesが2以上、conversion_errorsが0であることを確認します。
中断した場合は同じコマンドに --resume を追加して再開できます。

## 3．まず小さいTensorFlowモデルを作成

```bash
python aite_tehai_gakushu.py --mode l --input-dir tenhou_data/converted/after --max-train 12000 --max-test 3000 --epochs 10 --output-dir tenhou_data/models/hajimete_01
```

ここでは対象をL（全員）にし、利用可能な達成後の対局を確保します。
実際に保持するデータは訓練12000行・評価3000行までなので、Lでも全行を学習するわけではありません。
同じ対局は学習と評価に跨がせません。人数を絞ると達成後の対局が2件未満になる場合があります。
人数をSの設定にしたい場合は --mode s に変更します。

## 4．完成したモデル

```text
tenhou_data/models/hajimete_01/tehai_yosoku.keras
```

評価は同じフォルダのmetrics.jsonに保存します。

```bash
cat tenhou_data/models/hajimete_01/metrics.json
```

これはまず学習から保存までを完了させるための初期モデルです。
対局数が少ないため、汎化性能の十分な検証には追加データが必要です。
再学習時は出力先をhajimete_02などに変更してください。

## 後から規模を変える

jikken_settei.pyのplayers・max_games・max_train・max_test・EPOCHSなどを変更します。
CSVに含まれる達成後の対象を使う範囲では、再変換は不要です。
従来の全日付変換も残しています：`python paifu_csv_henkan.py`（出力先converted/all）。
日付で絞ったモデル作成には、上記のafterのデータを使ってください。
