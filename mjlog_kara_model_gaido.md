# まずモデル作成を終えたい場合

`hajimete_model_gaido.md` を先に開いてください。
全日付ではなく達成翌日以降だけを先にCSV化する、短い手順を追加しました。

---

# MJLOGを通常のCSVに保存し、TensorFlowモデルを作る

## 1．全牌譜をCSVへ変換する

VS CodeのWSL Ubuntuターミナルで1行ずつ実行します。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
python paifu_csv_henkan.py
```

元のZIPを手動で解凍する必要はありません。保存済みmanifestに載る全員・全日付が対象です。
同一対局の重複は除きます。S/M/Lによる人数の選択と達成日フィルターは学習時に行います。

### 保存先

```text
tenhou_data/converted/all/games/
├── <対局ID>.csv          ← 添付と同じ451列の通常CSV
├── <対局ID>.meta.csv     ← 鳴き・河などの追加の公開情報
└── <対局ID>.done.json    ← 途中再開用の完成記録
```

`.csv` は無圧縮、UTF-8 BOM付きです。4人分の手牌を含む参照データとして保存します。
席番号は元XMLの0～3です。ファイル名の &tw= による席の回転は行いません。
添付と同じ列名・列順ですが、値は実際のMJLOGを逐次復元して計算します。
行動は0=配牌、1=ツモ、2=打牌、3=鳴き、4=リーチ、5=ドラ追加です。行動者なしは-1です。
1局面を1行として保存し、学習時に各観測者・相手の組み合わせへ展開します。
参照CSV・補助CSV・再開記録・一覧はフォルダ一式で保管してください。
通常CSVは圧縮CSVより大きくなるため、保存先の空き容量を確認してください。

## 2．中断した場合は続きから再開する

```bash
python paifu_csv_henkan.py --resume
```

新しい形式で正常に完成した対局を再利用し、未完了・不一致の対局を作り直します。
今回削除した旧.csv.gz形式のデータは再開対象ではありません。
途中で中断した保存先を学習に使わず、全件処理が終わってから次へ進んでください。

### 並列処理数を変更する場合

既定は最大4プロセスです。PCが重い場合は2や1に下げます。

```bash
python paifu_csv_henkan.py --workers 2
python paifu_csv_henkan.py --resume --workers 2
```

この2行は使い分けの例です。新規なら1行目、再開なら2行目だけを実行します。
ZIPをプロセス内で再利用し、同じ局面の12重保存とgzip圧縮を省いています。
実データでの速度測定はまだ行っていません。CPU・ディスク性能によって速度は変わります。

## 3．変換完了を確認する

```bash
cat tenhou_data/converted/all/summary.json
```

| 項目 | 見る点 |
|---|---|
| converted_games | 正常に保存した対局数 |
| training_rows | 保存した局面数。旧版の12視点へ展開済み行数とは単位が違います |
| conversion_errors | 0なら復元エラーなし |

復元エラーはconversion_errors.csv、無効な牌譜IDなどはlogs_manifest.csvで確認します。
未完了・変換エラーがある全件データでの学習は停止する仕様です。

## 4．学習の設定を変更する

`jikken_settei.py` を開き、値を編集してCtrl+Sで保存します。このファイルを直接実行する必要はありません。

| 設定 | 意味 |
|---|---|
| S/M/Lのplayers | 選ぶ人数。Noneは全員 |
| max_games | 学習に選ぶ最大対局数。Noneは条件に合う全対局 |
| max_train / max_test | 訓練／評価に使う最大行数 |
| EPOCHS | 学習を繰り返す回数 |
| BATCH_SIZE | 1回の更新に使う行数 |
| LEARNING_RATE | 学習率 |

人数・対局数・CSV行数は別の単位です。条件に合う元対局が少ない場合、上限まで使えないことがあります。

## 5．TensorFlowでモデルを作る

```bash
python aite_tehai_gakushu.py --mode s --output-dir tenhou_data/models/S_01
```

Sの設定で選んだ人の、天鳳位達成翌日以降の対局だけを使います。
参照CSVには他家の手牌もありますが、学習コードが自身の手牌・公開情報をX、相手の手牌をyに分けます。
全数値列をそのままモデル入力に使わないでください。
他家の非公開ツモ牌・裏ドラ・未来情報を入力から除外し、同じ対局を訓練と評価に跨がせません。
正規化は訓練側だけで計算します。

| 生成物 | 保存先 |
|---|---|
| 学習済みモデル | tenhou_data/models/S_01/tehai_yosoku.keras |
| 評価結果 | tenhou_data/models/S_01/metrics.json |
| 学習履歴 | tenhou_data/models/S_01/gakushu_rireki.json |
| 選んだ人・対局 | 同フォルダのgakushu_players.csv・gakushu_games.csv |

設定を変えて再学習するときは、CSV変換は不要です。
既存モデルを残すため、出力先をS_02など別名にします。

```bash
python aite_tehai_gakushu.py --mode s --epochs 30 --output-dir tenhou_data/models/S_02
```

M/Lを使うときは --mode m / --mode l を指定します。
モデルは34種類の牌の予測枚数を出し、合法な13枚の手牌の生成までは行いません。

## 今回の変更・確認

- 中断した旧変換結果190ファイルを削除。元ZIP・manifest・コード・設定・TensorFlow環境は保持。
- 添付と同じ451列の非圧縮CSVへ変更。
- ZIP再利用・1局面1行・並列処理・途中再開を実装。
- 合成データによる12件の回帰テストに成功。並列/逐次のCSV一致、再開、失敗除外、情報漏洩防止、対局分割を検証。
- 実牌譜の全件変換・モデル学習・実データでの速度測定は未実行。
