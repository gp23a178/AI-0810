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


## 自宅にモデルを持ち帰る

学内で評価したモデルを書き出します。

```bash
python moderu_jikko.py --name jikken_02 --action export
```

mochikaeri/jikken_02.zipを自宅へコピーし、展開先のjitaku_gaido.txtに従います。
MAE・学習設定の閲覧にはPythonだけで足ります。付属サンプルの予測にはTensorFlowとNumPyを使います。
同名のZIPがある場合は --output-dir で別フォルダを指定します。

新規の変換データは.gitignoreで除外します。既にGitで追跡されている変換データを今後のコミットから外すには、学内のリポジトリで次を実行してから変更をコミットします。

```bash
git rm -r --cached -- tenhou_data/converted/
```

この操作はローカルのCSVを保持します。過去のコミットに入ったデータは履歴に残るため、自宅では持ち帰り用ZIPだけで実行できます。


## 変換済みデータの全項目

達成後の40対局でヘッダーを照合しました。範囲で記載した列は、各番号について個別に存在します。pは席番号0～3です。

### 対局CSVの451列

| 列 | 数 | 内容 |
|---|---:|---|
| round, honba, kyotaku, remain_tsumo, step | 5 | 局番号、本場、供託、残りツモ数、局内の進行番号 |
| actor, action, target_tile, tsumogiri | 4 | 行動者、行動種別、対象牌、ツモ切りフラグ |
| dora_1～5 | 5 | 表ドラ表示牌。未公開枠は0 |
| p0_score～p3_score | 4 | 各席の点数 |
| visible_1～37 | 37 | 公開されている牌の種類別枚数 |
| p{p}_is_reach | 4 | 各席のリーチ状態 |
| p{p}_hand_1～37 | 148 | 4人の伏せた手牌の種類別枚数。学習時は自分の分を入力、予測相手の分を正解に使用 |
| p{p}_meld_1～37 | 148 | 各席の副露・槓の種類別牌枚数 |
| p{p}_disc_1～24 | 96 | 各席の河を順に格納。負数はツモ切り、0は空き枠 |

牌種コードは通常34種と赤5萬・赤5筒・赤5索の計37種です。行動種別は配牌・ツモ・打牌・鳴き・リーチ・ドラ追加です。

### 補助CSVの236列

| 列 | 数 | 内容 |
|---|---:|---|
| state_index, oya, game_wind, hand_number | 4 | 対局内の局面番号、親の席、場風、局の番号 |
| p{p}_meld_count, p{p}_chi_count, p{p}_pon_count, p{p}_daiminkan_count, p{p}_ankan_count, p{p}_kakan_count, p{p}_nuki_count | 28 | 各席の鳴き総数と種類別回数。抜きの列は共通形式用 |
| p{p}_meld{s}_type, p{p}_meld{s}_from_rel, p{p}_meld{s}_tile1～4（s=1～4） | 96 | 各席の副露・槓ごとの種別、鳴いた相手の相対席、構成牌 |
| p{p}_disc_called_1～24 | 96 | 各河の牌が鳴かれたかどうか |
| p{p}_ippatsu, p{p}_double_riichi, p{p}_seat_wind | 12 | 各席の一発・ダブルリーチ・自風 |

### 管理ファイルの項目

| ファイル | 項目 |
|---|---|
| selected_players.csv | player, achievement_date, battles_at_listing, zip_url, zip_filename, source_page, local_path, status, size_bytes, sha256, error |
| logs_manifest.csv | game_id, game_date, source_player, source_zip, member, reason |
| selected_games.csv | game_id, game_date, source_player, source_zip, member, csv_file, meta_file, training_rows |
| conversion_errors.csv | game_id, error |
| 対局ごとのdone.json | schema_version, fingerprint, rows, csv_size, meta_size |
| summary.json | schema_version, layout, workers, mode, seed, rank_policy, check_log, date_rule, requested_players, max_games, selected_players, discovered_logs, exclusion_counts, candidate_games_after_date_filter, converted_games, training_rows, row_unit, conversion_errors, full_reference_written, contains_hidden_hands, output_dir |

### 学習に渡す情報

1局面を4観測席×3対象席の12通りに展開します。入力は公開情報・自分の手牌・観測席と対象席の計578列です。入力のvisible_1～37には公開牌に加えて自分の手牌を数えます。他家のツモ牌は値を伏せ、target_tile_is_hiddenで非公開を表します。対局ID・対局日・元プレイヤー・元ZIP・state_indexは管理用です。正解は相手の伏せた手牌を赤牌込みの34種類にまとめた枚数です。

## 公開牌の枚数を利用する改良モデル

各牌種の予測範囲を0枚から残り枚数までに制限します。残り枚数は4枚から公開牌と自分の手牌を引いて求め、赤牌は通常牌に合算します。相手の副露数とツモ直後かどうかで伏せ牌の標準枚数を見積もり、残り牌の均等配分を出発点にします。小さなネットワークが捨て牌などの情報からその予測を補正します。ドロップアウトとL2正則化を使います。出力は期待枚数で、合法な手牌の組み合わせを生成する機能はありません。

比較は両モデルが学習に使っていない同一対局・同一行に対して行います。

    python moderu_jikko.py --name miekazu_01 --compare jikken_02

モデルの計算はKerasの標準演算で構成しています。
参考 https://keras.io/api/ops/numpy/

同一評価の結果（8対局・12,000行）

- jikken_02のMAE 0.491574
- miekazu_01のMAE 0.461566
- 改良モデルのRMSE 0.595195
- 自宅用ZIPはmochikaeri/miekazu_01.zip


## 保存済みZIPの全日付を変換して学習する

この手順は達成前の牌譜も使用します。全件CSVは数百GiBになる見込みです。
Dドライブに保存する例です。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
python paifu_csv_henkan.py --rank-policy all --output-dir /mnt/d/tenhou_converted_all --workers 8 --resume
```

変換完了後、まず300対局を抽出して学習・評価します。

```bash
python aite_tehai_gakushu.py --rank-policy all --input-dir /mnt/d/tenhou_converted_all --mode l --max-games 300 --max-train 12000 --max-test 3000 --epochs 10 --name all_01
python moderu_jikko.py --name all_01
```

全対局を抽出候補にするには --max-games 300 を外します。
学習・評価の行数上限はそのまま適用され、保存された全行で学習する指定ではありません。
設定を変える場合はモデル名を変え、変換済みCSVは再利用します。


## 他家3人の同時推定

既存CSVと保存済みモデルを使い、打牌直後の同じ局面で3方式を比較します。

```bash
python moderu_jikko.py --name miekazu_01 --action douji --cases 256
```

各人の伏せ牌の枚数を揃えた上で、残り牌に比例した配分・モデルの個別推定・3人で残り枚数を共有する同時推定を比較します。出力は `douji_hyoka/metrics.json`（全体・対局別）と `cases.csv`（局面別）です。
初回の評価は8対局・256ケースでした。個別推定の共有上限違反は19ケース、同時推定では0ケース。一致枚数は1人あたり約0.003枚増加しました。
河の直近の手出しやツモ切り、副露前後の打牌は、既存CSVから追加特徴量を作れます。新しい牌譜情報を保存する場合に変換コードとCSVの更新を検討します。


## 好きな局面で他家3人の候補牌を表示する

モデルは自分で学習して名前を付けます。既存CSVを再利用できます。

```bash
python aite_tehai_gakushu.py --mode l --name watashi_01
python moderu_jikko.py --name watashi_01 --action sannin --observer 0 --state-index 30 --top 5
```

observerは自分の席番号（0～3）、state-indexは対局内の局面番号（0始まり）、topは相手1人あたりの表示牌種数です。対局IDの省略時は保存済みの最初の評価対局を読みます。

```bash
python moderu_jikko.py --name watashi_01 --action sannin --game-id 2011020613gm-00a9-0000-3774f8d1 --observer 2 --state-index 100 --top 5
```

対局IDは変換先のselected_games.csvにあります。別の変換フォルダを使う場合は --input-dir で指定します。
候補順位は予測保有枚数に基づきます。打牌直後は3人の残り枚数を共有した整数の推定手牌も表示します。
--show-answer を付けると候補牌の実際の枚数を比較用に表示します。
この機能は変換済み牌譜の局面を対象にします。持ち帰り用ZIPの固定サンプルでは従来の --action yosoku を使用します。


## 評価のランダム抽出と数値表示

```bash
python moderu_jikko.py --name miekazu_01 --action hyoka
```

評価専用に確保した対局から毎回新しいseedで行を重複なしで抽出します。既定の行数上限はモデル保存時の設定です。評価対局自体も減らして抽出する場合は `--test-games`、行数を変える場合は `--test-rows` を指定します。

```bash
python moderu_jikko.py --name miekazu_01 --action hyoka --test-games 4 --test-rows 3000 --eval-seed 42
```

同じモデル・元データ・抽出条件でseedを固定すると同じ行を再現します。比較対象を `--compare jikken_02` で指定すると両モデルを同じ行で評価します。学習対局との重複を検査します。
結果は `hyoka/metrics.json` に保存され、端末にはMAE、MSE、RMSE、枚数一致率、保有ありの適合率・再現率・F1などが出ます。後半4指標は予測枚数を四捨五入した値で計算します。枚数一致率には0枚同士の一致も含みます。
`--action douji` でも `--eval-seed` と `--test-games` を使用できます。評価する局面の上限は `--cases` で指定します。
