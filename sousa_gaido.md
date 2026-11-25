# 天鳳AI 操作ガイド

| ファイル | 役割 |
|---|---|
| paifu_ikkatsu_hozon.py | 天鳳から牌譜ZIPを一括保存 |
| paifu_csv_henkan.py | MJLOGを451列の通常CSVへ変換。並列・途中再開に対応 |
| aite_tehai_gakushu.py | 冒頭でパラメータを設定し、名前付きTensorFlowモデルを構築 |
| tehai_moderu_seisei.py | Lモードで相手手牌推定モデルを学習（既定名 miekazu_02） |
| moderu_jikko.py | モデル名を指定して評価、または1局面の予測を実行 |

## 準備

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
```

## 現在のPIFW比較に使うモデル

学習済みの39対局版3モデルは `tenhou_data/models/39_games` に保存しています。

| フォルダ | 収録モデル |
| --- | --- |
| `39_games`（7対局を追加） | `pifw_mse_achievement_day`、`pifw_weighted_negative_achievement_day`、`pifw_hard_negative_achievement_day` |

Ubuntuでは従来どおり、評価にはフォルダ名を付けずモデル名を指定します。

```bash
python pifw_moderu_seisei.py compare --names pifw_mse_achievement_day \
  pifw_weighted_negative_achievement_day pifw_hard_negative_achievement_day
```

旧モデルの `miekazu_01`、`miekazu_02`、`watashi_01`、`pifw_01`、`pifw_02_recent` を使う過去の例は、保管時点のモデルを復元してから実行してください。例えば `miekazu_01` は次のように復元できます。

```bash
git restore --source=6893d025 -- tenhou_data/models/miekazu_01
```

### Ubuntuで1局面の答え合わせを表示する

リポジトリのルートで、仮想環境を有効にして実行します。

```bash
python pifw_moderu_seisei.py demo
python pifw_moderu_seisei.py demo --seed 42 --name pifw_mse_achievement_day
```

評価用の未学習対局から、4人の河が合計24枚以上ある打牌後の1局面を選び、正解の手牌・予想の手牌・予想的中率を相手ごとに表示します。的中率はPIFWの表示基準に沿って出した予想牌のうち正解と一致した枚数の割合です。同じseedと観測席で別モデルを指定すると同じ局面で比較できます。局面を直接指定する場合は出力の対局IDと局面番号を `--game-id` と `--state-index` に渡します。

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

端末にはモデルMAE、平均予測MAE、誤差削減率、保有ありのF1を表示します。
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

既存CSVを使って新しいモデルを学習できます。既定の保存名は miekazu_02 です。別名で学習する場合は --name を指定します。

```bash
python tehai_moderu_seisei.py
python moderu_jikko.py --name miekazu_02 --action sannin --observer 0 --min-probability 40
```

observerは自分の席番号（0～3）、state-indexは対局内の局面番号（0始まり）です。局面番号を指定した場合は、その局面を使います。局面番号を省略すると、残りツモが40枚以下の打牌直後からランダムに選びます。対局IDを指定した場合はその対局内、省略した場合は保存済みの評価対局から選びます。残りツモ数は `--min-remaining` と `--max-remaining`、局面番号は `--min-state-index` と `--max-state-index` で範囲を指定できます。固定局面を指定する場合も、明示した範囲の条件を満たす必要があります。同じランダム選択を再現するには `--eval-seed` を指定します。

```bash
python moderu_jikko.py --name miekazu_02 --action sannin --observer 0 --min-remaining 10 --max-remaining 35 --min-state-index 60 --max-state-index 300 --min-probability 45 --max-probability 65
```

対局IDは変換先のselected_games.csvにあります。別の変換フォルダを使う場合は --input-dir で指定します。
各相手の正解と予想の伏せ牌を表示します。「その相手が1枚以上持つ」推定確率が40%以上の牌を、確率の高い順にすべて表示します。`--min-probability 45 --max-probability 65` なら45〜65%の範囲です。確率の上限だけを指定した場合、下限は1%です。各範囲は境界値を含みます。`--top 5` を指定した場合は、基準を満たす牌種のうち、各相手の上位5種類まで表示します。残りが r 枚でモデルの予測枚数が m 枚の場合、各1枚の保有を独立と仮定し、確率を 1 − (1 − m/r)^r と近似します。残り0枚の場合は0%です。表示枚数は予測枚数を四捨五入し、候補に選ばれた牌は最低1枚とします。的中率は表示した予想枚数に対する一致枚数の割合です。同じ牌は正解と予想の少ない方の枚数まで一致と数えます。3人で残り枚数を共有する配分の評価には --action douji を使います。
この機能は変換済み牌譜の局面を対象にします。持ち帰り用ZIPの固定サンプルでは従来の --action yosoku を使用します。


## 評価のランダム抽出と数値表示

```bash
python moderu_jikko.py --name miekazu_01 --action hyoka
```

評価専用に確保した対局のうち、残りツモが40枚以下の局面から毎回新しいseedで行を重複なしで抽出します。`--min-remaining 10 --max-remaining 35` のように残りツモ数の範囲を指定できます。`--max-remaining 70` で全局面を対象にします。既定の行数上限はモデル保存時の設定です。評価対局自体も減らして抽出する場合は `--test-games`、行数を変える場合は `--test-rows` を指定します。

```bash
python moderu_jikko.py --name miekazu_01 --action hyoka --test-games 4 --test-rows 3000 --eval-seed 42
```

同じモデル・元データ・抽出条件でseedを固定すると同じ行を再現します。比較対象を `--compare jikken_02` で指定すると両モデルを同じ行で評価します。学習対局との重複を検査します。
詳細な指標は `hyoka/metrics.json` に保存します。保有ありのF1は推定確率が `--min-probability` 以上の全牌種から計算します。 `--action douji` も指定した残りツモ数の範囲内の打牌を対象にします。
`--action douji` でも `--eval-seed` と `--test-games` を使用できます。評価する局面の上限は `--cases` で指定します。


## 別端末で実行する

修正をコミット・pushし、別端末でclone先を更新します。保存済みモデルに学内の絶対パスがあっても、clone先のtenhou_data以下からCSVを探します。CSVを別フォルダに置いた場合は --input-dir で変換結果のフォルダを指定します。

```bash
python moderu_jikko.py --name miekazu_01 --action info
python moderu_jikko.py --name miekazu_01 --action sannin
```

infoはPythonだけで保存済み結果を表示します。予測や評価にはTensorFlowとNumPyが必要です。大容量のCSVを持ち帰らない場合は、自宅用ZIPを使います。


## PIFWの4人麻雀で他家3人の手牌を継続推定する

既定の学習済みモデルは `tenhou_data/models/39_games/pifw_mse_achievement_day/` にあります。配牌後から終局前まで、指定した席の視点で推論できます。自分の手番以外も対象です。PIFWの自動ツモなど、一つの着手コマンド内に複数の処理がある場合は、コマンド確定後に観測します。

情報源は、自分の手牌、4人の河・副露・立直・伏せ牌の枚数、公開ドラ表示牌です。他家がツモった牌の種類や山の中身は入力に含めません。鳴かれた打牌は河から除き、副露で1回だけ数えます。牌種は1〜34で、天鳳の赤牌は通常の5に合算します。

### 対局中の公開情報が変わるたびに予測する

PIFWのローカルコンソールで `mahjong_watch 2P` を実行すると2P視点の観測が始まります。観測が変わるたびにPIFWがJSONを1行出力し、`--stream` は学習済みモデルを一度読み込んで順に予測します。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate

{
  printf '%s\n' \
    'mahjong_watch 2P' \
    'clear_board gm 9 bs 1 1 pn 4 fp 0 ai 2 lv 1 dp 1 mo 0 md 2'
  while IFS= read -r command
  do
    printf '%s\n' "$command"
    [ "$command" = quit ] && break
  done
} | (cd ~/AI-1009/pifw_engine && bin/pifw -ncsx) \
  | sed -un '/^{"schema":"pifw-mahjong-observation-v2",/p' \
  | python pifw_tehai_yosoku.py --stream --allocation joint
```

この例では先手は1Pで、観測席は2Pです。最初の予測に続けて、同じ端末に `genmove 1P` を入力すると先手のAI着手を進めます。`play` で手動着手もできます。`quit` で終了します。別端末でソースからエンジンを作るには `(cd ~/AI-1009/pifw_engine && make -j4)` を実行します。Unityで使う場合は「Unityの4人麻雀で手牌予測を表示する」を参照してください。

1局面だけ取得するコマンドは `mahjong_public_observation 2P` です。そのJSONを `python pifw_tehai_yosoku.py` の標準入力に渡すか、`--input` でファイルを指定できます。出力の席番号は0始まりです。

`expected_counts` は相手ごとの34牌種の期待枚数です。たとえば `0.8` は期待0.8枚を表します。`assigned_counts` は、その時点の各相手の伏せ牌枚数と、3人が共有する残り牌の上限を満たす整数の仮手牌です。`unassigned_counts` は配らなかった未見牌です。入力の `concealed_sizes` には、各席のツモによる増加や副露による減少が反映されます。

### 再評価と特徴量の変更

```bash
python pifw_moderu_seisei.py evaluate --name pifw_mse_achievement_day
python pifw_moderu_seisei.py train --name pifw_02_recent --features recent --seed 42 --epochs 12
```

再評価では保存済みの評価用対局を使い、全体とイベント別のMAEを表示します。`counts` は牌種別枚数を中心とした390特徴量です。`recent` は河に残る直近3枚の順序を加えた798特徴量です。学習した別モデルは `--model-dir` で指定できます。旧版 `pifw_01` は従来の自分の打牌前の局面で利用できます。

- `--allocation joint` はモデルの期待枚数に沿うよう3人分を同時配分します。
- `--allocation proportional` は同じ配分処理に残り牌比例の期待枚数を渡します。
- `--allocation uniform --seed 42` は未見牌を重複なしで無作為配分します。
- `--allocation none` は期待枚数だけを出します。

特徴量は `pifw_tehai_yosoku.py` の `encode_observation`、ネットワークは `pifw_moderu_seisei.py` の `build_model`、仮手牌の配分は `pifw_tehai_yosoku.py` の `assign_hands` で変更できます。Pythonから複数局面で予測するときは `HandPredictor` を一度作って `predict` を呼び出します。SavedModelの `serving_default` は `float32` の `features` を受け取り、`expected_counts` を返します。特徴量の順序は `model.json` に保存しています。

## Unityの4人麻雀で手牌予測を表示する

`~/AI-1009/pifw_gui` でPIFWエンジンに接続し、4人麻雀を開始します。手牌予測欄は公開局面の更新に合わせて変わります。Pythonは既定で `pifw_mse_achievement_day` を使います。他の39対局版で対局用JSONを試す場合は、`python pifw_live.py --name モデル名` に公開局面JSONを1行ずつ渡します。

手牌予測は初期状態で閉じています。右下の「手牌予測」から中央のパネルを開き、相手の席を選択すると、牌の画像・予想枚数・保有の推定割合（仮手牌で1枚以上あった割合）を表示します。予測パネルを閉じても、相手の手牌列の外側にある7段階の危険度マークは盤面の更新に合わせて切り替わります。新しい局面の応答を待っている間はマークに「更新中」と表示します。

- GUIが使うのは自分の手牌、公開された河・副露・立直・ドラ、各席の伏せ牌枚数です。他家の牌種はGUIの予測処理で読み取りません。正解牌を使うMAEなどは、未学習牌譜を使うオフライン評価にあります。
- モデルは34牌種それぞれの期待枚数を出します。仮手牌では同じ牌を複数枚割り当てられます。3人で同種の残り枚数を共有し、表示する候補は期待枚数0.35枚以上かつ仮手牌に入った牌です。 対局用JSONの `count_hypotheses` は牌種ごとの0・1・2・3・4枚の仮説割合です。16組は予測枚数で重み付けした未見牌から重複なしでサンプリングします。
- 相手ごとに仮手牌16組を評価し、向聴数、残っていそうな有効牌、公開立直、ロン可能な仮説を組み合わせた危険度を表示します。手の狙いは仮手牌と副露から頻出した役候補です。
- 「最短あと何自摸」は各仮説の向聴数に1を足した中央値です。自摸の概算は、仮説の未知牌と有効牌から求めた待ち時間が、以降の進行でも同じと仮定した目安です。鳴きや他家の打牌は省略します。十分な仮説が揃ったときだけ表示し、残りの自摸回数の目安を超える場合はその旨を示します。
- 配牌時は0点です。新しい打牌が直前の保有仮説と自摸の可能性に照らして予想しにくいときは、不確実性を加算します。危険度が低めの局面では負の点数で示します。負の点数も放銃の少なさを示す数値ではありません。立直や高い危険度を優先して表示します。

| 表示する点 | マークの段階 |
| --- | --- |
| −11以下 | 1 |
| −10〜−1 | 2 |
| 0〜10 | 3 |
| 11〜20 | 4 |
| 21〜30 | 5 |
| 31〜39 | 6 |
| 40以上 | 7 |

### 役・フリテンを考慮した打牌別のロン仮説

手元の牌を通常打牌した場合について、仮手牌でその牌にロンできた割合を表示します。役なしやフリテンを除きます。役の判定には MahjongRepository の mahjong を使用します。河底は公開の山の枚数から判定し、搶槓や自摸に固有の役は通常打牌の対象外です。

画面の割合は探索した仮説に対する集計です。牌譜の実手牌との比較や誤差補正はオフラインの `pifw_danger_calibrate.py` に残しています。

役判定ライブラリ [MahjongRepository mahjong](https://mahjongrepository.github.io/mahjong/)

### 打牌・副露の傾向による枚数補正

39対局版の3モデルには、対象者の河にある同じ牌や近い数牌、色の偏り、副露の色、打牌の進行度を使う補正を組み込みました。追加係数は学習39対局の教師ラベルで求め、別の8対局で評価しました。序盤の検証範囲は4人の河にある牌が24枚以下の4,408局面です。

`model.json` の `discard_pattern` に学習済み係数と導入前後の評価値があります。通常の `compare` やPIFWの推論はこの補正を使います。保存済みのTensorFlow本体の直接呼び出しは本体の出力を返します。

## 非保有牌の誤予測を減らす損失関数の比較

同じ牌譜、390特徴量、39学習対局・8評価対局、seed 42、12エポック、幅64、バッチ128、学習率0.001で、損失関数を変えて比較します。対局中のモデル入力は公開情報と自分の手牌です。相手の実手牌は訓練時の正解ラベルと評価で使います。

| 評価時のモデル名 | 学習の違い |
| --- | --- |
| `pifw_mse_achievement_day` | 標準MSE |
| `pifw_weighted_negative_achievement_day` | MSEに、実際には持っていない牌の二乗誤差の平均を0.25倍して加える |
| `pifw_hard_negative_achievement_day` | MSEに、実際には持っていない牌のうち期待枚数が高い最大5種の二乗誤差の平均を0.25倍して加える |

32対局版の3モデルは比較当時のZIPに保管しています。再現する場合はZIP内のモデルを `tenhou_data/models/32_games` に復元してください。

公開牌で残り0枚と分かる牌には、既存の出力制約が効きます。負例の追加損失は、その制約を掛けても正の期待枚数が出ている非保有牌に対して計算します。通常のMSEにも非保有牌の誤差は含まれます。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
python pifw_moderu_seisei.py compare
```

`compare` は共通の評価局面でMAE、MSE、保有判定の適合率・再現率・F1、相手1人あたりの予測総枚数を出します。保有判定は期待枚数0.5枚以上です。仮手牌の一致枚数は共通の256観測で3人同時配分した結果です。`--names` で比較対象を指定できます。同じ特徴量・対局分割・主要な学習条件のモデルが対象です。

別名で追試する場合は、`train` で `--loss weighted_negative` または `--loss hard_negative` を使います。`--negative-weight` は追加損失の係数です。`--hard-k` は `hard_negative` で重視する非保有牌の最大種類数です。

### 天鳳位の達成日当日の牌譜を学習に追加する

`--include-achievement-day` は元ZIPを調べ、対象者が達成日当日に天鳳位（牌譜の段位コード20）で開始した四人打ちの対局を学習側に追加します。既定の `converted/after` を分割して評価対局を固定してから、追加対局を一時変換します。追加した対局IDはモデルの `model.json` の `achievement_day_train_game_ids` に保存されます。

```bash
python pifw_moderu_seisei.py train --name pifw_mse_achievement_day_retrial \
  --include-achievement-day --loss mse --epochs 12 --seed 42
```

既定の変換データと元ZIP30件で、学習32対局に当日の7対局を追加できます。比較は同じ未学習8対局で行います。再学習する場合は `--name` に未使用のモデル名を指定してください。

### 九段・十段の牌譜を追加する標準MSE実験

既存の学習39対局に、元ZIPから達成日前の九段19対局と十段20対局を加えます。段位は牌譜の対局開始時の記録を使用し、同卓の4人全員が七段以上の対局を対象とします。元ZIPの対象者ごとに最大2対局まで選び、追加分の対局IDと開始段位は `model.json` の `high_dan_train_games` に保存されます。評価8対局のIDは固定し、追加対局との重複を除きます。

```bash
python pifw_moderu_seisei.py train --name pifw_mse_high_dan_78 \
  --include-achievement-day --add-high-dan-games 39 --epochs 12
python pifw_moderu_seisei.py compare --allow-different-training-games \
  --names pifw_mse_achievement_day pifw_mse_high_dan_78
```

新モデルは `tenhou_data/models/78_games` に保存されます。標準MSEと打牌傾向補正は学習側の78対局で推定し直します。同名で保存済みの場合は比較コマンドを実行してください。
