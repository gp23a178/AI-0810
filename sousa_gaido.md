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

学習済みモデルは `tenhou_data/models/pifw_02/` にあります。配牌後から終局前まで、指定した席の視点で推論できます。自分の手番以外も対象です。PIFWの自動ツモなど、一つの着手コマンド内に複数の処理がある場合は、コマンド確定後に観測します。

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
python pifw_moderu_seisei.py evaluate --name pifw_02
python pifw_moderu_seisei.py train --name pifw_02_recent --features recent --seed 42 --epochs 12
```

再評価では保存済みの評価用対局を使い、全体とイベント別のMAEを表示します。`counts` は牌種別枚数を中心とした390特徴量です。`recent` は河に残る直近3枚の順序を加えた798特徴量です。学習した別モデルは `--model-dir` で指定できます。旧版 `pifw_01` は従来の自分の打牌前の局面で利用できます。

- `--allocation joint` はモデルの期待枚数に沿うよう3人分を同時配分します。
- `--allocation proportional` は同じ配分処理に残り牌比例の期待枚数を渡します。
- `--allocation uniform --seed 42` は未見牌を重複なしで無作為配分します。
- `--allocation none` は期待枚数だけを出します。

特徴量は `pifw_tehai_yosoku.py` の `encode_observation`、ネットワークは `pifw_moderu_seisei.py` の `build_model`、仮手牌の配分は `pifw_tehai_yosoku.py` の `assign_hands` で変更できます。Pythonから複数局面で予測するときは `HandPredictor` を一度作って `predict` を呼び出します。SavedModelの `serving_default` は `float32` の `features` を受け取り、`expected_counts` を返します。特徴量の順序は `model.json` に保存しています。

## Unityの4人麻雀で手牌予測を表示する

`~/AI-1009/pifw_gui` を対応するUnity Editorで開き、PIFWエンジンに接続して**4人麻雀**を開始します。麻雀シーンの右上に「手牌予測」が表示されます。Windows版は `Ubuntu` のWSLで、Linux版は同じ端末で、`~/WSL/venv/bin/python -B ~/WSL/AI-0810/pifw_live.py` を自動起動します。Pythonは対局中に `pifw_02` を1回読み込みます。

- 局面が変わるたび、4人の河・副露・立直・伏せ牌枚数、自分の手牌、公開ドラを使って予測します。更新が重なったときは最新局面を優先します。
- 他家ごとに34牌種の期待枚数を保持します。表示する候補は共通の残り牌制約を満たす仮手牌のうち、期待枚数が0.35枚以上の牌です。括弧は期待枚数、×の後ろは候補枚数です。各家の表示枚数は伏せ牌枚数以下です。
- 各相手のマークは、手元の牌を切った場合の最大の放銃推定値で切り替わります。計算の条件と精度の見方は次節を参照してください。
- 「研究用の答え合わせ」を有効にすると、Unity内で同じ局面の正解と比較し、相手別にMAE、MSE、候補の一致枚数を表示します。実際の他家の手牌はモデルに送信しません。

期待枚数を `p_i`、正解の枚数を `y_i` とすると、MAEは `Σ |p_i−y_i| / 34`、MSEは `Σ (p_i−y_i)² / 34` です。一致枚数は牌種ごとの「表示枚数と正解枚数の小さい方」を合計します。表示基準と探索回数は `pifw_live.py`、マークの色や境界は `MahjongPredictionPanel.cs` で変更できます。打牌AIの着手選択には既存の方式を使います。

### 役・フリテンを考慮した放銃推定

自分の手元にある牌を通常打牌したとき、相手ごとにロンできる見込みを表示します。既定は危険度上位3牌で、「手元の全牌の放銃推定を表示」で全候補を確認できます。相手ごとの最大値でマークが変わります（10%以下、10%超25%未満、25%以上）。相手の手牌がツモ後などで14枚相当の間は、その相手への危険度を保留します。

- 各仮手牌を、公開副露・立直・一発・自風・場風・山の残り枚数と組み合わせます。標準の立直麻雀の役判定には MahjongRepository の mahjong 2.1.0 を使い、喰いタンありで計算します。和了形でも役が無ければロン候補から外し、ドラだけでは役になりません。河底は山の枚数から判定し、搶槓やツモで発生する役は通常打牌の対象外です。現在のPIFW画面が開始する東場・1P親に合わせています。
- 公開された全捨て牌の履歴を保存し、鳴かれた牌もフリテン判定に使います。仮手牌の待ちの一つでも本人が切っていれば全ての待ちをロン不可とします。同巡や立直後の見逃しは、公開打牌後に対局が進んだ履歴を使って仮手牌ごとに推定します。観測できない見逃しの有無は確定しません。
- 牌譜を補正用と検証用に対局単位で分割し、補正用で予測した枚数と正解のRMSEを局面群ごとに測ります。RMSEは仮手牌探索のばらつきに使います。仮手牌から得たロン候補数を `h`、有効探索数を `n`、補正用牌譜の発生率を `p0`、誤差に応じた重みを `m = 16 × (0.5 + RMSE² / 0.10)` として、`(h + m × p0) / (n + m)` を表示します。現物は0%です。立直済みで聴牌形の仮説が得られない場合は牌譜の発生率で補います。
- 対局中に受け取った実際の他家手牌は、任意の答え合わせだけで使います。危険度ワーカーに送るのは自分の手牌と公開情報です。「研究用の答え合わせ」を表示すると、危険度に使う過去のRMSEも確認できます。

補正ファイルを作り直す場合は、既存のvenvで次を実行します。普段のUnity利用時は保存済みファイルを読み込みます。

```bash
cd ~/WSL/AI-0810
source ~/WSL/venv/bin/activate
python -m pip install -r requirements.txt
python pifw_danger_calibrate.py
```

保存先は `tenhou_data/models/pifw_02/danger_calibration.json` です。検証値は、他家が実際にロンを選んだ頻度ではなく、牌譜の実手牌ならその候補でロンが可能だったかを基準にしています。

今回の補正には4対局・640局面、動作検証には別の4対局・96局面を使用しました。現物を除いた候補2,789件のうち、ロン可能だったものは31件です。確率の二乗誤差（Brier score）は探索と補正の組み合わせで0.01050、牌譜の発生率だけで0.01045でした。今回の検証では探索による改善は確認できませんでした。PIFWでの校正は未評価です。

役判定ライブラリ [MahjongRepository mahjong](https://mahjongrepository.github.io/mahjong/)
