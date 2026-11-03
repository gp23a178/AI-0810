# Pythonファイル名の対応表

変更範囲：`~/WSL/AI-0810` 内の研究コード。仮想環境とライブラリは対象外。

| 旧ファイル名 | 新ファイル名 |
|---|---|
| `train_hand_predictor_safe.py` | `aite_tehai_gakushu.py` |
| `convert_tenhou_safe.py` | `paifu_csv_henkan.py` |
| `tenhou_ikkatu.py` | `paifu_ikkatsu_hozon.py` |
| `test_tenhou_safety.py` | `joho_more_kakunin.py` |
| `過去のログ/CSV2.py` | `過去のログ/kyu_suchi_gakushu_02.py` |
| `過去のログ/Dual-2.py` | `過去のログ/kyu_paifu_henkan_kaitei_02.py` |
| `過去のログ/Dual3.py` | `過去のログ/kyu_paifu_henkan_betsu_03.py` |
| `過去のログ/CSV_3.py` | `過去のログ/kyu_suchi_gakushu_betsu_03.py` |
| `過去のログ/Dual2.py` | `過去のログ/kyu_paifu_henkan_betsu_02.py` |
| `過去のログ/CSV.py` | `過去のログ/kyu_suchi_gakushu_kihon.py` |
| `過去のログ/CSV1.py` | `過去のログ/kyu_suchi_gakushu_01.py` |
| `過去のログ/CSV_2.py` | `過去のログ/kyu_suchi_gakushu_betsu_02.py` |
| `過去のログ/CSV_1.py` | `過去のログ/kyu_suchi_gakushu_betsu_01.py` |
| `過去のログ/Dual.py` | `過去のログ/kyu_paifu_henkan_kihon.py` |
| `過去のログ/Dual-1.py` | `過去のログ/kyu_paifu_henkan_kaitei_01.py` |

## 現在使うコマンド

```bash
cd ~/WSL/AI-0810
python3 paifu_csv_henkan.py --help
python3 aite_tehai_gakushu.py --help
python3 joho_more_kakunin.py
```

牌譜を追加ダウンロードするコードは `paifu_ikkatsu_hozon.py` です。
学習の実行には仮想環境のライブラリ導入が必要です。詳細は README_tenhou_safe.md を参照してください。

`jikken_settei.py`：S/M/Lの人数・対局数・学習/評価行数・学習回数をまとめて設定。
`backups/` は削除済みです。学習は既存の `~/WSL/venv` のTensorFlowを使用します。
