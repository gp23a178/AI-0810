import csv
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import io
import gzip

# -------------------------------------------------------------
# 人間向けTXT用：天鳳の牌ID (0〜135) を日本語名に変換する関数
# TXT版にはツモの要素がないため、ツモ切りか手出しかの判定は不要
# -------------------------------------------------------------
def tile_id_to_str(tile_id):
    if tile_id is None:
        return ""
    tile_id = int(tile_id)
    if tile_id == 16: return "赤5萬"
    elif tile_id == 52: return "赤5筒"
    elif tile_id == 88: return "赤5索"
    kinds = ["萬", "筒", "索"]
    honors = ["東", "南", "西", "北", "白", "發", "中"]
    if tile_id < 108:
        suit = tile_id // 36
        num = ((tile_id % 36) // 4) + 1
        return f"{num}{kinds[suit]}"
    else:
        idx = (tile_id - 108) // 4
        return honors[idx]

# XMLパースエラーを防ぐためのテキスト浄化関数
def clean_xml_string(xml_text):
    text = xml_text.replace('\x00', '').lstrip('\ufeff').strip()
    if text.startswith('%3C') or text.startswith('%3c'):
        text = urllib.parse.unquote(text)
    text = re.sub(r'<\?xml[^>]*\?>', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE).strip()
    return text

# -------------------------------------------------------------
# .mjlog の内容を「人間用TXT」と「機械学習用CSV」に解析する関数
# -------------------------------------------------------------
def parse_mjlog_content(xml_text, file_name):
    xml_text = clean_xml_string(xml_text)
    
    try:
        root = ET.fromstring(f"<root>{xml_text}</root>")
    except ET.ParseError:
        xml_text_clean = re.sub(r'&(?![a-zA-Z]+;)', '&amp;', xml_text)
        try:
            root = ET.fromstring(f"<root>{xml_text_clean}</root>")
        except ET.ParseError as e:
            print(f"\nスキップ: {file_name} の解析に失敗しました ({e})")
            return "", []

    txt_lines = []
    csv_rows = []
    player_names = ["東家", "南家", "西家", "北家"]
    
    un_tag = root.find(".//UN")
    if un_tag is not None:
        for i in range(4):
            attr = f"n{i}"
            if attr in un_tag.attrib:
                player_names[i] = urllib.parse.unquote(un_tag.attrib[attr])

    txt_lines.append(f"対局データ分析ログ: {file_name}")
    txt_lines.append(f"対局者: {', '.join(player_names)}")
    txt_lines.append("=" * 60 + "\n")

    current_round_num = -1
    current_honba = -1
    last_drawn_tile = {0: -1, 1: -1, 2: -1, 3: -1}
    step_count = 0

    # CSV用の空行テンプレート
    base_row = {
        "round": -1, "honba": -1, "step": -1, "actor": -1, "action": -1, 
        "tile": -1, "tsumogiri": -1
    }
    for p in range(4):
        for i in range(14):
            base_row[f"p{p}_hp_{i}"] = -1

    for elem in root.iter():
        tag = elem.tag
        if tag == "INIT":
            step_count = 0
            seed = elem.attrib.get("seed", "").split(",")
            current_round_num = int(seed[0]) if len(seed) > 0 else 0
            current_honba = int(seed[1]) if len(seed) > 1 else 0
            dora_id = int(seed[5]) if len(seed) > 5 else -1
            
            # --- TXT処理 ---
            field = "東" if current_round_num < 4 else "南"
            kyoku = (current_round_num % 4) + 1
            txt_lines.append(f"--- 【{field}{kyoku}局 {current_honba}本場】 ---")
            txt_lines.append(f"ドラ表示牌: {tile_id_to_str(dora_id)}")
            
            # --- CSV処理 ---
            row = base_row.copy()
            row["round"] = current_round_num
            row["honba"] = current_honba
            row["step"] = step_count
            row["action"] = 0
            row["tile"] = dora_id
            
            for p in range(4):
                hai_str = elem.attrib.get(f"hai{p}", "")
                tiles_str = [tile_id_to_str(tid) for tid in hai_str.split(",") if tid]
                txt_lines.append(f"  {player_names[p]} 配牌: {' '.join(tiles_str)}")
                
                tiles_int = [int(tid) for tid in hai_str.split(",") if tid]
                for i in range(14):
                    row[f"p{p}_hp_{i}"] = tiles_int[i] if i < len(tiles_int) else -1
                    
            txt_lines.append("-" * 40)
            csv_rows.append(row)

        elif len(tag) >= 2 and tag[0] in ["T", "U", "V", "W", "D", "E", "F", "G"] and tag[1:].isdigit():
            action_code = tag[0]
            tile_id = int(tag[1:])
            step_count += 1
            
            row = base_row.copy()
            row["round"] = current_round_num
            row["honba"] = current_honba
            row["step"] = step_count
            row["tile"] = tile_id
            
            if action_code in ["T", "U", "V", "W"]:
                p_idx = ["T", "U", "V", "W"].index(action_code)
                last_drawn_tile[p_idx] = tile_id
                
                # ★修正箇所1: TXTにも「ツモ」の情報を出力する
                txt_lines.append(f"  [{step_count:03d}] {player_names[p_idx]}: ツモ {tile_id_to_str(tile_id)}")
                
                row["actor"] = p_idx
                row["action"] = 1
                csv_rows.append(row)
                
            elif action_code in ["D", "E", "F", "G"]:
                p_idx = ["D", "E", "F", "G"].index(action_code)
                is_tsumogiri = (last_drawn_tile[p_idx] == tile_id)
                tsumo_flag_int = 1 if is_tsumogiri else 0
                
                tsumo_flag_str = "ツモ切り" if is_tsumogiri else "手出し"
                txt_lines.append(f"  [{step_count:03d}] {player_names[p_idx]}: 打 {tile_id_to_str(tile_id)} ({tsumo_flag_str})")
                
                row["actor"] = p_idx
                row["action"] = 2
                row["tsumogiri"] = tsumo_flag_int
                csv_rows.append(row)
                
                last_drawn_tile[p_idx] = -1

        elif tag == "N":
            p_idx = int(elem.attrib.get("who", 0))
            meld_id = int(elem.attrib.get("m", -1))
            step_count += 1
            
            # ★修正箇所2: TXTにも「鳴きID」を出力し、CSVと情報を一致させる
            txt_lines.append(f"  [{step_count:03d}] {player_names[p_idx]}: 鳴き(副露) ID:{meld_id}")
            
            row = base_row.copy()
            row["round"] = current_round_num
            row["honba"] = current_honba
            row["step"] = step_count
            row["actor"] = p_idx
            row["action"] = 3
            row["tile"] = meld_id
            csv_rows.append(row)
            
        elif tag == "REACH":
            p_idx = int(elem.attrib.get("who", 0))
            step = elem.attrib.get("step", "")
            if step == "1":
                txt_lines.append(f"  [*] {player_names[p_idx]}: リーチ宣言")
                
                row = base_row.copy()
                row["round"] = current_round_num
                row["honba"] = current_honba
                row["step"] = step_count
                row["actor"] = p_idx
                row["action"] = 4
                csv_rows.append(row)
                
        elif tag == "DORA":
            new_dora_id = int(elem.attrib.get("hai", -1))
            txt_lines.append(f"  [*] 新ドラ追加: {tile_id_to_str(new_dora_id)}")
            
            row = base_row.copy()
            row["round"] = current_round_num
            row["honba"] = current_honba
            row["step"] = step_count
            row["action"] = 5
            row["tile"] = new_dora_id
            csv_rows.append(row)

    if txt_lines:
        txt_lines.append("\n")
        return "\n".join(txt_lines), csv_rows
    else:
        return "", []

# -------------------------------------------------------------
# ZIP抽出メイン処理
# -------------------------------------------------------------
def process_mjlogs_from_zip(input_zip, output_txt_zip, output_csv_zip):
    with zipfile.ZipFile(output_txt_zip, 'w', zipfile.ZIP_DEFLATED) as txt_zip, \
         zipfile.ZipFile(output_csv_zip, 'w', zipfile.ZIP_DEFLATED) as csv_zip, \
         zipfile.ZipFile(input_zip, 'r') as in_zip:

        target_files = [f for f in in_zip.infolist() 
                        if (f.filename.endswith(".mjlog") or f.filename.endswith(".xml")) 
                        and os.path.basename(f.filename)]
        total_files = len(target_files)
        
        if total_files == 0:
            print("処理対象のファイルが見つかりませんでした。")
            return

        processed_count = 0
        
        fieldnames = ["round", "honba", "step", "actor", "action", "tile", "tsumogiri"]
        for p in range(4):
            for i in range(14):
                fieldnames.append(f"p{p}_hp_{i}")

        print(f"処理対象: {total_files} ファイル")

        for index, file_info in enumerate(target_files, start=1):
            base_filename = os.path.basename(file_info.filename)
            raw_bytes = in_zip.read(file_info)
            
            if raw_bytes.startswith(b'\x1f\x8b'):
                try:
                    raw_bytes = gzip.decompress(raw_bytes)
                except Exception as e:
                    print(f"\n解凍エラー: {base_filename} ({e})")
                    continue
            
            xml_content = raw_bytes.decode("utf-8", errors="ignore")
            txt_content, csv_rows = parse_mjlog_content(xml_content, base_filename)
            
            if txt_content and csv_rows:
                txt_filename = os.path.splitext(base_filename)[0] + ".txt"
                txt_zip.writestr(txt_filename, txt_content)
                
                csv_filename = os.path.splitext(base_filename)[0] + ".csv"
                csv_buffer = io.StringIO()
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
                csv_zip.writestr(csv_filename, '\ufeff' + csv_buffer.getvalue())
                
                processed_count += 1
            
            print(f"\r進捗: {index}/{total_files} 件 完了 (成功: {processed_count}件)", end="")

    print(f"\n\n抽出・変換処理が完了しました。")
    print(f"テキスト(人間用ログ): {output_txt_zip} に保存されました")
    print(f"CSV(機械学習用行列): {output_csv_zip} に保存されました")

if __name__ == "__main__":
    INPUT_ZIP = "Son.zip"
    TXT_ZIP_OUT = "Son_t.zip"
    CSV_ZIP_OUT = "Son_c.zip"
    
    if os.path.exists(INPUT_ZIP):
        process_mjlogs_from_zip(INPUT_ZIP, TXT_ZIP_OUT, CSV_ZIP_OUT)
    else:
        print(f"エラー: 指定されたZIPファイル '{INPUT_ZIP}' が見つかりません。")