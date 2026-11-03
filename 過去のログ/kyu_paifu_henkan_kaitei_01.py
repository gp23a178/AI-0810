import csv
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import io
import gzip
import concurrent.futures

# -------------------------------------------------------------
# 1. 牌ID変換 (0はパディング/空きマス)
# -------------------------------------------------------------
def to_37_id(tile_id):
    if tile_id is None or tile_id == "": return 0
    tid = int(tile_id)
    if tid == 16: return 35
    elif tid == 52: return 36
    elif tid == 88: return 37
    else: return (tid // 4) + 1

def tile_id_to_str(tile_id):
    if tile_id is None or tile_id == "": return ""
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

def clean_xml_string(xml_text):
    text = xml_text.replace('\x00', '').lstrip('\ufeff').strip()
    if text.startswith('%3C') or text.startswith('%3c'):
        text = urllib.parse.unquote(text)
    text = re.sub(r'<\?xml[^>]*\?>', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE).strip()
    return text

# -------------------------------------------------------------
# 2. 天鳳の副露(Nタグ)を37-IDのリストに解読する関数（完全修正版）
# -------------------------------------------------------------
def decode_meld_to_37(m_val):
    if not m_val: return []
    m = int(m_val)
    tiles_37 = []
    
    try:
        if m & 0x0004:  # チー
            hai0, hai1, hai2 = (m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3
            base = (m >> 10) & 63
            color, num = base // 7, base % 7
            tiles_37.append(to_37_id((color * 9 + num) * 4 + hai0))
            tiles_37.append(to_37_id((color * 9 + num + 1) * 4 + hai1))
            tiles_37.append(to_37_id((color * 9 + num + 2) * 4 + hai2))
            
        elif m & 0x0018:  # ポン・加槓
            hai0, hai1, hai2 = (m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3
            base = (m >> 9) & 127
            t = base // 3
            tiles_37.extend([to_37_id(t * 4 + hai0), to_37_id(t * 4 + hai1), to_37_id(t * 4 + hai2)])
            if m & 0x0010:  # 加槓
                hai3 = (m >> 14) & 3
                tiles_37.append(to_37_id(t * 4 + hai3))
                
        elif m & 0x0020:  # 抜きドラ
            tid = (m >> 8) & 255
            tiles_37.append(to_37_id(tid))
            
        else:  # 暗槓・大明槓
            hai0 = (m >> 8) & 255
            t = hai0 // 4
            tiles_37.extend([to_37_id(t * 4), to_37_id(t * 4 + 1), to_37_id(t * 4 + 2), to_37_id(t * 4 + 3)])
    except Exception:
        pass
        
    return [x for x in tiles_37 if 1 <= x <= 37]

# -------------------------------------------------------------
# 3. .mjlog 解析関数
# -------------------------------------------------------------
def parse_mjlog_content(xml_text, file_name):
    xml_text = clean_xml_string(xml_text)
    try:
        root = ET.fromstring(f"<root>{xml_text}</root>")
    except ET.ParseError:
        xml_text_clean = re.sub(r'&(?![a-zA-Z]+;)', '&amp;', xml_text)
        try:
            root = ET.fromstring(f"<root>{xml_text_clean}</root>")
        except ET.ParseError:
            return "", []

    txt_lines = []
    csv_rows = []
    player_names = ["東家", "南家", "西家", "北家"]
    
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == "UN":
            for i in range(4):
                attr = f"n{i}"
                if attr in elem.attrib:
                    player_names[i] = urllib.parse.unquote(elem.attrib[attr])
            break

    txt_lines.append(f"対局データ分析ログ: {file_name}")
    txt_lines.append(f"対局者: {', '.join(player_names)}")
    txt_lines.append("=" * 60 + "\n")

    current_round_num = 0
    current_honba = 0
    last_drawn_tile = {0: -1, 1: -1, 2: -1, 3: -1}
    step_count = 0

    hands = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
    melds = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
    player_discards = {0: [], 1: [], 2: [], 3: []}
    reaches = {0: 0, 1: 0, 2: 0, 3: 0}
    doras = []

    def append_csv_row(actor, action, target_tile_37, tsumogiri=0):
        normalized_actor = actor if 0 <= actor <= 3 else 4
        row = {
            "round": current_round_num, 
            "honba": current_honba, 
            "step": step_count, 
            "actor": normalized_actor, 
            "action": action, 
            "target_tile": target_tile_37, 
            "tsumogiri": tsumogiri
        }
        for i in range(5):
            row[f"dora_{i+1}"] = doras[i] if i < len(doras) else 0
            
        for p in range(4):
            row[f"p{p}_is_reach"] = reaches[p]
            for i in range(1, 38):
                row[f"p{p}_hand_{i}"] = hands[p][i]
                row[f"p{p}_meld_{i}"] = melds[p][i]
            for i in range(24):
                row[f"p{p}_disc_{i+1}"] = player_discards[p][i] if i < len(player_discards[p]) else 0
                
        csv_rows.append(row)

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == "INIT":
            step_count = 0
            hands = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
            melds = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
            player_discards = {0: [], 1: [], 2: [], 3: []}
            reaches = {0: 0, 1: 0, 2: 0, 3: 0}
            doras = []
            
            seed = elem.attrib.get("seed", "").split(",")
            current_round_num = int(seed[0]) if len(seed) > 0 else 0
            current_honba = int(seed[1]) if len(seed) > 1 else 0
            dora_id = int(seed[5]) if len(seed) > 5 else -1
            
            dora_37 = to_37_id(dora_id)
            if dora_37 > 0: doras.append(dora_37)
            
            txt_lines.append(f"--- 局開始: {current_round_num}局 {current_honba}本場 ---")
            for p in range(4):
                hai_str = elem.attrib.get(f"hai{p}", "")
                tiles_int = [int(tid) for tid in hai_str.split(",") if tid]
                txt_lines.append(f"[{player_names[p]}] 配牌: " + " ".join([tile_id_to_str(t) for t in tiles_int]))
                for tid in tiles_int:
                    t37 = to_37_id(tid)
                    if t37 > 0: hands[p][t37] += 1
                    
            append_csv_row(-1, 0, dora_37)

        elif len(tag) >= 2 and tag[0] in ["T", "U", "V", "W", "D", "E", "F", "G"] and tag[1:].isdigit():
            action_code = tag[0]
            tile_id = int(tag[1:])
            t37 = to_37_id(tile_id)
            step_count += 1
            
            if action_code in ["T", "U", "V", "W"]:
                p_idx = ["T", "U", "V", "W"].index(action_code)
                last_drawn_tile[p_idx] = tile_id
                if t37 > 0: hands[p_idx][t37] += 1
                append_csv_row(p_idx, 1, t37)
                
            elif action_code in ["D", "E", "F", "G"]:
                p_idx = ["D", "E", "F", "G"].index(action_code)
                is_tsumogiri = (last_drawn_tile[p_idx] == tile_id)
                tsumo_flag_int = 1 if is_tsumogiri else 0
                
                if t37 > 0:
                    if hands[p_idx][t37] > 0: hands[p_idx][t37] -= 1
                    player_discards[p_idx].append(t37)
                
                append_csv_row(p_idx, 2, t37, tsumo_flag_int)
                last_drawn_tile[p_idx] = -1

        elif tag == "N":
            p_idx = int(elem.attrib.get("who", 0))
            m_val = elem.attrib.get("m", 0)
            step_count += 1
            
            exposed_tiles = decode_meld_to_37(m_val)
            for t37 in exposed_tiles:
                if t37 > 0: melds[p_idx][t37] += 1
                
            txt_lines.append(f"Step {step_count}: [{player_names[p_idx]}] 鳴き")
            append_csv_row(p_idx, 3, 0)
            
        elif tag == "REACH":
            p_idx = int(elem.attrib.get("who", 0))
            step = elem.attrib.get("step", "")
            if step == "1":
                reaches[p_idx] = 1
                txt_lines.append(f"Step {step_count}: [{player_names[p_idx]}] リーチ")
                append_csv_row(p_idx, 4, 0)
                
        elif tag == "DORA":
            new_dora_id = int(elem.attrib.get("hai", -1))
            dora_37 = to_37_id(new_dora_id)
            if dora_37 > 0: doras.append(dora_37)
            append_csv_row(-1, 5, dora_37)

    return "\n".join(txt_lines), csv_rows

# -------------------------------------------------------------
# 4. 並列処理ワーカー関数
# -------------------------------------------------------------
def worker_process_file(args):
    filename, raw_bytes, fieldnames = args
    if raw_bytes.startswith(b'\x1f\x8b'):
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception:
            return None
    
    xml_content = raw_bytes.decode("utf-8", errors="ignore")
    txt_content, csv_rows = parse_mjlog_content(xml_content, filename)
    
    if txt_content and csv_rows:
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
        return {
            "txt_name": txt_filename,
            "txt_data": txt_content,
            "csv_name": csv_filename,
            "csv_data": '\ufeff' + csv_buffer.getvalue()
        }
    return None

# -------------------------------------------------------------
# 5. ZIP抽出＆並列処理メイン関数
# -------------------------------------------------------------
def process_mjlogs_from_zip(input_zip, output_txt_zip, output_csv_zip):
    fieldnames = ["round", "honba", "step", "actor", "action", "target_tile", "tsumogiri"]
    for i in range(5): fieldnames.append(f"dora_{i+1}")
    for p in range(4):
        fieldnames.append(f"p{p}_is_reach")
        for i in range(1, 38): fieldnames.append(f"p{p}_hand_{i}")
        for i in range(1, 38): fieldnames.append(f"p{p}_meld_{i}")
        for i in range(24): fieldnames.append(f"p{p}_disc_{i+1}")

    print("ZIPファイルからデータを読み込み中...")
    tasks = []
    with zipfile.ZipFile(input_zip, 'r') as in_zip:
        target_files = [f for f in in_zip.infolist() if f.filename.endswith(".mjlog") or f.filename.endswith(".xml")]
        for file_info in target_files:
            tasks.append((os.path.basename(file_info.filename), in_zip.read(file_info), fieldnames))
            
    total_files = len(tasks)
    if total_files == 0:
        print("処理対象のファイルが見つかりません。")
        return

    processed_count = 0
    print(f"並列処理を開始します（対象: {total_files}件, CPUコアフル活用）")
    
    with zipfile.ZipFile(output_txt_zip, 'w', zipfile.ZIP_DEFLATED) as txt_zip, \
         zipfile.ZipFile(output_csv_zip, 'w', zipfile.ZIP_DEFLATED) as csv_zip:
         
        max_workers = os.cpu_count() or 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(worker_process_file, tasks):
                processed_count += 1
                if result:
                    txt_zip.writestr(result["txt_name"], result["txt_data"])
                    csv_zip.writestr(result["csv_name"], result["csv_data"])
                
                if processed_count % 100 == 0 or processed_count == total_files:
                    print(f"\r進捗: {processed_count}/{total_files} 件完了", end="")

    print(f"\n\n処理完了: {processed_count}件のファイルを保存しました。")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    INPUT_ZIP = "Nisimura.zip"
    TXT_ZIP_OUT = "Nisimura_t.zip"
    CSV_ZIP_OUT = "Nisimura_c.zip"
    
    if os.path.exists(INPUT_ZIP): 
        process_mjlogs_from_zip(INPUT_ZIP, TXT_ZIP_OUT, CSV_ZIP_OUT)
    else:
        print(f"エラー: {INPUT_ZIP} が見つかりません。")