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
# 1. 牌ID変換
# -------------------------------------------------------------
def to_37_id(tile_id):
    if tile_id is None or tile_id == "": return 0
    tid = int(tile_id)
    if tid == 16: return 35
    elif tid == 52: return 36
    elif tid == 88: return 37
    else: return (tid // 4) + 1

def clean_xml_string(xml_text):
    text = xml_text.replace('\x00', '').lstrip('\ufeff').strip()
    if text.startswith('%3C') or text.startswith('%3c'):
        text = urllib.parse.unquote(text)
    text = re.sub(r'<\?xml[^>]*\?>', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE).strip()
    return text

# -------------------------------------------------------------
# 2. 鳴き処理（公開状態、手牌の減少、見えている牌のカウントを同時処理）
# -------------------------------------------------------------
def process_meld_state(m_val, p_idx, hands, melds, visible, last_discard_37):
    m = int(m_val)
    try:
        if m & 0x0004:  # チー
            hai0, hai1, hai2 = (m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3
            base = (m >> 10) & 63
            color, num = base // 7, base % 7
            t0 = to_37_id((color * 9 + num) * 4 + hai0)
            t1 = to_37_id((color * 9 + num + 1) * 4 + hai1)
            t2 = to_37_id((color * 9 + num + 2) * 4 + hai2)
            tiles = [t0, t1, t2]
            for t in tiles: melds[p_idx][t] += 1
            
            from_hand = tiles.copy()
            if last_discard_37 in from_hand:
                from_hand.remove(last_discard_37)
            for t in from_hand:
                visible[t] += 1
                if hands[p_idx][t] > 0: hands[p_idx][t] -= 1

        elif m & 0x0018:  # ポン・加槓
            hai0, hai1, hai2 = (m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3
            base = (m >> 9) & 127
            t_base = base // 3
            if m & 0x0010:  # 加槓 (手牌から1枚追加)
                hai3 = (m >> 14) & 3
                t3 = to_37_id(t_base * 4 + hai3)
                melds[p_idx][t3] += 1
                visible[t3] += 1
                if hands[p_idx][t3] > 0: hands[p_idx][t3] -= 1
            else:  # ポン
                t0, t1, t2 = to_37_id(t_base * 4 + hai0), to_37_id(t_base * 4 + hai1), to_37_id(t_base * 4 + hai2)
                tiles = [t0, t1, t2]
                for t in tiles: melds[p_idx][t] += 1
                from_hand = tiles.copy()
                if last_discard_37 in from_hand:
                    from_hand.remove(last_discard_37)
                for t in from_hand:
                    visible[t] += 1
                    if hands[p_idx][t] > 0: hands[p_idx][t] -= 1

        elif m & 0x0020:  # 抜きドラ
            t = to_37_id((m >> 8) & 255)
            melds[p_idx][t] += 1
            visible[t] += 1
            if hands[p_idx][t] > 0: hands[p_idx][t] -= 1

        else:  # 暗槓・大明槓
            hai0 = (m >> 8) & 255
            t_base = hai0 // 4
            tiles = [to_37_id(t_base * 4), to_37_id(t_base * 4 + 1), to_37_id(t_base * 4 + 2), to_37_id(t_base * 4 + 3)]
            who_from = m & 0x0003
            
            for t in tiles: melds[p_idx][t] += 1
            
            if who_from == 0:  # 暗槓 (4枚すべて手牌から)
                for t in tiles:
                    visible[t] += 1
                    if hands[p_idx][t] > 0: hands[p_idx][t] -= 1
            else:  # 大明槓 (3枚手牌から)
                from_hand = tiles.copy()
                if last_discard_37 in from_hand:
                    from_hand.remove(last_discard_37)
                for t in from_hand:
                    visible[t] += 1
                    if hands[p_idx][t] > 0: hands[p_idx][t] -= 1
    except Exception:
        pass

# -------------------------------------------------------------
# 3. .mjlog 解析関数
# -------------------------------------------------------------
def parse_mjlog_content(xml_text, file_name):
    xml_text = clean_xml_string(xml_text)
    try:
        root = ET.fromstring(f"<root>{xml_text}</root>")
    except ET.ParseError:
        xml_text_clean = re.sub(r'&(?![a-zA-Z]+;)', '&amp;', xml_text)
        try: root = ET.fromstring(f"<root>{xml_text_clean}</root>")
        except ET.ParseError: return "", []

    csv_rows = []
    
    current_round_num = 0
    current_honba = 0
    kyotaku = 0
    remain_tsumo = 70
    scores = [25000, 25000, 25000, 25000]
    
    last_drawn_tile = {0: -1, 1: -1, 2: -1, 3: -1}
    last_discard_37 = 0
    step_count = 0

    hands = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
    melds = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
    player_discards = {0: [], 1: [], 2: [], 3: []}
    reaches = {0: 0, 1: 0, 2: 0, 3: 0}
    visible = [0] * 38
    doras = []

    def append_csv_row(actor, action, target_tile_37, tsumogiri=0):
        normalized_actor = actor if 0 <= actor <= 3 else 4
        row = {
            "round": current_round_num, "honba": current_honba, "kyotaku": kyotaku,
            "remain_tsumo": remain_tsumo, "step": step_count, "actor": normalized_actor, 
            "action": action, "target_tile": target_tile_37, "tsumogiri": tsumogiri
        }
        for i in range(5): row[f"dora_{i+1}"] = doras[i] if i < len(doras) else 0
        for i in range(4): row[f"p{i}_score"] = scores[i]
        for i in range(1, 38): row[f"visible_{i}"] = visible[i]
            
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
            remain_tsumo = 70
            hands = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
            melds = {0: [0]*38, 1: [0]*38, 2: [0]*38, 3: [0]*38}
            player_discards = {0: [], 1: [], 2: [], 3: []}
            reaches = {0: 0, 1: 0, 2: 0, 3: 0}
            visible = [0] * 38
            doras = []
            
            seed = elem.attrib.get("seed", "").split(",")
            current_round_num = int(seed[0]) if len(seed) > 0 else 0
            current_honba = int(seed[1]) if len(seed) > 1 else 0
            kyotaku = int(seed[2]) if len(seed) > 2 else 0
            
            ten = elem.attrib.get("ten", "250,250,250,250").split(",")
            scores = [int(x)*100 for x in ten]
            
            dora_id = int(seed[5]) if len(seed) > 5 else -1
            dora_37 = to_37_id(dora_id)
            if dora_37 > 0: 
                doras.append(dora_37)
                visible[dora_37] += 1
            
            for p in range(4):
                hai_str = elem.attrib.get(f"hai{p}", "")
                tiles_int = [int(tid) for tid in hai_str.split(",") if tid]
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
                remain_tsumo = max(0, remain_tsumo - 1)
                if t37 > 0: hands[p_idx][t37] += 1
                append_csv_row(p_idx, 1, t37)
                
            elif action_code in ["D", "E", "F", "G"]:
                p_idx = ["D", "E", "F", "G"].index(action_code)
                is_tsumogiri = (last_drawn_tile[p_idx] == tile_id)
                tsumo_flag_int = 1 if is_tsumogiri else 0
                
                if t37 > 0:
                    visible[t37] += 1
                    last_discard_37 = t37
                    if hands[p_idx][t37] > 0: hands[p_idx][t37] -= 1
                    # ツモ切りはマイナス、手出しはプラスで記録！
                    record_val = -t37 if is_tsumogiri else t37
                    player_discards[p_idx].append(record_val)
                
                append_csv_row(p_idx, 2, t37, tsumo_flag_int)
                last_drawn_tile[p_idx] = -1

        elif tag == "N":
            p_idx = int(elem.attrib.get("who", 0))
            m_val = elem.attrib.get("m", 0)
            step_count += 1
            
            process_meld_state(m_val, p_idx, hands, melds, visible, last_discard_37)
            append_csv_row(p_idx, 3, 0)
            
        elif tag == "REACH":
            p_idx = int(elem.attrib.get("who", 0))
            step = elem.attrib.get("step", "")
            if step == "1":
                reaches[p_idx] = 1
                append_csv_row(p_idx, 4, 0)
            elif step == "2":
                scores[p_idx] -= 1000
                kyotaku += 1
                ten_str = elem.attrib.get("ten", None)
                if ten_str: scores = [int(x)*100 for x in ten_str.split(",")]
                
        elif tag == "DORA":
            new_dora_id = int(elem.attrib.get("hai", -1))
            dora_37 = to_37_id(new_dora_id)
            if dora_37 > 0: 
                doras.append(dora_37)
                visible[dora_37] += 1
            append_csv_row(-1, 5, dora_37)

    return "", csv_rows

# -------------------------------------------------------------
# 4. 並列処理・書き出し
# -------------------------------------------------------------
def worker_process_file(args):
    filename, raw_bytes, fieldnames = args
    if raw_bytes.startswith(b'\x1f\x8b'):
        try: raw_bytes = gzip.decompress(raw_bytes)
        except Exception: return None
    
    xml_content = raw_bytes.decode("utf-8", errors="ignore")
    _, csv_rows = parse_mjlog_content(xml_content, filename)
    
    if csv_rows:
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        return {"csv_name": csv_filename, "csv_data": '\ufeff' + csv_buffer.getvalue()}
    return None

def process_mjlogs_from_zip(input_zip, output_csv_zip):
    fieldnames = ["round", "honba", "kyotaku", "remain_tsumo", "step", "actor", "action", "target_tile", "tsumogiri"]
    for i in range(5): fieldnames.append(f"dora_{i+1}")
    for i in range(4): fieldnames.append(f"p{i}_score")
    for i in range(1, 38): fieldnames.append(f"visible_{i}")
    for p in range(4):
        fieldnames.append(f"p{p}_is_reach")
        for i in range(1, 38): fieldnames.append(f"p{p}_hand_{i}")
        for i in range(1, 38): fieldnames.append(f"p{p}_meld_{i}")
        for i in range(24): fieldnames.append(f"p{p}_disc_{i+1}")

    print("ZIPファイル読み込み中...")
    tasks = []
    with zipfile.ZipFile(input_zip, 'r') as in_zip:
        target_files = [f for f in in_zip.infolist() if f.filename.endswith((".mjlog", ".xml"))]
        for file_info in target_files:
            tasks.append((os.path.basename(file_info.filename), in_zip.read(file_info), fieldnames))
            
    total_files = len(tasks)
    if total_files == 0: return

    processed_count = 0
    print(f"データ変換処理を開始します（対象: {total_files}件）")
    
    with zipfile.ZipFile(output_csv_zip, 'w', zipfile.ZIP_DEFLATED) as csv_zip:
        max_workers = os.cpu_count() or 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(worker_process_file, tasks):
                processed_count += 1
                if result: csv_zip.writestr(result["csv_name"], result["csv_data"])
                if processed_count % 100 == 0 or processed_count == total_files:
                    print(f"\r進捗: {processed_count}/{total_files} 件完了", end="")
    print(f"\n\n処理完了: {processed_count}件保存。")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    INPUT_ZIP = "Syobon.zip"
    CSV_ZIP_OUT = "Syobon_c_v2.zip"
    
    if os.path.exists(INPUT_ZIP): 
        process_mjlogs_from_zip(INPUT_ZIP, CSV_ZIP_OUT)
    else:
        print(f"エラー: {INPUT_ZIP} が見つかりません。")