import csv
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import io
import gzip

# ======================================================================
# 1. 牌ID・鳴きコードの変換・解析
# ======================================================================

def get_tile_str(tile_id):
    if tile_id is None or tile_id < 0: return ""
    if tile_id == 16: return "赤5萬"
    if tile_id == 52: return "赤5筒"
    if tile_id == 88: return "赤5索"
    
    suit = tile_id // 36
    if suit < 3:
        num = ((tile_id % 36) // 4) + 1
        return f"{num}{['萬', '筒', '索'][suit]}"
    else:
        return ["東", "南", "西", "北", "白", "發", "中"][(tile_id - 108) // 4]

def decode_meld_details(m):
    m = int(m)
    consumed = []
    meld_str = ""
    
    if m & 0x4: # チー
        t0 = (m >> 10) & 0x3F
        p = t0 % 3
        t0 = (t0 // 3) * 4 + p
        r = m & 0x3
        if r == 0: consumed = [t0 + 4, t0 + 8]
        elif r == 1: consumed = [t0 - 4, t0 + 4]
        elif r == 2: consumed = [t0 - 8, t0 - 4]
        tiles = [get_tile_str(t) for t in sorted([t0] + consumed)]
        meld_str = f"チー ({' '.join(tiles)})"
        
    elif m & 0x8: # ポン
        t0 = (m >> 9) & 0x7F
        p = t0 % 3
        t0 = (t0 // 3) * 4
        all_ids = [t0, t0+1, t0+2, t0+3]
        consumed = [id_ for id_ in all_ids if id_ % 4 != p][:2]
        meld_str = f"ポン ({get_tile_str(t0)}x3)"
        
    elif m & 0x10: # 加槓
        t0 = (m >> 9) & 0x7F
        p = t0 % 3
        t0 = (t0 // 3) * 4
        consumed = [t0 + p]
        meld_str = f"加槓 ({get_tile_str(t0)}x4)"
        
    else: # 暗槓 または 大明槓
        t0 = (m >> 8) & 0xFF
        t0 = (t0 // 4) * 4
        is_ankan = (m & 0x3) == 0
        consumed = [t0, t0+1, t0+2, t0+3][:4 if is_ankan else 3]
        meld_type = "暗槓" if is_ankan else "大明槓"
        meld_str = f"{meld_type} ({get_tile_str(t0)}x4)"
        
    return meld_str, consumed

# ======================================================================
# 2. XML解析・データ抽出コア処理 (★ここから大きく改修)
# ======================================================================

def parse_mjlog(xml_text, file_name):
    xml_text = xml_text.replace('\x00', '').lstrip('\ufeff').strip()
    if xml_text.startswith('%3C') or xml_text.startswith('%3c'):
        xml_text = urllib.parse.unquote(xml_text)
    
    try:
        root = ET.fromstring(f"<root>{re.sub(r'<\?xml[^>]*\?>|<!DOCTYPE[^>]*>', '', xml_text, flags=re.IGNORECASE)}</root>")
    except ET.ParseError:
        return "", []

    txt_lines = []
    csv_rows = []
    
    players = ["Player0", "Player1", "Player2", "Player3"]
    un_tag = root.find(".//UN")
    if un_tag is not None:
        for i in range(4):
            if f"n{i}" in un_tag.attrib:
                players[i] = urllib.parse.unquote(un_tag.attrib[f"n{i}"])

    txt_lines.extend([f"対局: {file_name}", f"対局者: {', '.join(players)}", "=" * 60])

    # ★状態追跡変数に「捨て牌」と「鳴き牌」を追加
    kyoku, honba, kyoutaku = 0, 0, 0
    step = 0
    hands = {0: [], 1: [], 2: [], 3: []}
    discard_counts = {p: [0]*34 for p in range(4)} # 34種類の捨て牌カウント
    meld_counts = {p: [0]*34 for p in range(4)}    # 34種類の鳴き牌カウント
    
    def create_row(actor=-1, action=-1, tile=-1, tsumogiri=0):
        row = {
            "kyoku": kyoku, "honba": honba, "kyoutaku": kyoutaku,
            "step": step, "actor": actor, "action": action, 
            "tile": tile, "tsumogiri": tsumogiri
        }
        for p in range(4):
            # 1. 手牌ID (従来通り)
            for i in range(14):
                row[f"p{p}_{i}"] = hands[p][i] if i < len(hands[p]) else -1
            # 2. 捨て牌の数 (新機能)
            for i in range(34):
                row[f"d{p}_{i}"] = discard_counts[p][i]
            # 3. 鳴きで晒した牌の数 (新機能)
            for i in range(34):
                row[f"m{p}_{i}"] = meld_counts[p][i]
        return row

    for elem in root.iter():
        tag = elem.tag
        
        # [0: INIT] 局開始
        if tag == "INIT":
            step = 0
            seed = elem.attrib.get("seed", "0,0,0,0,0,0").split(",")
            kyoku, honba, kyoutaku = int(seed[0]), int(seed[1]), int(seed[2])
            dora_id = int(seed[5])
            
            # ★局の開始時に捨て牌・鳴きの記憶をリセット
            for p in range(4):
                discard_counts[p] = [0]*34
                meld_counts[p] = [0]*34
            
            txt_lines.append(f"\n--- 【{['東','南','西','北'][kyoku//4]}{(kyoku%4)+1}局 {honba}本場】 供託:{kyoutaku} ドラ表示:{get_tile_str(dora_id)} ---")
            
            for p in range(4):
                hands[p] = sorted([int(t) for t in elem.attrib.get(f"hai{p}", "").split(",") if t])
                txt_lines.append(f"  {players[p]} 配牌: {' '.join([get_tile_str(t) for t in hands[p]])}")
                
            csv_rows.append(create_row(action=0, tile=dora_id))

        # [1: ツモ / 2: 打牌]
        elif len(tag) >= 2 and tag[0] in "TUVWXYZDEFG" and tag[1:].isdigit():
            action_code = tag[0]
            tile_id = int(tag[1:])
            step += 1
            
            if action_code in "TUVW":
                p_idx = "TUVW".index(action_code)
                hands[p_idx].append(tile_id)
                txt_lines.append(f"  [{step:03d}] {players[p_idx]}: ツモ {get_tile_str(tile_id)}")
                csv_rows.append(create_row(actor=p_idx, action=1, tile=tile_id))
                
            elif action_code in "DEFG":
                p_idx = "DEFG".index(action_code)
                is_tsumogiri = (hands[p_idx] and hands[p_idx][-1] == tile_id)
                
                if tile_id in hands[p_idx]:
                    hands[p_idx].remove(tile_id)
                hands[p_idx].sort()
                
                # ★打牌されたら、その牌の種類(0~33)のカウントを+1する
                discard_counts[p_idx][tile_id // 4] += 1
                
                txt_lines.append(f"  [{step:03d}] {players[p_idx]}: 打 {get_tile_str(tile_id)} ({'ツモ切り' if is_tsumogiri else '手出し'})")
                csv_rows.append(create_row(actor=p_idx, action=2, tile=tile_id, tsumogiri=1 if is_tsumogiri else 0))

        # [3: 鳴き(副露)]
        elif tag == "N":
            p_idx = int(elem.attrib.get("who", 0))
            meld_id = int(elem.attrib.get("m", -1))
            step += 1
            
            meld_str, consumed_tiles = decode_meld_details(meld_id)
            for c_tile in consumed_tiles:
                # ★鳴きで手牌から出た牌をカウント
                meld_counts[p_idx][c_tile // 4] += 1
                
                for h_tile in hands[p_idx]:
                    if h_tile // 4 == c_tile // 4:
                        hands[p_idx].remove(h_tile)
                        break
            hands[p_idx].sort()
            
            txt_lines.append(f"  [{step:03d}] {players[p_idx]}: 鳴き {meld_str}")
            csv_rows.append(create_row(actor=p_idx, action=3, tile=meld_id))

        # [4: リーチ]
        elif tag == "REACH":
            if elem.attrib.get("step") == "1":
                p_idx = int(elem.attrib.get("who", 0))
                txt_lines.append(f"  [*] {players[p_idx]}: リーチ宣言")
                csv_rows.append(create_row(actor=p_idx, action=4))
            elif elem.attrib.get("step") == "2":
                kyoutaku += 1

        # [5: 新ドラ追加]
        elif tag == "DORA":
            dora_id = int(elem.attrib.get("hai", -1))
            txt_lines.append(f"  [*] 新ドラ: {get_tile_str(dora_id)}")
            csv_rows.append(create_row(action=5, tile=dora_id))

    return "\n".join(txt_lines), csv_rows

# ======================================================================
# 3. メイン処理
# ======================================================================

def process_mjlogs(input_zip, output_txt_zip, output_csv_zip):
    # ★CSVのヘッダー（列名）を大幅に拡張
    fieldnames = ["kyoku", "honba", "kyoutaku", "step", "actor", "action", "tile", "tsumogiri"]
    for p in range(4):
        for i in range(14): fieldnames.append(f"p{p}_{i}")
        for i in range(34): fieldnames.append(f"d{p}_{i}") # 捨て牌列 (d0_0 〜 d3_33)
        for i in range(34): fieldnames.append(f"m{p}_{i}") # 鳴き牌列 (m0_0 〜 m3_33)

    with zipfile.ZipFile(output_txt_zip, 'w', zipfile.ZIP_DEFLATED) as txt_zip, \
         zipfile.ZipFile(output_csv_zip, 'w', zipfile.ZIP_DEFLATED) as csv_zip, \
         zipfile.ZipFile(input_zip, 'r') as in_zip:

        target_files = [f for f in in_zip.infolist() if f.filename.endswith((".mjlog", ".xml"))]
        total = len(target_files)
        print(f"変換開始: {total} ファイル")

        for idx, file_info in enumerate(target_files, 1):
            base_name = os.path.basename(file_info.filename)
            raw_bytes = in_zip.read(file_info)
            
            if raw_bytes.startswith(b'\x1f\x8b'):
                try: raw_bytes = gzip.decompress(raw_bytes)
                except: continue
                
            txt_content, csv_rows = parse_mjlog(raw_bytes.decode("utf-8", errors="ignore"), base_name)
            
            if csv_rows:
                txt_zip.writestr(base_name.replace(".mjlog", ".txt").replace(".xml", ".txt"), txt_content)
                
                csv_buffer = io.StringIO()
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator='\n')
                writer.writeheader()
                writer.writerows(csv_rows)
                csv_zip.writestr(base_name.replace(".mjlog", ".csv").replace(".xml", ".csv"), csv_buffer.getvalue())
                
            print(f"\r処理中: {idx}/{total} ({idx/total*100:.1f}%)", end="")
            
    print("\n変換処理が完了しました。")

if __name__ == "__main__":
    # ここを処理したいZIPファイル名に変更して実行してください
    # （例: Nisimura.zip, Syobon.zip, Son.zip, CLS.zip をそれぞれ順に実行）
    INPUT_FILE = "CLS.zip"
    
    if os.path.exists(INPUT_FILE):
        process_mjlogs(INPUT_FILE, "CLS_t.zip", "CLS_c.zip")
    else:
        print(f"エラー: {INPUT_FILE} が見つかりません。")