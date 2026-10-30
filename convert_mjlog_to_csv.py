import os
import glob
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ディレクトリ設定
INPUT_DIR = "./houou_paifu_data"
OUTPUT_DIR = "./houou_csv_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 天鳳牌番号 (0~135) を 37種 (1~37) に変換するマップ関数
def tile_136_to_37(tile_136):
    if tile_136 is None or tile_136 < 0:
        return 0
    # 赤ドラ判定 (51: 赤5m, 88: 赤5p, 108: 赤5s)
    if tile_136 == 51:
        return 35
    elif tile_136 == 88:
        return 36
    elif tile_136 == 108:
        return 37
    
    tile_34 = tile_136 // 4
    return tile_34 + 1

# 鳴き (m値) から鳴いた牌の37種表現を抽出
def parse_meld_tiles_37(m):
    m = int(m)
    tiles_37 = []
    if m & 0x4: # チー
        pattern = (m >> 10) & 0x3F
        t_seq = pattern // 3
        color = t_seq // 7
        n = t_seq % 7
        t0 = color * 9 + n
        tiles_37 = [t0 + 1, t0 + 2, t0 + 3]
    elif m & 0x8: # ポン
        t_34 = (m >> 9) & 0x7F
        tiles_37 = [t_34 // 3 + 1] * 3
    elif m & 0x20: # 加カン
        t_34 = (m >> 9) & 0x7F
        tiles_37 = [t_34 // 3 + 1] * 4
    else: # 大明カン/暗カン
        t_34 = (m >> 8) & 0xFF
        tiles_37 = [t_34 // 4 + 1] * 4
    return tiles_37

def parse_mjlog_to_dataframe(xml_content, perspective_player=0, mask_opponents=True):
    root = ET.fromstring(xml_content)
    rows = []
    
    round_num = 0
    honba = 0
    kyotaku = 0
    scores = [25000, 25000, 25000, 25000]
    last_drawn_tile_136 = {p: None for p in range(4)}
    
    for elem in root:
        tag = elem.tag
        
        if tag == 'INIT':
            seed = [int(x) for x in elem.attrib['seed'].split(',')]
            round_num = seed[0]
            honba = seed[1]
            kyotaku = seed[2]
            dora_first = seed[5]
            
            ten = [int(x) for x in elem.attrib['ten'].split(',')]
            scores = [t * 100 for t in ten]
            
            hands = {p: [0] * 38 for p in range(4)}
            for p in range(4):
                hai_attr = elem.attrib.get(f'hai{p}', '')
                if hai_attr:
                    for tile_136 in map(int, hai_attr.split(',')):
                        t37 = tile_136_to_37(tile_136)
                        hands[p][t37] += 1
                        
            melds = {p: [0] * 38 for p in range(4)}
            discards = {p: [] for p in range(4)}
            reaches = [0] * 4
            doras = [tile_136_to_37(dora_first)]
            visibles = [0] * 38
            for d in doras:
                visibles[d] += 1
                
            remain_tsumo = 70
            step = 0
            last_drawn_tile_136 = {p: None for p in range(4)}
            
        elif tag in ['N', 'REACH', 'DORA']:
            if tag == 'DORA':
                doras.append(tile_136_to_37(int(elem.attrib['hai'])))
            elif tag == 'REACH':
                who = int(elem.attrib['who'])
                step_val = int(elem.attrib.get('step', 0))
                if step_val == 1:
                    reaches[who] = 1
            elif tag == 'N':
                who = int(elem.attrib['who'])
                m = elem.attrib['m']
                m_tiles = parse_meld_tiles_37(m)
                for t in m_tiles:
                    melds[who][t] += 1
                    visibles[t] += 1
                    if hands[who][t] > 0:
                        hands[who][t] -= 1
                        
        # ツモ・打牌イベント (2文字以上で先頭が指定文字かつ残りがすべて数字である場合のみ対象)
        elif len(tag) > 1 and tag[0] in ['T', 'U', 'V', 'W', 'D', 'E', 'F', 'G'] and tag[1:].isdigit():
            actor = -1
            action_type = -1
            tile_136 = int(tag[1:])
            tile_val = tile_136_to_37(tile_136)
            tsumogiri = 0
            
            # 打牌 (D=p0, E=p1, F=p2, G=p3)
            if tag[0] in ['D', 'E', 'F', 'G']:
                actor = ['D', 'E', 'F', 'G'].index(tag[0])
                action_type = 0
                
                # ツモ切り判定
                if last_drawn_tile_136[actor] == tile_136:
                    tsumogiri = 1
                last_drawn_tile_136[actor] = None
                
                discards[actor].append(tile_val)
                visibles[tile_val] += 1
                if hands[actor][tile_val] > 0:
                    hands[actor][tile_val] -= 1
                    
            # ツモ (T=p0, U=p1, V=p2, W=p3)
            elif tag[0] in ['T', 'U', 'V', 'W']:
                actor = ['T', 'U', 'V', 'W'].index(tag[0])
                action_type = 1
                last_drawn_tile_136[actor] = tile_136
                
                hands[actor][tile_val] += 1
                remain_tsumo -= 1

            row = {
                'round': round_num, 'honba': honba, 'kyotaku': kyotaku,
                'remain_tsumo': max(0, remain_tsumo), 'step': step,
                'actor': actor, 'action': action_type, 'target_tile': tile_val,
                'tsumogiri': tsumogiri
            }
            
            for i in range(1, 6):
                row[f'dora_{i}'] = doras[i-1] if i <= len(doras) else 0
                
            for p in range(4):
                rel_p = (p - perspective_player) % 4
                row[f'p{rel_p}_score'] = scores[p]
                
            for t in range(1, 38):
                row[f'visible_{t}'] = visibles[t]
                
            for p in range(4):
                rel_p = (p - perspective_player) % 4
                row[f'p{rel_p}_is_reach'] = reaches[p]
                
                # 相手手牌のマスク処理 (データリーク防止)
                for t in range(1, 38):
                    if mask_opponents and rel_p != 0:
                        row[f'p{rel_p}_hand_{t}'] = 0
                    else:
                        row[f'p{rel_p}_hand_{t}'] = hands[p][t]
                        
                    row[f'p{rel_p}_meld_{t}'] = melds[p][t]
                    
                for d_idx in range(1, 25):
                    if d_idx <= len(discards[p]):
                        row[f'p{rel_p}_disc_{d_idx}'] = discards[p][d_idx-1]
                    else:
                        row[f'p{rel_p}_disc_{d_idx}'] = 0
                        
            rows.append(row)
            step += 1
            
    return pd.DataFrame(rows)

def convert_all_mjlogs():
    mjlog_files = glob.glob(os.path.join(INPUT_DIR, "*.mjlog"))
    print(f"変換対象の .mjlog ファイル数: {len(mjlog_files)} 件")
    
    for idx, filepath in enumerate(mjlog_files, 1):
        filename = os.path.basename(filepath).replace('.mjlog', '')
        
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            if raw_data.startswith(b'\x1f\x8b'):
                xml_content = gzip.decompress(raw_data).decode('utf-8', errors='ignore')
            else:
                xml_content = raw_data.decode('utf-8', errors='ignore')
                
            df = parse_mjlog_to_dataframe(xml_content, perspective_player=0, mask_opponents=True)
            
            output_csv_path = os.path.join(OUTPUT_DIR, f"{filename}&tw=0.csv")
            df.to_csv(output_csv_path, index=False)
            
            if idx % 10 == 0 or idx == len(mjlog_files):
                print(f"変換進捗: {idx}/{len(mjlog_files)} 完了 -> {output_csv_path}")
                
        except Exception as e:
            print(f"エラー発生 ({filename}): {e}")

if __name__ == "__main__":
    convert_all_mjlogs()