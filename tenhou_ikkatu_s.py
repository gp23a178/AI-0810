import os
import re
import random
import urllib.request
import urllib.parse
import zipfile
import time

# 天鳳公式ランキング（ranking.html）に記載されている四人打ち天鳳位の正確な名前
ALL_PLAYERS = [
    "しょぼんｎ", "西村雄一郎", "☆孫燕姿☆", "あまあん", "花道桜木",
    "小林流牌手", "吃貨小北北", "COEDO緑", "じょにおん！！", "わっしょい君2号",
    "火時計を押せ！", "いばらぎ", "yoteru", "CLS", "藤井聡ふと",
    "右折するひつじ", "お知らせ", "gousi", "おかもと", "トトリ先生19歳",
    "ウルトラ立直", "就活生@川村軍団", "かにマジン", "コーラ下さい", "タケオしゃん",
    "太くないお", "すずめクレイジー", "独歩", "（≧▽≦）", "ASAPIN"
]

# ランダムに5名を選択（学習3人、検証2人）
selected_players = random.sample(ALL_PLAYERS, 5)
TRAIN_PLAYERS = selected_players[:3]
VAL_PLAYERS = selected_players[3:]

def get_player_zip_links():
    """
    ランキングページ(ranking.html)を解析し、各プレイヤーの「牌譜 DL (ZIP)」リンクを取得する
    """
    url = "https://tenhou.net/ranking.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    links = {}
    
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            raw_data = res.read()
            try:
                html = raw_data.decode('utf-8')
            except UnicodeDecodeError:
                html = raw_data.decode('cp932', errors='ignore')
                
        # HTMLを <br> タグや改行で完全に分割して、1人ずつのデータ行にする
        segments = re.split(r'<br\s*/?>|</tr>|</li>|\n', html, flags=re.IGNORECASE)
        
        for segment in segments:
            # HTMLタグを除去したプレーンテキストを生成
            plain_text = re.sub(r'<[^>]+>', '', segment)
            
            # 「DL」と「戦」の文字が含まれていれば、天鳳位のログ行と判定
            if 'DL' in plain_text and '戦' in plain_text:
                # aタグの href="〇〇" を抽出
                href_match = re.search(r'href=["\']([^"\']+)["\']', segment, re.IGNORECASE)
                if href_match:
                    href = href_match.group(1)
                    
                    # この行が「どのプレイヤーの行か」を判定してURLを紐付ける
                    for p in ALL_PLAYERS:
                        if p in plain_text:
                            # 相対パス（sc/raw/dat/...）を絶対URLに変換
                            links[p] = urllib.parse.urljoin(url, href)
                            
    except Exception as e:
        print(f"ランキングページの通信・解析エラー: {e}")
        
    return links

def download_and_extract(player, zip_url, output_dir):   os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{player}_logs.zip")
    
    print(f"[{player}] のZIPデータをダウンロード中... ({zip_url})")
    try:
        # 修正ポイント：Tenhouサーバーに弾かれないよう「Referer」を追加
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://tenhou.net/ranking.html'
        }
        req = urllib.request.Request(zip_url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=60) as res:
            with open(zip_path, 'wb') as f:
                f.write(res.read())
                
        print(f"  -> ダウンロード完了。展開(解凍)しています...")
        # ZIPの解凍処理
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        print(f"  -> 展開完了！ (保存先: {output_dir}/)")
        
        # 不要になったZIPファイルを削除
        os.remove(zip_path)
        
    except Exception as e:
        print(f"  -> エラー: {e}")

def main():
    print("【Smallサイズ】全体30名から5名をランダム抽出します。")
    print(f"学習データ (3人): {TRAIN_PLAYERS}")
    print(f"検証データ (2人): {VAL_PLAYERS}\n")
    
    print("天鳳公式ランキングページから一括ダウンロードURLを取得中...")
    player_links = get_player_zip_links()
    
    if not player_links:
        print("エラー: ランキングページからリンクを取得できませんでした。")
        return
        
    print(f"-> 成功: {len(player_links)} 名分のZIPリンクを取得しました。\n")
        
    # 学習用データ(Train)の処理
    print("--- 学習データ (Train) のダウンロード開始 ---")
    for p in TRAIN_PLAYERS:
        if p in player_links:
            download_and_extract(p, player_links[p], "mjlogs_s/train")
        else:
            print(f"[{p}] のリンクが見つかりませんでした。")
        time.sleep(2)  # サーバー負荷軽減
            
    # 検証用データ(Val)の処理
    print("\n--- 検証データ (Val) のダウンロード開始 ---")
    for p in VAL_PLAYERS:
        if p in player_links:
            download_and_extract(p, player_links[p], "mjlogs_s/val")
        else:
            print(f"[{p}] のリンクが見つかりませんでした。")
        time.sleep(2)

    print("\n==========================================")
    print("すべての処理が完全に完了しました。")
    print("指定したフォルダ内に天鳳位の牌譜データが保存されています。")
    print("==========================================")

if __name__ == "__main__":
    main()