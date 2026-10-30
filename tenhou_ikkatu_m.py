# -*- coding: utf-8 -*-
"""天鳳ランキング掲載の四人麻雀牌譜ZIPをダウンロードするスクリプト（標準ライブラリのみ）"""

from html.parser import HTMLParser
from html import unescape as html_unescape
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urlparse
from pathlib import Path
from datetime import datetime
import csv
import random
import re
import shutil
import time
import zipfile
import sys

# =========================
# 設定：ここだけ変更すればOK
# =========================
RANKING_URL = "https://tenhou.net/ranking.html"
DATASET_MODE = "small"   # "small"=5人、"medium"=12人、"large"=全員
RANDOM_SEED = 42           # None にすると実行ごとに抽出対象が変わる
BASE_DIR = Path("tenhou_data")
DOWNLOAD_DELAY_SECONDS = 1.0
EXTRACT_ZIPS = False       # True にするとZIP内の牌譜も展開（追加の空き容量が必要）
RETRY_COUNT = 3

ZIP_RE = re.compile(r"mjlog_pf4-20_n\d+\.zip", re.IGNORECASE)
DATE_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}")
BATTLE_RE = re.compile(r"([\d,]+)\s*戦")
BLOCK_TAGS = {"br", "tr", "li", "p", "div", "h1", "h2", "h3", "h4", "h5", "hr", "section"}


class RankingLineParser(HTMLParser):
    """HTMLを簡易的に行単位に分け、各行のテキストとリンクを取得する。"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines = []
        self.parts = []
        self.links = []
        self.anchor = None

    def flush(self):
        text = re.sub(r"\s+", " ", "".join(self.parts)).strip()
        if text or self.links:
            self.lines.append({"text": text, "links": self.links[:]})
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in BLOCK_TAGS:
            self.flush()
        if tag == "a":
            attrs_dict = dict(attrs)
            self.anchor = {"href": attrs_dict.get("href", ""), "text": []}

    def handle_data(self, data):
        self.parts.append(data)
        if self.anchor is not None:
            self.anchor["text"].append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "a" and self.anchor is not None:
            self.anchor["text"] = "".join(self.anchor["text"]).strip()
            self.links.append(self.anchor)
            self.anchor = None
        if tag in BLOCK_TAGS:
            self.flush()

    def close(self):
        super().close()
        self.flush()


def fetch_html(url):
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; TenhouLogDownloader/1.0)",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def clean_player_name(text):
    text = text.strip()
    text = text.strip(" |｜\t\r\n　:：")
    return re.sub(r"\s+", " ", text).strip()


def parse_entry_text(text):
    """行テキストから日付、対戦数、プレイヤー名を取り出す。"""
    text = re.sub(r"\s+", " ", text).strip()
    date_match = DATE_RE.search(text)
    battle_match = BATTLE_RE.search(text)
    if not date_match or not battle_match:
        return None
    player = clean_player_name(text[battle_match.end():])
    if not player:
        return None
    return {
        "date": date_match.group(0),
        "battles": battle_match.group(1).replace(",", ""),
        "player": player,
    }


def raw_context_for_link(html, href):
    """行パーサで名前が取れない時の保険。元HTMLからリンク周辺の1行を抽出する。"""
    filename_match = ZIP_RE.search(href or "")
    key = filename_match.group(0) if filename_match else (href or "")
    pos = html.find(href) if href else -1
    if pos < 0 and key:
        pos = html.find(key)
    if pos < 0:
        return ""
    a_start = html.rfind("<a", 0, pos)
    if a_start >= 0:
        pos = a_start
    boundary = re.compile(r"<br\b[^>]*>|</(?:tr|li|p|div|section)\s*>", re.IGNORECASE)
    previous = list(boundary.finditer(html[:pos]))
    start = previous[-1].end() if previous else max(0, pos - 1500)
    following = boundary.search(html[pos:])
    end = pos + following.start() if following else min(len(html), pos + 1500)
    fragment = html[start:end]
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html_unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def scrape_entries():
    print(f"公式ランキングを確認します: {RANKING_URL}")
    html = fetch_html(RANKING_URL)
    parser = RankingLineParser()
    parser.feed(html)
    parser.close()

    entries = []
    seen_urls = set()
    parse_failures = []

    for line in parser.lines:
        for link in line["links"]:
            href = link.get("href", "")
            zip_match = ZIP_RE.search(href)
            if not zip_match:
                continue
            url = urljoin(RANKING_URL, href)
            if url in seen_urls:
                continue
            seen_urls.add(url)

            info = parse_entry_text(line["text"])
            if info is None:
                context = raw_context_for_link(html, href)
                info = parse_entry_text(context)
            if info is None:
                parse_failures.append((href, line["text"]))
                continue

            info["url"] = url
            info["archive_name"] = zip_match.group(0)
            entries.append(info)

    if parse_failures:
        print("\n注意：一部リンクでプレイヤー情報を読み取れませんでした。")
        for href, text in parse_failures[:5]:
            print(f"  リンク: {href}\n  行テキスト: {text[:180]}")

    if not entries:
        raise RuntimeError(
            "四人麻雀の牌譜DLリンクを取得できませんでした。\n"
            "公式ページのHTML構造またはリンク形式が変わった可能性があります。\n"
            "ページを開けるか確認し、表示されたエラー全文を教えてください。"
        )
    return entries


def write_csv(path, fieldnames, rows, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def safe_folder_name(name):
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip(" .")
    return name or "unknown_player"


def choose_players(entries):
    # 1人が複数回掲載されていても、抽出対象の人数には1人として数える。
    by_player = {}
    for entry in entries:
        by_player.setdefault(entry["player"], []).append(entry)

    player_names = list(by_player.keys())
    mode = DATASET_MODE.lower().strip()
    if mode == "large":
        selected = player_names[:]
    elif mode == "medium":
        k = min(12, len(player_names))
        selected = random.Random(RANDOM_SEED).sample(player_names, k)
    elif mode == "small":
        k = min(5, len(player_names))
        selected = random.Random(RANDOM_SEED).sample(player_names, k)
    else:
        raise ValueError(
            'DATASET_MODE は "small", "medium", "large" のいずれかにしてください。'
        )

    return by_player, selected


def download_zip(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"  既存ZIPを使用: {destination.name}")
        return "既存ファイルを使用"

    part_path = destination.with_suffix(destination.suffix + ".part")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TenhouLogDownloader/1.0)",
        "Referer": RANKING_URL,
        "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
    }

    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            if part_path.exists():
                part_path.unlink()
            request = Request(url, headers=headers)
            downloaded = 0
            next_report = 50 * 1024 * 1024
            with urlopen(request, timeout=120) as response, part_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"    {downloaded / (1024 * 1024):.0f} MB 取得済み...")
                        next_report += 50 * 1024 * 1024

            if not zipfile.is_zipfile(part_path):
                preview = part_path.read_bytes()[:120] if part_path.exists() else b""
                raise RuntimeError(
                    f"取得ファイルがZIP形式ではありません。先頭データ={preview!r}"
                )

            part_path.replace(destination)
            print(
                f"  保存完了: {destination} "
                f"({downloaded / (1024 * 1024):.1f} MB)"
            )
            return "ダウンロード成功"

        except Exception as exc:
            last_error = exc
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            if attempt < RETRY_COUNT:
                wait = attempt * 3
                print(
                    f"  エラー（{attempt}/{RETRY_COUNT}回目）: {exc}\n"
                    f"  {wait}秒後に再試行します"
                )
                time.sleep(wait)

    raise RuntimeError(str(last_error))


def safe_extract(zip_path, destination):
    marker = destination / ".extracted_ok"
    if marker.exists():
        print("  展開済みのためスキップ")
        return

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted_files = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                print(f"  安全のため展開をスキップ: {info.filename}")
                continue

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1

    marker.write_text(
        f"extracted files: {extracted_files}\n",
        encoding="utf-8"
    )
    print(f"  ZIP展開完了: {extracted_files} ファイル")


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    entries = scrape_entries()

    # 全リンクの一覧を保存
    all_rows = []
    for e in entries:
        all_rows.append({
            "player": e["player"],
            "date": e["date"],
            "battles": e["battles"],
            "archive_name": e["archive_name"],
            "url": e["url"],
        })

    write_csv(
        BASE_DIR / "players.csv",
        ["player", "date", "battles", "archive_name", "url"],
        all_rows
    )

    by_player, selected_players = choose_players(entries)
    print(f"\n取得した牌譜DLリンク数: {len(entries)}")
    print(f"プレイヤー数（重複整理後）: {len(by_player)}")
    print(f"抽出モード: {DATASET_MODE} / 選択人数: {len(selected_players)}")
    print("\n選択されたプレイヤー:")

    for name in selected_players:
        print(f"  - {name}（DLリンク {len(by_player[name])}件）")

    selected_rows = [
        {"player": name, "archive_count": len(by_player[name])}
        for name in selected_players
    ]
    write_csv(
        BASE_DIR / "selected_players.csv",
        ["player", "archive_count"],
        selected_rows
    )

    log_path = BASE_DIR / "download_log.csv"
    log_fields = [
        "timestamp", "player", "date", "archive_name",
        "url", "zip_path", "status", "error"
    ]

    total = sum(len(by_player[name]) for name in selected_players)
    completed = 0
    failures = 0

    print(f"\nダウンロード対象ZIP数: {total}")

    for player in selected_players:
        for entry in by_player[player]:
            completed += 1
            filename = (
                Path(urlparse(entry["url"]).path).name
                or entry["archive_name"]
            )
            player_dir = BASE_DIR / "archives" / safe_folder_name(player)
            zip_path = player_dir / filename

            print(
                f"\n[{completed}/{total}] {player} / "
                f"{entry['date']} / {filename}"
            )

            status = ""
            error = ""

            try:
                status = download_zip(entry["url"], zip_path)

                if EXTRACT_ZIPS:
                    extract_dir = (
                        BASE_DIR / "extracted" /
                        safe_folder_name(player) / Path(filename).stem
                    )
                    safe_extract(zip_path, extract_dir)

            except Exception as exc:
                failures += 1
                status = "失敗"
                error = str(exc)
                print(f"  ダウンロード失敗: {error}")

            write_csv(
                log_path,
                log_fields,
                [{
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "player": player,
                    "date": entry["date"],
                    "archive_name": filename,
                    "url": entry["url"],
                    "zip_path": str(zip_path),
                    "status": status,
                    "error": error,
                }],
                append=True,
            )

            time.sleep(DOWNLOAD_DELAY_SECONDS)

    print("\n処理終了")
    print(f"保存先: {BASE_DIR.resolve()}")
    print(f"成功／既存利用: {total - failures}件、失敗: {failures}件")
    print(f"一覧: {BASE_DIR / 'players.csv'}")
    print(f"抽出対象: {BASE_DIR / 'selected_players.csv'}")
    print(f"ログ: {log_path}")

    if failures:
        print(
            "失敗したリンクは download_log.csv の status/error 列を確認し、"
            "後から再実行してください。"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n処理を続行できませんでした:")
        print(exc)
        sys.exit(1)