# -*- coding: utf-8 -*-
"""天鳳ZIPから公開情報と自身の手牌だけをX、他家の伏せた手牌をyにするCSVを生成。
標準ライブラリのみ。全人数・全日付を変換し、学習時に人数と達成日を絞ります。
games/<game_id>.csv（添付と同じ451列）、公開情報の.meta.csv、再開用.done.jsonを出力。
参照CSVの全手牌は正解の復元用。入力への混入を学習側で防ぎます。
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import jikken_settei as settei
import csv
import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# S/M/Lの人数・対局数・seedは jikken_settei.py で設定します。
MAX_DISCARDS_PER_PLAYER = 24
MAX_MELD_SLOTS = 4

RED_RAW_TO_CODE = {16: 35, 52: 36, 88: 37}  # 赤5萬/赤5筒/赤5索を通常5と別カテゴリにする
GAME_ID_RE = re.compile(r"(?P<id>(?P<date>20\d{6})(?:\d{2})?gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4,}-[0-9a-fA-F]{8})")

ACTION_INITIAL = 0
ACTION_DRAW = 1
ACTION_DISCARD = 2
ACTION_MELD = 3
ACTION_REACH = 4
ACTION_DORA = 5
MELD_TYPE = {"none": 0, "chi": 1, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 5, "nuki": 6}
MELD_TYPE_NAME = {v: k for k, v in MELD_TYPE.items()}


# ========================== 列定義 ==========================
BASE_COLUMNS = [
    "round", "honba", "kyotaku", "remain_tsumo", "step", "actor",
    "action", "target_tile", "tsumogiri",
]
BASE_COLUMNS += [f"dora_{i}" for i in range(1, 6)]
BASE_COLUMNS += [f"p{i}_score" for i in range(4)]
BASE_COLUMNS += [f"visible_{i}" for i in range(1, 38)]

REFERENCE_COLUMNS = BASE_COLUMNS[:]
for _p in range(4):
    REFERENCE_COLUMNS.append(f"p{_p}_is_reach")
    REFERENCE_COLUMNS += [f"p{_p}_hand_{i}" for i in range(1, 38)]
    REFERENCE_COLUMNS += [f"p{_p}_meld_{i}" for i in range(1, 38)]
    REFERENCE_COLUMNS += [f"p{_p}_disc_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
assert len(REFERENCE_COLUMNS) == 451, f"参照列数が451列ではありません: {len(REFERENCE_COLUMNS)}"

MELD_EXTRA_COLUMNS = []
for _p in range(4):
    MELD_EXTRA_COLUMNS += [
        f"p{_p}_meld_count", f"p{_p}_chi_count", f"p{_p}_pon_count",
        f"p{_p}_daiminkan_count", f"p{_p}_ankan_count", f"p{_p}_kakan_count",
        f"p{_p}_nuki_count",
    ]
    for _slot in range(1, MAX_MELD_SLOTS + 1):
        MELD_EXTRA_COLUMNS += [
            f"p{_p}_meld{_slot}_type", f"p{_p}_meld{_slot}_from_rel",
            f"p{_p}_meld{_slot}_tile1", f"p{_p}_meld{_slot}_tile2",
            f"p{_p}_meld{_slot}_tile3", f"p{_p}_meld{_slot}_tile4",
        ]
RIVER_EXTRA_COLUMNS = []
PUBLIC_STATUS_COLUMNS = []
ANKAN_TRUE_COLUMNS = []
for _p in range(4):
    RIVER_EXTRA_COLUMNS += [f"p{_p}_disc_called_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
    PUBLIC_STATUS_COLUMNS += [f"p{_p}_ippatsu", f"p{_p}_double_riichi", f"p{_p}_seat_wind"]
    # Legacy reference schema only; actual red details in concealed kans are excluded from X.
    ANKAN_TRUE_COLUMNS += [f"p{_p}_ankan_true_{i}" for i in range(1, 38)]
META_COLUMNS = ["game_id", "game_date", "source_player", "source_zip", "state_index", "oya", "game_wind", "hand_number"]
FULL_COLUMNS = REFERENCE_COLUMNS + META_COLUMNS + MELD_EXTRA_COLUMNS + RIVER_EXTRA_COLUMNS + PUBLIC_STATUS_COLUMNS + ANKAN_TRUE_COLUMNS

TRAIN_META_COLUMNS = ["game_id", "game_date", "source_player", "source_zip", "state_index", "observer", "target_player"]
TRAIN_BASE_COLUMNS = [
    "round", "honba", "kyotaku", "remain_tsumo", "step", "actor", "action",
    "target_tile", "target_tile_is_hidden", "tsumogiri",
]
TRAIN_BASE_COLUMNS += [f"dora_{i}" for i in range(1, 6)]
TRAIN_BASE_COLUMNS += [f"p{i}_score" for i in range(4)]
TRAIN_BASE_COLUMNS += [f"visible_{i}" for i in range(1, 38)]
TRAIN_PUBLIC_COLUMNS = []
for _p in range(4):
    TRAIN_PUBLIC_COLUMNS.append(f"p{_p}_is_reach")
    TRAIN_PUBLIC_COLUMNS += [f"p{_p}_meld_{i}" for i in range(1, 38)]
    TRAIN_PUBLIC_COLUMNS += [f"p{_p}_disc_{i}" for i in range(1, MAX_DISCARDS_PER_PLAYER + 1)]
TRAIN_PUBLIC_COLUMNS += ["oya", "game_wind", "hand_number"] + MELD_EXTRA_COLUMNS + RIVER_EXTRA_COLUMNS + PUBLIC_STATUS_COLUMNS
TRAIN_LABEL_COLUMNS = [f"target_hand_{i}" for i in range(1, 38)]
TRAIN_COLUMNS = TRAIN_META_COLUMNS + TRAIN_BASE_COLUMNS + TRAIN_PUBLIC_COLUMNS + [f"observer_hand_{i}" for i in range(1, 38)] + TRAIN_LABEL_COLUMNS

REFERENCE_EXTRA_COLUMNS = ['state_index'] + [c for c in TRAIN_PUBLIC_COLUMNS if c not in REFERENCE_COLUMNS]


def reference_to_training(ref, extra, meta):
    """参照CSVの全手牌を無条件に入力に渡さず、許可列から自分視点を構築。"""
    public_names = TRAIN_BASE_COLUMNS + TRAIN_PUBLIC_COLUMNS
    for observer in range(4):
        row = {name:int(extra.get(name,ref.get(name,0))) for name in public_names}
        row.update({k:meta[k] for k in ['game_id','game_date','source_player','source_zip']}, state_index=int(extra['state_index']),observer=observer)
        hidden = row['action']==ACTION_DRAW and row['actor']!=observer
        row['target_tile_is_hidden'] = int(hidden)
        if hidden: row['target_tile']=0
        for t in range(1,38):
            own=int(ref[f'p{observer}_hand_{t}'])
            row[f'observer_hand_{t}']=own
            row[f'visible_{t}']=int(ref[f'visible_{t}'])+own
        for target in range(4):
            if target==observer:continue
            out=row.copy()
            out['target_player']=target
            for t in range(1,38):out[f'target_hand_{t}']=int(ref[f'p{target}_hand_{t}'])
            yield out


@lru_cache(maxsize=2)
def cached_zip(path):
    return zipfile.ZipFile(path)


def convert_one(job):
    c, out_dir, resume = job
    out=Path(out_dir)
    stem=c['game_id']
    final=out/'games'/f'{stem}.csv'
    extra=out/'games'/f'{stem}.meta.csv'
    done=out/'games'/f'{stem}.done.json'
    tmp=final.with_suffix('.csv.partial')
    xtmp=extra.with_suffix('.csv.partial')
    try:
        if c['source_zip']:
            z=cached_zip(c['source_zip'])
            info=z.getinfo(c['member'])
            fingerprint=[info.CRC,info.file_size]
        else:
            fingerprint=hashlib.sha256(Path(c['member']).read_bytes()).hexdigest()
        if resume and final.is_file() and extra.is_file() and done.is_file():
            marker=json.loads(done.read_text())
            if (marker.get('schema_version')==4 and marker.get('fingerprint')==fingerprint
                and marker.get('csv_size')==final.stat().st_size and marker.get('meta_size')==extra.stat().st_size):
                return dict(c,csv_file=str(final.relative_to(out)),meta_file=str(extra.relative_to(out)),
                            training_rows=marker['rows'],resumed=True),None
        data=z.read(c['member']) if c['source_zip'] else Path(c['member']).read_bytes()
        rows=0
        with tmp.open('w',encoding='utf-8-sig',newline='',buffering=1024*1024) as f, xtmp.open('w',encoding='utf-8-sig',newline='',buffering=1024*1024) as xf:
            w=csv.writer(f);xw=csv.writer(xf)
            w.writerow(REFERENCE_COLUMNS);xw.writerow(REFERENCE_EXTRA_COLUMNS)
            for r in snapshots(read_xml(data),reference=True):
                w.writerow([r[name] for name in REFERENCE_COLUMNS])
                xw.writerow([r[name] for name in REFERENCE_EXTRA_COLUMNS])
                rows+=1
        if not rows:raise ValueError('局面が0件です')
        tmp.replace(final);xtmp.replace(extra)
        marker=dict(schema_version=4,fingerprint=fingerprint,rows=rows,csv_size=final.stat().st_size,meta_size=extra.stat().st_size)
        dtmp=done.with_suffix('.json.partial')
        dtmp.write_text(json.dumps(marker))
        dtmp.replace(done)
        return dict(c,csv_file=str(final.relative_to(out)),meta_file=str(extra.relative_to(out)),training_rows=rows,resumed=False),None
    except Exception as exc:
        tmp.unlink(missing_ok=True);xtmp.unlink(missing_ok=True)
        done.unlink(missing_ok=True)
        return c,dict(game_id=stem,error=f'{type(exc).__name__}: {exc}')


def convert_jobs(jobs,workers):
    if workers==1:
        for job in jobs:yield convert_one(job)
        return
    # 同時投入数を制限し、15万件のFutureをメモリに溜めない。
    executor=ProcessPoolExecutor(max_workers=workers)
    pending=set()
    iterator=iter(jobs)
    try:
        for _ in range(workers*2):
            job=next(iterator,None)
            if job is not None:pending.add(executor.submit(convert_one,job))
        while pending:
            ready,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                yield future.result()
                job=next(iterator,None)
                if job is not None:pending.add(executor.submit(convert_one,job))
    finally:
        executor.shutdown(wait=True,cancel_futures=True)


def tile(raw):
    if not 0 <= raw < 136:
        raise ValueError(f'不正な牌ID: {raw}')
    return RED_RAW_TO_CODE.get(raw, raw // 4 + 1)


def counts(raws):
    c = Counter(tile(t) for t in raws)
    return [c[i] for i in range(1, 38)]


def decode_meld(m, who):
    rel = m & 3
    source = (who + rel) % 4
    if m & 4:
        v = (m >> 10) & 63
        called = v % 3
        base = (v // 3 // 7 * 9 + v // 3 % 7) * 4
        ts = [base + 4*i + ((m >> (3+2*i)) & 3) for i in range(3)]
        return 'chi', ts, ts[called], source
    if m & 24:
        v = (m >> 9) & 127
        base = v // 3 * 4
        unused = (m >> 5) & 3
        ts = [base+i for i in range(4) if i != unused]
        if m & 8:
            return 'pon', ts, ts[v % 3], source
        return 'kakan', ts + [base+unused], base+unused, source
    if m & 32:
        raise ValueError('三人麻雀の抜きドラは対象外')
    v = (m >> 8) & 255
    ts = list(range(v // 4 * 4, v // 4 * 4 + 4))
    return ('ankan' if rel == 0 else 'daiminkan'), ts, v, source


def read_xml(data):
    if data.startswith(b'\x1f\x8b'):
        data = gzip.decompress(data)
    root = ET.fromstring(data)
    if root.tag != 'mjloggm':
        raise ValueError(f'牌譜XMLではありません: {root.tag}')
    go = root.find('GO')
    if go is not None and int(go.get('type', '0')) & 16:
        raise ValueError('三人麻雀は対象外')
    return root


def snapshots(root, reference=False):
    """逐次更新のみ。終局情報・未来のイベントを参照しない。"""
    active = False
    state_index = 0
    for e in root:
        tag = e.tag
        a = e.attrib
        actor, action, raw, tsumogiri = -1, None, None, 0
        if tag == 'INIT':
            seed = list(map(int, a['seed'].split(',')))
            hands = [set(map(int, a[f'hai{p}'].split(','))) for p in range(4)]
            if any(len(h) != 13 for h in hands):
                raise ValueError('配牌が13枚ではありません')
            round_no, honba, kyotaku = seed[:3]
            oya = int(a['oya'])
            scores = [int(v)*100 for v in a['ten'].split(',')]
            doras = [seed[5]]
            rivers, melds = [[] for _ in range(4)], [[] for _ in range(4)]
            reaches, ippatsu, double = [0]*4, [0]*4, [0]*4
            last_draw, pending_reach = [None]*4, [False]*4
            remain, step = 70, 0
            active, action = True, ACTION_INITIAL
        elif not active:
            continue
        elif re.fullmatch(r'[TUVWDEFGtuvwdefg]\d+', tag):
            letter = tag[0].upper()
            raw = int(tag[1:])
            tile(raw)
            if letter in 'TUVW':
                actor, action = 'TUVW'.index(letter), ACTION_DRAW
                if raw in hands[actor]:
                    raise ValueError('重複したツモ牌')
                hands[actor].add(raw)
                last_draw[actor] = raw
                remain -= 1
            else:
                actor, action = 'DEFG'.index(letter), ACTION_DISCARD
                if raw not in hands[actor]:
                    raise ValueError(f'手牌にない打牌: seat={actor}, tile={raw}')
                tsumogiri = int(last_draw[actor] == raw)
                hands[actor].remove(raw)
                rivers[actor].append([raw, tsumogiri, 0])
                last_draw[actor] = None
                if reaches[actor] and not pending_reach[actor]:
                    ippatsu[actor] = 0
                pending_reach[actor] = False
        elif tag == 'N':
            actor, action = int(a['who']), ACTION_MELD
            kind, ts, called, source = decode_meld(int(a['m']), actor)
            if kind == 'kakan':
                old = next((v for v in melds[actor] if v['kind'] == 'pon' and v['tiles'][0]//4 == called//4), None)
                if old is None:
                    raise ValueError('加槓に対応するポンがありません')
                hands[actor].remove(called)
                old.update(kind=kind, tiles=ts)
            else:
                remove = ts if kind == 'ankan' else [t for t in ts if t != called]
                for t in remove:
                    hands[actor].remove(t)
                if kind != 'ankan':
                    if not rivers[source] or rivers[source][-1][0] != called:
                        raise ValueError('鳴き牌が直前の河と一致しません')
                    rivers[source][-1][2] = 1
                melds[actor].append(dict(kind=kind, tiles=ts, source=source))
            last_draw[actor] = None
            ippatsu = [0]*4
            raw = called if kind != 'ankan' else None
        elif tag == 'REACH':
            actor, action = int(a['who']), ACTION_REACH
            if a['step'] == '1':
                reaches[actor] = 1
                double[actor] = int(not rivers[actor] and not any(melds))
                pending_reach[actor] = True
            elif a['step'] == '2':
                scores[actor] -= 1000
                kyotaku += 1
                ippatsu[actor] = 1
        elif tag == 'DORA':
            action, raw = ACTION_DORA, int(a['hai'])
            doras.append(raw)
        elif tag in ('AGARI', 'RYUUKYOKU'):
            active = False
            continue
        else:
            continue
        if remain < 0 or any(len(r) > MAX_DISCARDS_PER_PLAYER for r in rivers):
            raise ValueError('局面が対応範囲外です')
        all_raw = [t for h in hands for t in h] + [t for ms in melds for m in ms for t in m['tiles']] + [r[0] for rs in rivers for r in rs if not r[2]] + doras
        if len(all_raw) != len(set(all_raw)):
            raise ValueError('局面内に同一の物理牌が重複しています')
        base = dict(round=round_no, honba=honba, kyotaku=kyotaku, remain_tsumo=remain,
                    step=step, actor=actor, action=action, target_tile=tile(raw) if raw is not None else 0,
                    tsumogiri=tsumogiri, oya=oya, game_wind=round_no//4+1, hand_number=round_no%4+1,
                    state_index=state_index)
        for i in range(1, 6):
            base[f'dora_{i}'] = tile(doras[i-1]) if i <= len(doras) else 0
        public = set(doras) | {r[0] for rs in rivers for r in rs}
        for p in range(4):
            base.update({f'p{p}_score': scores[p], f'p{p}_is_reach': reaches[p],
                         f'p{p}_ippatsu': ippatsu[p], f'p{p}_double_riichi': double[p],
                         f'p{p}_seat_wind': (p-oya)%4+1, f'p{p}_meld_count': len(melds[p])})
            public_meld = []
            for kind in MELD_TYPE:
                if kind != 'none':
                    base[f'p{p}_{kind}_count'] = sum(m['kind'] == kind for m in melds[p])
            for slot in range(1, 5):
                m = melds[p][slot-1] if slot <= len(melds[p]) else None
                vals = []
                if m:
                    # 暗槓の牌種は公開。赤牌の有無は他家視点で隠す。
                    vals = [t//4+1 if m['kind'] == 'ankan' else tile(t) for t in m['tiles']]
                    if m['kind'] != 'ankan':
                        public.update(m['tiles'])
                    public_meld.extend(vals)
                base[f'p{p}_meld{slot}_type'] = MELD_TYPE[m['kind']] if m else 0
                base[f'p{p}_meld{slot}_from_rel'] = (m['source']-p)%4 if m else 0
                for j in range(1, 5):
                    base[f'p{p}_meld{slot}_tile{j}'] = vals[j-1] if j <= len(vals) else 0
            mc = Counter(public_meld)
            for t in range(1, 38):
                base[f'p{p}_meld_{t}'] = mc[t]
            for i in range(1, MAX_DISCARDS_PER_PLAYER+1):
                r = rivers[p][i-1] if i <= len(rivers[p]) else None
                base[f'p{p}_disc_{i}'] = tile(r[0])*(-1 if r[1] else 1) if r else 0
                base[f'p{p}_disc_called_{i}'] = r[2] if r else 0
        if reference:
            # 添付と同じ451列。見える牌は公開分のみとし、私有手牌を混ぜない。
            ref = base.copy()
            visible = Counter(tile(t) for t in public)
            for ms in melds:
                for m in ms:
                    if m['kind'] == 'ankan':
                        visible[m['tiles'][0]//4+1] += 4
            for p in range(4):
                for t,n in enumerate(counts(hands[p]),1):
                    ref[f'p{p}_hand_{t}'] = n
            for t in range(1,38):
                ref[f'visible_{t}'] = visible[t]
            yield ref
            state_index += 1
            step += 1
            continue
        for observer in range(4):
            visible = Counter(tile(t) for t in public | hands[observer])
            for ms in melds:
                for m in ms:
                    if m['kind'] == 'ankan':
                        visible[m['tiles'][0]//4+1] += 4
            row = base.copy()
            row['observer'] = observer
            hidden = action == ACTION_DRAW and actor != observer
            row['target_tile_is_hidden'] = int(hidden)
            if hidden:
                row['target_tile'] = 0
            for t, n in enumerate(counts(hands[observer]), 1):
                row[f'observer_hand_{t}'] = n
            for t in range(1, 38):
                row[f'visible_{t}'] = visible[t]
            for target in range(4):
                if observer == target:
                    continue
                out = row.copy()
                out['target_player'] = target
                for t, n in enumerate(counts(hands[target]), 1):
                    out[f'target_hand_{t}'] = n
                yield out
        state_index += 1
        step += 1


def write_table(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='MJLOGをCSVへ変換。モデル作成を優先する場合は --rank-policy after を指定')
    parser.add_argument('--rank-policy',choices=['all','after'],default='all',help='all=全日付、after=達成翌日以降だけ変換')
    parser.add_argument('--base-dir', type=Path, default=settei.DATA_DIR)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1),help='並列処理数。既定は最大4')
    parser.add_argument('--resume',action='store_true',help='完成済みCSVを再利用して途中から続行')
    parser.add_argument('--check-log', type=Path, help='単独MJLOGの診断。正式な全件データとは別の出力先を指定')
    args = parser.parse_args()
    args.mode, args.seed, args.max_games = 'ALL', None, None
    player_count = None
    out = args.output_dir or args.base_dir/'converted'/args.rank_policy
    if args.workers<1:parser.error('--workersは1以上です')
    if ((out/'summary.json').exists() or (out/'games').exists()) and not args.resume:
        parser.error(f'既存データがあります。続きは --resume、別保存先は --output-dir: {out}')
    if args.resume and (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if old.get('schema_version')!=4:parser.error('古い形式は途中再開できません。別の保存先を指定してください')
    # 再実行中のデータを完了済みとして学習側に読ませない。
    (out/'summary.json').unlink(missing_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'games').mkdir(exist_ok=True)
    print(f'変換開始: mode={args.mode}, rank-policy={args.rank_policy}, 出力={out}', flush=True)
    players, candidates, logs, errors = [], [], [], []
    if args.check_log:
        candidates = [dict(game_id=GAME_ID_RE.search(args.check_log.name).group('id') if GAME_ID_RE.search(args.check_log.name) else hashlib.sha256(args.check_log.read_bytes()).hexdigest(), game_date='', source_player='', source_zip='', member=str(args.check_log))]
    else:
        with (args.base_dir/'download_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'player', 'achievement_date', 'zip_filename'}
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f'manifestに必要な列: {sorted(required)}')
            players = list(reader)
        if len({p['player'] for p in players}) != len(players):
            raise ValueError('manifestのプレイヤーが重複しています')
        players.sort(key=lambda p: p['player'])
        write_table(out/'selected_players.csv', players, list(players[0]) if players else ['player'])
        seen = set()
        for p in players:
            achieved = date.fromisoformat(p['achievement_date']) if args.rank_policy == 'after' else None
            zp = args.base_dir/'archives'/Path(p['zip_filename']).name
            print(f'ZIP検査: {p["player"]} / {zp.name}', flush=True)
            with zipfile.ZipFile(zp) as z:
                for name in sorted(z.namelist()):
                    if name.endswith('/'):
                        continue
                    match = GAME_ID_RE.search(name)
                    gid = match.group('id') if match else ''
                    gd = date.fromisoformat(match.group('date')[:4]+'-'+match.group('date')[4:6]+'-'+match.group('date')[6:8]) if match else None
                    reason = 'eligible'
                    if gd is None:
                        reason = 'invalid_game_id'
                    elif achieved and gd <= achieved:
                        reason = 'on_or_before_achievement_date'
                    elif gid in seen:
                        reason = 'duplicate_game_id'
                    record = dict(game_id=gid, game_date=gd.isoformat() if gd else '', source_player=p['player'], source_zip=str(zp), member=name)
                    logs.append(dict(record, reason=reason))
                    if reason == 'eligible':
                        seen.add(gid)
                        candidates.append(record)
    logcols = ['game_id', 'game_date', 'source_player', 'source_zip', 'member']
    write_table(out/'logs_manifest.csv', logs, logcols+['reason'])
    converted, rows, selected = 0, 0, []
    jobs=((c,str(out),args.resume) for c in candidates)
    for result,error in convert_jobs(jobs,args.workers):
        if error:
            errors.append(error)
            print(f'変換失敗: {error}',file=sys.stderr,flush=True)
            continue
        rows+=result['training_rows'];converted+=1
        resumed=result.pop('resumed')
        selected.append(result)
        status='再利用' if resumed else '変換成功'
        print(f'{status}: {converted}/{len(candidates)}対局 / {rows:,}局面 / {result["game_id"]}',flush=True)
    selected.sort(key=lambda r:r['game_id'])
    write_table(out/'selected_games.csv',selected,logcols+['csv_file','meta_file','training_rows'])
    write_table(out/'conversion_errors.csv', errors, ['game_id', 'error'])
    summary = dict(schema_version=4, layout="reference451_csv", workers=args.workers, mode=args.mode, seed=args.seed, rank_policy=args.rank_policy,
                   check_log=bool(args.check_log), date_rule=('game_date > achievement_date' if args.rank_policy=='after' else 'all dates saved; achievement filter applied by trainer'),
                   requested_players=player_count, max_games=args.max_games, selected_players=len(players), discovered_logs=len(logs),
                   exclusion_counts=dict(Counter(r['reason'] for r in logs)),
                   candidate_games_after_date_filter=len(candidates), converted_games=converted,
                   training_rows=rows, row_unit="one state per row; 12 training samples generated at training time", conversion_errors=len(errors), full_reference_written=True, contains_hidden_hands=True,
                   output_dir=str(out.resolve()))
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not converted or errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
