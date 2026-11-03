from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
base = Path(__file__).resolve().parent
# 以下は実装をなぞるテストではなく、隠れた牌と未来に対する不変性の検査。
import sys
sys.path.insert(0, str(base))
import paifu_csv_henkan as c
import aite_tehai_gakushu as t


def fixture(extra=''):
    hands = ['0,1,2,3,4,5,6,7,8,9,10,11,12',
             '16,17,18,19,20,21,22,23,24,25,26,27,28',
             '32,33,34,35,36,37,38,39,40,41,42,43,44',
             '48,49,50,51,52,53,54,55,56,57,58,59,60']
    attrs = ' '.join(f'hai{i}="{h}"' for i,h in enumerate(hands))
    return f'<mjloggm><GO type="9"/><INIT seed="0,0,0,0,0,132" ten="250,250,250,250" oya="0" {attrs}/><U80/><E80/>{extra}</mjloggm>'


class SafetyTests(unittest.TestCase):
    def features(self, rows):
        return [{k:v for k,v in r.items() if not k.startswith('target_hand_')} for r in rows if r['observer']==0]

    def test_opponent_draw_is_hidden(self):
        a = list(c.snapshots(ET.fromstring(fixture())))
        b = list(c.snapshots(ET.fromstring(fixture().replace('U80','U84').replace('E80','E84'))))
        # 初期局面と他家ツモ直後は同じ入力。公開打牌後は異なってよい。
        self.assertEqual(self.features(a[:24]), self.features(b[:24]))
        self.assertNotEqual(a[12]['target_hand_21'], b[12]['target_hand_21'])

    def test_hidden_initial_hand(self):
        a = list(c.snapshots(ET.fromstring(fixture())))
        b = list(c.snapshots(ET.fromstring(fixture().replace('16,17,18,19','64,65,66,67'))))
        self.assertEqual(self.features(a), self.features(b))

    def test_future_does_not_change_prefix(self):
        a = list(c.snapshots(ET.fromstring(fixture())))
        b = list(c.snapshots(ET.fromstring(fixture('<DORA hai="128"/><AGARI doraHaiUra="124"/>'))))
        self.assertEqual(a, b[:len(a)])

    def test_bad_discard_rejected(self):
        with self.assertRaises(ValueError):
            list(c.snapshots(ET.fromstring(fixture('<D120/>'))))

    def test_ankan_red_detail_hidden(self):
        # 暗槓の赤5情報は公開特徴量では通常5に統合。
        rows = list(c.snapshots(ET.fromstring(fixture('<N who="1" m="4096"/>'))))
        r = next(r for r in rows[36:] if r['observer']==0)
        self.assertEqual(r['p1_meld_5'],4)
        self.assertEqual(r['p1_meld_35'],0)
        self.assertEqual(r['visible_35'],0)

    def test_legacy_game_id(self):
        self.assertEqual(c.GAME_ID_RE.search('20061028gm-0001-0000-134c5173&tw=0.mjlog').group('date'), '20061028')

    def test_saved_csv_selection_and_group_split(self):
        # 合成CSVのみ。実牌譜の変換やTensorFlow学習は行わない。
        import csv,gzip,json,tempfile
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            d=Path(tmp)
            (d/'games').mkdir()
            def write(name,rows):
                with (d/name).open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]))
                    w.writeheader();w.writerows(rows)
            write('selected_players.csv',[
                dict(player='A',achievement_date='2020-01-01'),
                dict(player='B',achievement_date='2022-01-01')])
            games=[]
            for gid,day in [('old','2019-01-01'),('shared','2021-01-01'),('new','2023-01-01')]:
                games.append(dict(game_id=gid,game_date=day,source_player='B',source_zip='',
                                  csv_file=f'games/{gid}.csv',meta_file=f'games/{gid}.meta.csv'))
                refs=list(c.snapshots(ET.fromstring(fixture()),reference=True))
                with (d/'games'/f'{gid}.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.writer(f);w.writerow(c.REFERENCE_COLUMNS)
                    w.writerows([[r[n] for n in c.REFERENCE_COLUMNS] for r in refs])
                with (d/'games'/f'{gid}.meta.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.writer(f);w.writerow(c.REFERENCE_EXTRA_COLUMNS)
                    w.writerows([[r[n] for n in c.REFERENCE_EXTRA_COLUMNS] for r in refs])
            write('selected_games.csv',games)
            write('logs_manifest.csv',[
                dict(game_id='old',game_date='2019-01-01',source_player='A',reason='eligible'),
                # 元の保存元Bは達成前だが、Aからみれば達成後の共有対局。
                dict(game_id='shared',game_date='2021-01-01',source_player='B',reason='eligible'),
                dict(game_id='shared',game_date='2021-01-01',source_player='A',reason='duplicate_game_id'),
                dict(game_id='new',game_date='2023-01-01',source_player='B',reason='eligible')])
            summary=dict(schema_version=4,layout='reference451_csv',check_log=False,conversion_errors=0,
                         converted_games=3,candidate_games_after_date_filter=3)
            (d/'summary.json').write_text(json.dumps(summary))
            paths,players,selected=t.select_csvs(d,'L',42,players_override=0)
            self.assertEqual({r['game_id'] for r in selected},{'shared','new'})
            self.assertEqual(len(paths),2)
            features,train,test,nt,nv=t.load_samples(paths,10,10,42)
            self.assertEqual(nt+nv,72)
            self.assertFalse({r[0] for r in train}&{r[0] for r in test})
            self.assertNotIn('target_hand_1',features)
            with self.assertRaises(ValueError):
                t.select_csvs(d,'L',42,players_override=0,max_games_override=1)
            games[0]['csv_file']='../outside.csv.gz'
            # 達成後のCSVパスが保存フォルダ外へ出る場合は拒否する。
            games[1]['csv_file']='../outside.csv.gz'
            write('selected_games.csv',games)
            with self.assertRaises(ValueError): t.select_csvs(d,'L',42,players_override=0)

    def test_reference_roundtrip_and_resume(self):
        import csv,json,tempfile
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            d=Path(tmp);(d/'games').mkdir()
            raw=d/'synthetic.mjlog';raw.write_text(fixture('<N who="1" m="4096"/>'))
            record=dict(game_id='synthetic',game_date='2020-01-01',source_player='A',source_zip='',member=str(raw))
            result,error=c.convert_one((record,str(d),False))
            self.assertIsNone(error)
            with (d/result['csv_file']).open(encoding='utf-8-sig',newline='') as f:
                self.assertEqual(next(csv.reader(f)),c.REFERENCE_COLUMNS)
            paths=[(d/result['csv_file'],d/result['meta_file'],record)]
            actual=list(t.iter_csv_rows(paths))
            expected=list(c.snapshots(ET.fromstring(raw.read_text())))
            for old,new in zip(expected,actual):
                old.update({k:record[k] for k in ['game_id','game_date','source_player','source_zip']})
                self.assertEqual({n:old[n] for n in c.TRAIN_COLUMNS},new)
            self.assertEqual(len(expected),len(actual))
            result2,error=c.convert_one((record,str(d),True))
            self.assertIsNone(error);self.assertTrue(result2['resumed'])
            # 途中ファイルを消した場合は再利用せず再作成する。
            (d/result['meta_file']).unlink()
            result3,error=c.convert_one((record,str(d),True))
            self.assertIsNone(error);self.assertFalse(result3['resumed'])

    def test_parallel_csv_matches_serial(self):
        import tempfile,zipfile,gzip
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            d=Path(tmp)
            archive=d/'synthetic.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('test.mjlog',gzip.compress(fixture().encode()))
            jobs=[]
            for sub in ['serial','parallel']:
                output=d/sub;(output/'games').mkdir(parents=True)
                group=[]
                for gid in ['one','two']:
                    record=dict(game_id=gid,game_date='2020-01-01',source_player='A',source_zip=str(archive),member='test.mjlog')
                    group.append((record,str(output),False))
                jobs.append(group)
            sequential=list(c.convert_jobs(jobs[0],1))
            parallel=list(c.convert_jobs(jobs[1],2))
            self.assertTrue(all(error is None for _,error in sequential+parallel))
            for gid in ['one','two']:
                for ext in ['csv','meta.csv']:
                    self.assertEqual((d/'serial/games'/f'{gid}.{ext}').read_bytes(),
                                     (d/'parallel/games'/f'{gid}.{ext}').read_bytes())

    def test_failed_reference_not_published(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            d=Path(tmp);(d/'games').mkdir()
            raw=d/'bad.mjlog';raw.write_text(fixture('<D120/>'))
            record=dict(game_id='bad',source_zip='',member=str(raw))
            result,error=c.convert_one((record,str(d),False))
            self.assertIsNotNone(error)
            self.assertFalse((d/'games/bad.csv').exists())
            self.assertFalse((d/'games/bad.done.json').exists())
            self.assertFalse(list((d/'games').glob('*.partial')))

    def test_size_settings(self):
        import jikken_settei as settings
        self.assertEqual(settings.size_settings('s'), settings.size_settings('S'))
        original = settings.SIZE_SETTINGS['S'].copy()
        try:
            settings.SIZE_SETTINGS['S']['players'] = 2
            self.assertEqual(settings.size_settings('S')['players'],2)
            settings.SIZE_SETTINGS['S']['players'] = 0
            with self.assertRaises(ValueError): settings.size_settings('S')
        finally:
            settings.SIZE_SETTINGS['S'] = original

    def test_feature_allowlist_and_split(self):
        features = [n for n in c.TRAIN_COLUMNS if n not in t.META_COLUMNS and n not in c.TRAIN_LABEL_COLUMNS]
        self.assertIn('observer', features)
        self.assertIn('target_player', features)
        self.assertFalse(any('target_hand' in n or '_hand_' in n and not n.startswith('observer_') for n in features))


if __name__=='__main__':
    unittest.main()
