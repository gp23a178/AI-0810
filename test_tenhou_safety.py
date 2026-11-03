from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
base = Path(__file__).resolve().parent
# 以下は実装をなぞるテストではなく、隠れた牌と未来に対する不変性の検査。
import sys
sys.path.insert(0, str(base))
import convert_tenhou_safe as c
import train_hand_predictor_safe as t


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

    def test_real_csv_split(self):
        path = base/'tenhou_data/converted/S/training_samples.csv.gz'
        if not path.exists():
            self.skipTest('実変換CSVがありません')
        features, train, test, ntrain, ntest = t.load_samples(path, 20, 10, 42)
        self.assertTrue(train and test)
        self.assertFalse({r[0] for r in train} & {r[0] for r in test})
        self.assertIn('observer',features)
        self.assertIn('target_player',features)
        self.assertEqual(ntrain+ntest, __import__('json').loads((path.parent/'summary.json').read_text())['training_rows'])

    def test_feature_allowlist_and_split(self):
        features = [n for n in c.TRAIN_COLUMNS if n not in t.META_COLUMNS and n not in c.TRAIN_LABEL_COLUMNS]
        self.assertIn('observer', features)
        self.assertIn('target_player', features)
        self.assertFalse(any('target_hand' in n or '_hand_' in n and not n.startswith('observer_') for n in features))
        self.assertEqual(t.belongs_to_test('a'), t.belongs_to_test('a'))


if __name__=='__main__':
    unittest.main()
