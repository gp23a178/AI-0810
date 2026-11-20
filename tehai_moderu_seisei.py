"""新しい相手手牌推定モデルを学習する。

使い方: python tehai_moderu_seisei.py
既定ではLモードで学習し、miekazu_02として保存する。
学習条件は --mode や --epochs などで上書きできる。
"""

from aite_tehai_gakushu import main


if __name__ == '__main__':
    main(default_mode='L', default_name='miekazu_02')
