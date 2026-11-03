# -*- coding: utf-8 -*-
"""名前を指定して保存済みモデルを実行。既定は評価、--action yosokuで1局面予測。"""
import argparse
import csv
import json
import os
from pathlib import Path
from aite_tehai_gakushu import (BASE_DIR,DATA_DIR,load_samples,split_game_ids,
                                format_evaluation,META_COLUMNS,iter_csv_rows,numeric_row)
from paifu_csv_henkan import TRAIN_COLUMNS,TRAIN_LABEL_COLUMNS

# ==================== 実行設定 ====================
MODEL_NAME = "shiken_01"  # 実行したいモデル名だけを変更できます
ACTION = "hyoka"         # "hyoka"=誤差を評価、"yosoku"=1局面の手牌予測

def saved_paths(root,records):
    root=Path(root).resolve()
    paths=[]
    for r in records:
        pair=[]
        for key in ['csv_file','meta_file']:
            p=(root/r[key]).resolve()
            if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'CSVがありません: {p}')
            pair.append(p)
        paths.append((*pair,{k:r.get(k,'') for k in ['game_id','game_date','source_player','source_zip']}))
    return paths


def evaluate(args):
    directory=args.model_dir.resolve()
    config_path=directory/'gakushu_settei.json'
    legacy=not config_path.exists()
    config=json.loads((directory/'metrics.json' if legacy else config_path).read_text())
    names=json.loads((directory/'feature_names.json').read_text())
    expected=[n for n in TRAIN_COLUMNS if n not in META_COLUMNS and n not in TRAIN_LABEL_COLUMNS]
    if names!=expected:raise ValueError('入力列の許可リストが一致しません')
    with (directory/'gakushu_games.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    paths=saved_paths(config['input'],records)
    train_ids,test_ids=split_game_ids(paths,config['seed'])
    if set(config['train_game_ids']) & test_ids or test_ids!=set(config['test_game_ids']):raise ValueError('保存時の対局分割と一致しません')
    cap_train=config['train_rows'] if legacy else config['max_train']
    cap_test=config['test_rows'] if legacy else config['max_test']
    # 旧モデルは旧式の共有乱数による抽出を再現。新モデルは評価側だけを読む。
    features,train,test,nt,nv=load_samples(paths,cap_train,cap_test,config['seed'],split='both' if legacy else 'test')
    if not test:raise ValueError('評価行がありません')
    os.environ.setdefault('KERAS_HOME',str(BASE_DIR/'.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import numpy as np
    import tensorflow as tf
    model=tf.keras.models.load_model(directory/'tehai_yosoku.keras',compile=False)
    x=np.asarray([r[3] for r in test],dtype=np.float32)
    y=np.asarray([r[4] for r in test],dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('評価データに非有限値があります')
    batch=config['batch_size']
    predicted=np.concatenate([model(x[i:i+batch],training=False).numpy() for i in range(0,len(x),batch)])
    if predicted.shape!=y.shape or not np.isfinite(predicted).all():raise ValueError('予測結果が不正です')
    baseline=np.asarray([r[4] for r in train],dtype=np.float32).mean(axis=0) if legacy else np.asarray(config['train_label_mean'],dtype=np.float32)
    mae=float(np.abs(y-predicted).mean())
    base=float(np.abs(y-baseline).mean())
    mse=float(np.square(y-predicted).mean())
    metrics=dict(model_name=directory.name,model_dir=str(directory),mae=mae,mean_baseline_mae=base,
                 mae_improvement=base-mae,mae_reduction_percent=(base-mae)/base*100 if base>0 else None,
                 mse=mse,rmse=float(np.sqrt(mse)),train_rows=config['train_rows'],test_rows=len(test),
                 available_test_rows=nv,train_game_ids=config['train_game_ids'],test_game_ids=sorted(test_ids),
                 game_overlap=0,seed=config['seed'],input=config['input'],
                 evaluated_saved_model=True,legacy_sampling_reproduced=legacy)
    out=args.output_dir or directory/'hyoka'
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'validation_examples.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['game_id','observer','target_player','actual_counts','predicted_counts'])
        for record,yp in zip(test[:20],predicted[:20]):w.writerow([*record[:3],json.dumps(record[4]),json.dumps(yp.tolist())])
    print(f'モデル: {directory.name} / 評価: {len(test)}行・{len(test_ids)}対局',flush=True)
    print(f'評価結果: {(out/"metrics.json").resolve()}',flush=True)
    print(format_evaluation(metrics),flush=True)


def predict(args):
    model_dir=args.model_dir.resolve()
    config_file=model_dir/'gakushu_settei.json'
    metrics=json.loads((config_file if config_file.exists() else model_dir/'metrics.json').read_text())
    names=json.loads((model_dir/'feature_names.json').read_text())
    expected=[n for n in TRAIN_COLUMNS if n not in META_COLUMNS and n not in TRAIN_LABEL_COLUMNS]
    if names!=expected:raise ValueError('モデルの入力列が現在の安全な許可列と一致しません')
    gid=args.game_id or metrics['test_game_ids'][0]
    root=Path(metrics['input']).resolve()
    with (model_dir/'gakushu_games.csv').open(encoding='utf-8-sig',newline='') as f:
        record=next((r for r in csv.DictReader(f) if r['game_id']==gid),None)
    if record is None:raise ValueError('このモデルの学習・評価対象に指定対局がありません')
    paths=[]
    for key in ['csv_file','meta_file']:
        path=(root/record[key]).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'CSVがありません: {path}')
        paths.append(path)
    info={k:record.get(k,'') for k in ['game_id','game_date','source_player','source_zip']}
    row=next((r for r in iter_csv_rows([(paths[0],paths[1],info)])
              if r['state_index']==args.state_index and r['observer']==args.observer
              and r['target_player']==args.target),None)
    if row is None:raise ValueError('指定した局面がありません。--state-index 0 など小さい値を指定してください')
    os.environ.setdefault('KERAS_HOME',str(BASE_DIR/'.keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import numpy as np
    import tensorflow as tf
    model=tf.keras.models.load_model(model_dir/'tehai_yosoku.keras',compile=False)
    x=np.asarray([numeric_row(row,names)],dtype=np.float32)
    prediction=model(x,training=False).numpy()[0]
    if prediction.shape!=(34,) or not np.isfinite(prediction).all():raise ValueError('予測結果が不正です')
    labels=[f'{i}{s}' for s in ['萬','筒','索'] for i in range(1,10)]+['東','南','西','北','白','發','中']
    print(f'対局: {gid}')
    print(f'局面: {args.state_index} / 自分の席: {args.observer} / 相手の席: {args.target}')
    print('予測枚数が多い牌（上位10種。保有確率ではありません）')
    if args.show_answer:
        from aite_tehai_gakushu import label_from_csv
        actual=label_from_csv(row)
    for i in np.argsort(-prediction)[:10]:
        extra=f' / 実際: {actual[i]:.0f}枚' if args.show_answer else ''
        print(f'{labels[i]:>3}: {prediction[i]:.3f}枚{extra}')
    print(f'予測枚数の合計: {prediction.sum():.3f}枚（合法な手牌枚数への制約なし）')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',help='実行するモデル名。省略時は冒頭のMODEL_NAME')
    parser.add_argument('--model-dir',type=Path,help='別の場所にあるモデルを指定')
    parser.add_argument('--action',choices=['hyoka','yosoku'],default=ACTION)
    parser.add_argument('--output-dir',type=Path,help='評価結果の保存先。既定はモデル内のhyoka')
    parser.add_argument('--state-index',type=int,default=30)
    parser.add_argument('--observer',type=int,choices=range(4),default=0)
    parser.add_argument('--target',type=int,choices=range(4),default=1)
    parser.add_argument('--game-id')
    parser.add_argument('--show-answer',action='store_true')
    args=parser.parse_args()
    name=args.name or MODEL_NAME
    if Path(name).name!=name or name in ('.','..') or '\\' in name:
        parser.error('モデル名にはフォルダを含めないでください')
    if args.name and args.model_dir:parser.error('--nameと--model-dirはどちらか一方を指定してください')
    args.model_dir=args.model_dir or DATA_DIR/'models'/name
    if not (args.model_dir/'tehai_yosoku.keras').is_file():
        parser.error(f'モデルがありません: {args.model_dir}')
    if args.action=='hyoka':
        evaluate(args)
    else:
        if args.observer==args.target:parser.error('observerとtargetは異なる席を指定してください')
        if args.state_index<0:parser.error('state-indexは0以上です')
        predict(args)


if __name__=='__main__':
    main()
