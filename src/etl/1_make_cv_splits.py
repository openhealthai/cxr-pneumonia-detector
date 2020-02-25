import pandas as pd
import numpy as np
import pickle
import cv2
import re
import os.path as osp

from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else MultilabelStratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df

df = pd.read_csv('../../../data/wider_face/train.csv')
img_df = df[['imgfile']].drop_duplicates().reset_index(drop=True)
img_df = create_double_cv(img_df, 'imgfile', 10, 10, stratified=None)
df = df.merge(img_df, on='imgfile')
df['x1'], df['y1'], df['x2'], df['y2'] = df['x'], df['y'], df['x']+df['w'], df['y']+df['h']

# Now, we need to turn it into MMDetection format ...
inner_cols = [col for col in df.columns if re.search(r'inner[0-9]+', col)]
annotations = []
for _fp, _df in tqdm(df.groupby('imgfile'), total=len(df['imgfile'].unique())):
    cv_splits = {col : _df[inner_cols].drop_duplicates()[col].iloc[0] for col in inner_cols}
    cv_splits['outer'] = _df['outer'].iloc[0]
    tmp_img = cv2.imread(osp.join('../../../data/wider_face', _fp))
    tmp_dict = {
        'filename': _fp,
        'height': tmp_img.shape[0],
        'width':  tmp_img.shape[1],
        'ann': {
            'bboxes': np.asarray(_df[['x1','y1','x2','y2']]),
            'labels': np.asarray([1] * len(_df))
        },
        'img_class': 1,
        'cv_splits': cv_splits
    }
    assert len(tmp_dict['ann']['bboxes']) == len(tmp_dict['ann']['labels'])
    annotations.append(tmp_dict)

with open('../../../data/wider_face/train_bbox_annotations_with_splits.pkl', 'wb') as f:
    pickle.dump(annotations, f)
