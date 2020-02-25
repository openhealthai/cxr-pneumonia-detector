import pandas as pd
import numpy as np
import pickle
import pydicom
import re
import os.path as osp

from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else StratifiedKFold
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

# Triple-annotated by STR radiologists
with open('stage_1_patientIds.txt', 'r') as f: 
    stage_1_train_ids = [line.strip() for line in f.readlines()]


df = pd.read_csv('../../data/stage_2_train_labels.csv')
df['x1'], df['y1'], df['x2'], df['y2'] = df['x'], df['y'], df['x']+df['width'], df['y']+df['height']

test_df = df[~df['patientId'].isin(stage_1_train_ids)].drop_duplicates().reset_index(drop=True)
df = df[df['patientId'].isin(stage_1_train_ids)].drop_duplicates().reset_index(drop=True)

pt_df = df[['patientId','Target']].drop_duplicates().reset_index(drop=True)
pt_df = create_double_cv(pt_df, 'patientId', 10, 10, stratified='Target')
df = df.merge(pt_df, on=['patientId','Target'])


# Now, we need to turn it into MMDetection format ...
inner_cols = [col for col in df.columns if re.search(r'inner[0-9]+', col)]
annotations = []
for ptId in tqdm(df['patientId'].unique(), total=len(df['patientId'].unique())):
    _df = df[df['patientId'] == ptId]
    _fp = '{}.dcm'.format(ptId)
    #tmp_dcm = pydicom.dcmread(osp.join('../data/dicoms/train/', _fp), stop_before_pixels=True)
    cv_splits = {col : _df[inner_cols].drop_duplicates()[col].iloc[0] for col in inner_cols}
    cv_splits['outer'] = _df['outer'].iloc[0]
    if _df['Target'].iloc[0] == 1:
        bboxes = np.asarray(_df[['x1','y1','x2','y2']])
        labels = np.asarray([1] * len(_df))
        img_class = 1
    elif _df['Target'].iloc[0] == 0:
        assert len(_df) == 1
        bboxes = np.asarray([])
        labels = np.asarray([])
        img_class = 0
    tmp_dict = {
        'filename': _fp,
        'height': 1024, #tmp_dcm.Rows,
        'width':  1024 ,#tmp_dcm.Columns,
        'ann': {
            'bboxes': bboxes,
            'labels': labels
        },
        'img_class': img_class,
        'cv_splits': cv_splits
    }
    assert len(tmp_dict['ann']['bboxes']) == len(tmp_dict['ann']['labels'])
    annotations.append(tmp_dict)

with open('../../data/train_bbox_annotations_with_splits.pkl', 'wb') as f:
    pickle.dump(annotations, f)
