import pandas as pd
import re


def reappend_prefix(imgfile):
    prefix = imgfile.split('/')[-1].split('_')[0]
    return '{}--{}'.format(prefix, imgfile)

def is_image(l):
    return re.search('jpg$', l)

def is_bbox(l):
    return len(l.split()) == 10

def get_annotations_df(annotations):
    with open(annotations) as f:
        ants = f.readlines()
    #
    ants = ' '.join(ants)
    ants = re.split(r'[0-9]+--', ants)
    ants = ants[1:]
    #
    dfs = []
    for a in ants:
        elements = a.split('\n')
        x, y, w, h = [], [], [], []
        for el in elements:
            if is_image(el): 
                imgfile = reappend_prefix(el)
            if is_bbox(el):
                x.append(el.split()[0])
                y.append(el.split()[1])
                w.append(el.split()[2])
                h.append(el.split()[3])
        minidf = pd.DataFrame({
                'imgfile': [imgfile] * len(x),
                'x': x, 'y': y, 'w': w, 'h': h
            })
        dfs.append(minidf)
    #
    df = pd.concat(dfs)
    return df

TRAIN = '../../../data/wider_face/wider_face_train_bbx_gt.txt'
VALID = '../../../data/wider_face/wider_face_val_bbx_gt.txt'

train_df = get_annotations_df(TRAIN)
valid_df = get_annotations_df(VALID)

train_df['imgfile'] = ['WIDER_train/images/{}'.format(_) for _ in train_df['imgfile']]
valid_df['imgfile'] = ['WIDER_val/images/{}'.format(_) for _ in valid_df['imgfile']]

df = pd.concat([train_df, valid_df])

df.to_csv('../../../data/wider_face/train.csv', index=False)

