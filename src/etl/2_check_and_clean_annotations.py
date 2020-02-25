import pickle
import numpy as np

PICKLEPATH = '../../../data/wider_face/train_bbox_annotations_with_splits.pkl'

with open(PICKLEPATH, 'rb') as f:
    ants = pickle.load(f)


for ind, a in enumerate(ants):
    bbox = a['ann']['bboxes']
    valid = [(bbox[_,2] > bbox[_,0]) and (bbox[_,3] > bbox[_,1]) for _ in range(len(bbox))]
    if np.sum(valid) != len(bbox):
        print ('Invalid bbox annotation detected ...')
        print ('Removing !')
    ants[ind]['ann']['bboxes'] = bbox[valid]
    ants[ind]['ann']['labels'] = a['ann']['labels'][valid]
    assert len(ants[ind]['ann']['bboxes']) == len(ants[ind]['ann']['labels'])

with open(PICKLEPATH, 'wb') as f:
    pickle.dump(ants, f)