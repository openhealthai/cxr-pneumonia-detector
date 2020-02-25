import pickle
import numpy as np
import cv2


def draw_bbox(img, rects):
    im = img.copy()
    for r in rects:
        xmin, ymin, xmax, ymax = r
        col = (255, 0, 0)
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), tuple(col), 2)
    return im


PICKLEPATH = '../../../data/wider_face/train_bbox_annotations_with_splits.pkl'
IMAGEDIR = '../../../data/wider_face/'

with open(PICKLEPATH, 'rb') as f:
    ants = pickle.load(f)






for ind, a in enumerate(ants):
    bbox = a['ann']['bboxes']
    valid = (bbox[:,2] > bbox[:,0]) and (bbox[:,3] > bbox[:,1])
    if np.sum(valid) != len(bbox):
        print ('Invalid bbox annotation detected ...')
        print ('Removing !')
    ants[ind]['ann']['bboxes'] = bbox[valid]
    ants[ind]['ann']['labels'] = a['ann']['labels'][valid]
    assert len(ants[ind]['ann']['bboxes']) == len(ants[ind]['ann']['labels'])

with open(PICKLEPATH, 'wb') as f:
    pickle.dump(ants, f)