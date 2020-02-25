from sklearn.metrics import f1_score, roc_auc_score

import numpy as np

# At least one dict key should match name of function

def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou[0]


def _two_by_two(y_true, _y_pred, class_threshold, iou_threshold):
    '''
    Calculates TP/FP/TN/FN for 1 class for 1 image 
    @ 1 classification score threshold and 1 IoU threshold.

    y_pred : numpy array of shape (num_pred, 5)
             y_pred[:,-1] = predicted classification scores
    y_true : numpy array of shape (num_true, 4)

    '''
    y_pred = _y_pred.copy()
    y_pred = y_pred[y_pred[:,-1] >= class_threshold, :-1]
    assert y_pred.shape[-1] == 4
    # For negative images ...
    if y_true.shape[0] == 0:
        # If no boxes in truth/predictions, return TP=1, FP=0, FN=0
        # For MAP calculation of 1 (true negative)
        if y_pred.shape[0] == 0: 
            return 1, 0, 0
        else:
            return 0
    else:
        # If no predictions, then ...
        # TP=0, FP=0, FN=num_true
        if y_pred.shape[0] == 0:
            return 0, 0, len(y_true)
    # For positive images ...
    tp = 0 ; fp = 0 ; fn = 0
    # Loop through ground truth boxes
    for ind, gt in enumerate(y_true):
        # If there are no predictions remaining,
        # assign FNs if gt boxes remain
        if len(y_pred) == 0: 
            fn += len(y_true) - ind
            break
        # Compute IoU of each ground truth box against all predictions
        # First, expand gt box 
        repeat_gt = np.repeat(np.expand_dims(gt, axis=0), len(y_pred), axis=0)
        iou_vals = iou(repeat_gt, y_pred)
        # Threshold based on IoU 
        accepted = iou_vals >= iou_threshold
        # If none meet IoU threshold, gt box is FN
        if np.sum(accepted) == 0:
            fn += 1
        else:
            tp += 1
            # Eliminate box with highest IoU from further consideration
            y_pred = np.delete(y_pred, np.argmax(iou_vals), 0)
    # Unmatched predictions are FPs
    fp += len(y_pred)
    return tp, fp, fn


def class_map(y_true, y_pred, class_threshold, iou_threshold):
    map_list = []
    for img in range(len(y_pred)):
        result = _two_by_two(y_true[img], y_pred[img], class_threshold, iou_threshold)
        if type(result) != tuple:
            map_list.append(result)
        else:
            tp, fp, fn = result
            map_list.append(float(tp)/(tp+fp+fn))
    return map_list


# MMDetection output format:
# y_pred = <mmdetection_output>
# len(y_pred) = # of samples
# len(y_pred[i]) = # of classes
#   Each i corresponds to class indices 1, 2, ..., num_classes
#   Remember, 0 is background class
# len(y_pred[i][c]) = 5
#   Predictions for cth class of ith sample
#   [x1, y1, x2, y2, p]

def overall_map(y_true, y_pred, class_thresholds, iou_thresholds):
    num_classes = len(y_pred[0])
    # Each element in this list will contain a dictionary corresponding
    # to each class. Each dictionary will have key:value pairs, 
    # where key is the threshold and value is the corresponding mAP.
    per_class_results_list = []
    for each_class in range(num_classes):
        threshold_results_dict = {}
        for class_thresh in class_thresholds:
            # Extract predictions and labels for that class
            class_preds = [pred[each_class] for pred in y_pred]
            # Add 1 to each_class because 0 represents background
            # for class annotations label format
            gt_labels = [gt['bboxes'][gt['labels'] == each_class+1] for gt in y_true]
            # I can take the mean over the lists
            # Because the negative image results will only be affected
            # by class threshold, not IoU threshold
            iou_results_list = []
            for iou_thresh in iou_thresholds:
                iou_results_list.append(class_map(gt_labels, class_preds, class_thresh, iou_thresh))
            iou_results_array = np.asarray(iou_results_list)
            averaged_over_iou = np.mean(iou_results_array, axis=0)
            averaged_over_img = np.mean(averaged_over_iou[averaged_over_iou != -1])
            threshold_results_dict[class_thresh] = averaged_over_img
        per_class_results_list.append(threshold_results_dict)
    # Now I need to find the max mAP/threshold pair for each class
    results_dict = {}
    overall_map = 0.
    for each_class in range(num_classes):
        threshold_results_dict = per_class_results_list[each_class]
        maps_for_class = [threshold_results_dict[ct] for ct in class_thresholds]
        results_dict['class{}_map'.format(each_class+1)] = np.max(maps_for_class)
        results_dict['class{}_thr'.format(each_class+1)] = class_thresholds[maps_for_class.index(np.max(maps_for_class))]
        overall_map += results_dict['class{}_map'.format(each_class+1)]
    results_dict['overall_map'] = overall_map / num_classes
    return results_dict

###

def _two_by_two(t, _p, iou_threshold):
    assert t.shape[-1] == _p.shape[-1] == 4
    p = _p.copy()
    # If using this function, assumes len(t) > 0 and len(p) > 0
    tp = 0 ; fp = 0 ; fn = 0
    for ind, gt in enumerate(t):
        # Compute IoU of each ground truth box against all predictions
        # First, expand ground truth 
        # If predictions run out, then add to false negatives
        if len(p) == 0: 
            fn += len(t) - ind
            break
        gtr = np.repeat(np.expand_dims(gt, axis=0), len(p), axis=0)
        scores = iou(gtr, p)   
        # Threshold based on IoU 
        match = scores >= iou_threshold
        # If none meet IoU threshold, gt box is FN
        if np.sum(match) == 0:
            fn += 1
        else:
            tp += 1
            # Eliminate box with highest IoU from further consideration
            p = np.delete(p, np.argmax(scores), 0)
    # Unmatched predictions are FPs
    fp += len(p)
    return tp / (tp + fp + fn)


def overall_map(y_true, y_pred, class_thresholds, iou_thresholds):
    num_classes = len(y_pred[0])
    # Collect results per sample
    class_results = {
        i: {
            'y_true': [],
            'y_pred': []
        }
        for i in range(1, num_classes+1)
    }
    for prediction, ground_truth in zip(y_pred, y_true):
        for each_class in range(1, num_classes+1):
            class_pred = prediction[each_class-1]
            class_true = ground_truth['bboxes'][ground_truth['labels'] == each_class]
            class_results[each_class]['y_pred'].append(class_pred)
            class_results[each_class]['y_true'].append(class_true)
    # Loop through classes
    class_maps = {

        c : {
            _ : [] for _ in class_thresholds
        }

        for c in range(1, num_classes + 1)

    }
    # Keep track of positives only
    pos_class_maps = {

        c : {
            _ : [] for _ in class_thresholds
        }

        for c in range(1, num_classes + 1)

    }
    for each_class in range(1, num_classes+1):
        results = class_results[each_class]
        p = results['y_pred']
        t = results['y_true']
        # Loop through class thresholds
        for class_thr in class_thresholds:
            # Loop through images
            positives = []
            for ind, (indiv_p, indiv_t) in enumerate(zip(p, t)):
                indiv_p = indiv_p[indiv_p[:, -1] >= class_thr, :-1]
                if len(indiv_t > 0): 
                    # Keep track of positive cases
                    positives.append(ind)
                if len(indiv_p) + len(indiv_t) == 0:
                    # True negative, ignore
                    img_map = np.nan
                elif len(indiv_p) > 0 and len(indiv_t) == 0:
                    # False positive, MAP=0
                    img_map = 0 
                elif len(indiv_p) == 0 and len(indiv_t) > 0: 
                    # False negative, MAP=0
                    img_map = 0
                else:
                    # Find true positive and false negative boxes
                    # Based on IoU
                    img_map = np.mean([_two_by_two(indiv_t, indiv_p, iou_t) for iou_t in iou_thresholds])
                class_maps[each_class][class_thr].append(img_map)
            # Average scores across all images 
            pos_class_maps[each_class][class_thr] = np.nanmean(
                np.asarray(class_maps[each_class][class_thr])[positives]
            )
            class_maps[each_class][class_thr] = np.nanmean(class_maps[each_class][class_thr])
     
    # Find best class threshold for class c
    def find_best(d, c):
        thresholds, scores = [], []
        for k,v in d[c].items(): 
            thresholds.append(k) ; scores.append(v)
        return thresholds[scores.index(np.max(scores))], np.max(scores)
    best_scores_and_thresholds = {
        c : find_best(class_maps, c) for c in class_maps.keys()
    }
    pos_best_scores_and_thresholds = {
        c : find_best(pos_class_maps, c) for c in pos_class_maps.keys()
    }
    if num_classes == 1:
        c = 1
        final_results_dict = {
            'overall_map': best_scores_and_thresholds[c][1],
            'overall_thr': best_scores_and_thresholds[c][0],
            'pos_overall_map': pos_best_scores_and_thresholds[c][1],
            'pos_overall_thr': pos_best_scores_and_thresholds[c][0]
        }
    elif num_classes > 1:
        final_results_dict = {}
        for c in range(1, num_classes+1):

            final_results_dict['class{}_map'.format(c)] = best_scores_and_thresholds[c][1]
            final_results_dict['class{}_thr'.format(c)] = best_scores_and_thresholds[c][0]
            final_results_dict['pos_class{}_map'.format(c)] = pos_best_scores_and_thresholds[c][1]
            final_results_dict['pos_class{}_thr'.format(c)] = pos_best_scores_and_thresholds[c][0]

        final_results_dict['overall_map'] = np.mean(['class{}_map'.format(c) for c in range(1, num_classes+1)])
    return final_results_dict


def overall_map_75(y_true, y_pred, class_thresholds, **kwargs):
    _results_dict = overall_map(y_true, y_pred, class_thresholds, [0.75])
    results_dict = {}
    for k,v in _results_dict.items():
        results_dict[k+'_75'] = v
    return results_dict


def overall_map_40(y_true, y_pred, class_thresholds, **kwargs):
    _results_dict = overall_map(y_true, y_pred, class_thresholds, [0.40])
    results_dict = {}
    for k,v in _results_dict.items():
        results_dict[k+'_40'] = v
    return results_dict


def overall_auc(y_true, y_pred, **kwargs):
    num_classes = len(y_pred[0])
    per_class_results_list = []
    for each_class in range(num_classes):
        class_preds = [pred[each_class] for pred in y_pred]
        class_probs = [np.max(pred[...,-1]) if len(pred) > 0 else 0 for pred in class_preds]
        gt_labels = [1 if gt['bboxes'][gt['labels'] == each_class+1].shape[0] > 0 else 0 for gt in y_true]
        per_class_results_list.append(roc_auc_score(gt_labels, class_probs))
    results_dict = {}
    overall_auc = 0.
    for each_class in range(num_classes):
        results_dict['class{}_auc'.format(each_class+1)] = per_class_results_list[each_class]
        overall_auc += results_dict['class{}_auc'.format(each_class+1)]
    results_dict['overall_auc'] = overall_auc / num_classes
    return results_dict


