"""This file calculates metrics including:
    1. adapted_rand_error
    2. variation_of_information
    3. Dice2
    4. pq
"""
from skimage.metrics import adapted_rand_error
from skimage.metrics import variation_of_information
import argparse
import numpy as np
import os
from glob import glob
from PIL import Image
from skimage.measure import label
import scipy
from scipy.optimize import linear_sum_assignment
from tifffile import imread
import scipy.ndimage as nd
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, binary_opening, disk


def create_border_mask(image, max_dist=None, background_label=0):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.

    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)
    target = image.copy()

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(image == padded[:-2, 1:-1], image == padded[2:, 1:-1]),
        np.logical_and(image == padded[1:-1, :-2], image == padded[1:-1, 2:])
    )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
    )

    border = distances <= max_dist
    target[border] = background_label

    return target


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    try:  # blinly remove background
        pred_id.remove(0)
    except ValueError:
        pass
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    # return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
    return dq * sq


#####
# def get_fast_dice_2(true, pred):
#     """Ensemble dice."""
#     true = np.copy(true)
#     pred = np.copy(pred)
#     true_id = list(np.unique(true))
#     pred_id = list(np.unique(pred))

#     overall_total = 0
#     overall_inter = 0

#     true_masks = [np.zeros(true.shape)]
#     for t in true_id[1:]:
#         t_mask = np.array(true == t, np.uint8)
#         true_masks.append(t_mask)

#     pred_masks = [np.zeros(true.shape)]
#     for p in pred_id[1:]:
#         p_mask = np.array(pred == p, np.uint8)
#         pred_masks.append(p_mask)

#     for true_idx in range(1, len(true_id)):
#         t_mask = true_masks[true_idx]
#         pred_true_overlap = pred[t_mask > 0]
#         pred_true_overlap_id = np.unique(pred_true_overlap)
#         pred_true_overlap_id = list(pred_true_overlap_id)
#         try:  # blinly remove background
#             pred_true_overlap_id.remove(0)
#         except ValueError:
#             pass  # just mean no background
#         for pred_idx in pred_true_overlap_id:
#             p_mask = pred_masks[pred_idx]
#             total = (t_mask + p_mask).sum()
#             inter = (t_mask * p_mask).sum()
#             overall_total += total
#             overall_inter += inter

#     return 2 * overall_inter / overall_total

def get_fast_dice(true, pred):
    return np.sum(pred[true]) * 2.0 / (np.sum(pred) + np.sum(true))


def eval_metric_instance(pred_tif, gt_tif):
    
    arand_error = []
    voi_merge = []
    voi_split = []
    fg_dice = []
    img_name = []

    gts = imread(gt_tif)
    for (i, img_tst, img_gt) in zip(range(gts.shape[0]), imread(pred_tif), gts):
        if i < 400:
            continue
        assert img_gt.shape == img_tst.shape

        # test_labels = label(img_tst, background=0)
        # gt_labels = label(img_gt, background=0)
        footprint = disk(6)

        edge_tst = img_tst == 0
        edge_tst = binary_closing(binary_opening(edge_tst, footprint=footprint), footprint=footprint)
        img_tst[edge_tst == 1] = 0
        seeds, _ = nd.label( nd.binary_erosion( img_tst == 1, iterations=5 ) )
        test_labels = watershed(-img_tst, seeds)

        edge_gt = img_gt == 0
        edge_gt = binary_closing(binary_opening(edge_gt, footprint=footprint), footprint=footprint)
        img_gt[edge_gt == 1] = 0
        seeds, _ = nd.label( nd.binary_erosion( img_gt == 1, iterations=5 ) )
        gt_labels = watershed(-img_gt, seeds)

        test_labels = remap_label(test_labels)
        gt_labels = remap_label(gt_labels)

        base = 2*np.log(max(len(np.unique(gt_labels)), len(np.unique(test_labels))))
        arand_error.append(adapted_rand_error(image_true=gt_labels, image_test=test_labels, ignore_labels=(0,))[0])
        under, over = variation_of_information(gt_labels, test_labels, ignore_labels=(0,))
        voi_merge.append(over / base)
        voi_split.append(under / base)
        fg_dice.append(get_fast_dice(test_labels > 0, gt_labels > 0))
        # pq_score.append(get_fast_pq(test_labels, gt_labels))

    arand_error = np.array(arand_error)
    voi_merge = np.array(voi_merge)
    voi_split = np.array(voi_split)
    fg_dice = np.array(fg_dice)

    return np.mean(fg_dice), np.mean(arand_error), np.mean(voi_merge), np.mean(voi_split)


def eval_metric_instance_reverse(pred_tif, gt_tif):
    
    arand_error = []
    voi_merge = []
    voi_split = []
    fg_dice = []
    img_name = []

    gts = imread(gt_tif)
    for (i, img_tst, img_gt) in zip(range(gts.shape[0]), imread(pred_tif), gts):
        if i < 75:
            continue
        assert img_gt.shape == img_tst.shape

        img_gt = np.array(np.logical_not(img_gt), dtype=np.float32)
        img_tst = np.array(np.logical_not(img_tst), dtype=np.float32)

        # test_labels = label(img_tst, background=0)
        # gt_labels = label(img_gt, background=0)
        footprint = disk(6)

        edge_tst = img_tst == 0
        edge_tst = binary_closing(binary_opening(edge_tst, footprint=footprint), footprint=footprint)
        img_tst[edge_tst == 1] = 0
        seeds, _ = nd.label( nd.binary_erosion( img_tst == 1, iterations=5 ) )
        test_labels = watershed(-img_tst, seeds)

        edge_gt = img_gt == 0
        edge_gt = binary_closing(binary_opening(edge_gt, footprint=footprint), footprint=footprint)
        img_gt[edge_gt == 1] = 0
        seeds, _ = nd.label( nd.binary_erosion( img_gt == 1, iterations=5 ) )
        gt_labels = watershed(-img_gt, seeds)

        test_labels = remap_label(test_labels)
        gt_labels = remap_label(gt_labels)

        base = 2*np.log(max(len(np.unique(gt_labels)), len(np.unique(test_labels))))
        arand_error.append(adapted_rand_error(image_true=gt_labels, image_test=test_labels, ignore_labels=(0,))[0])
        under, over = variation_of_information(gt_labels, test_labels, ignore_labels=(0,))
        voi_merge.append(over / base)
        voi_split.append(under / base)
        fg_dice.append(get_fast_dice(test_labels > 0, gt_labels > 0))
        # pq_score.append(get_fast_pq(test_labels, gt_labels))

    arand_error = np.array(arand_error)
    voi_merge = np.array(voi_merge)
    voi_split = np.array(voi_split)
    fg_dice = np.array(fg_dice)

    return np.mean(fg_dice), np.mean(arand_error), np.mean(voi_merge), np.mean(voi_split)
