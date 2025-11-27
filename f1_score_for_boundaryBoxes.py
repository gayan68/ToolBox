import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

"""
Create a matrix where each cell is the IoU between GT[i] and Pred[j].
A predicted box is matched to the ground-truth box with the highest IoU
Each GT can be matched to only one prediction
Good detections count as TP
Extra predictions count as FP
Missed GT count as FN

Pred < GT (7 pred, 10 GT)	7 correct → 7 TP, 0 FP, 3 FN		
Pred > GT (12 pred, 10 GT)	8 correct → 8 TP, 4 FP, 2 FN

supports greedy and Hungarian strategies (Need to submit in `best_match_strategy`)

Args:
    gt_xyxy: List of ground-truth bounding boxes in (x1, y1, x2, y2) format
    pred_xyxy: List of predicted bounding boxes in (x1, y1, x2, y2) format
    iou_thresh: IoU threshold to consider a detection as true positive
    best_match_strategy: "greedy" or "hungarian" for matching strategy
    page_size: (width, height) of the page/image. If None, it will be estimated from the boxes. 

Returns:
    A dictionary with precision, recall, and F1-score.

Author: GAYAN PATHIRAGE
"""

def calculate_iou(mask1, mask2):
    """
    mask1, mask2: binary numpy arrays with shape (H, W), dtype=bool or 0/1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def points_to_mask(points_doc, page_size):
    w, h = page_size
    binary_mask = []
    for points in points_doc:
        blank_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(blank_mask, [pts], color=255)
        binary_mask.append(blank_mask.astype(bool))
    return np.array(binary_mask)


def compute_f1(gt_xyxy, pred_xyxy, iou_thresh=0.5, best_match_strategy="hungarian", page_size=None):

    gt_points = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for (x1, y1, x2, y2) in gt_xyxy]
    pred_points = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for (x1, y1, x2, y2) in pred_xyxy]
    
    if page_size is None:
        # Estimate the image size (The Estimation is sufficient as we need to create masks based on the max x and y)
        all_points = np.vstack((np.array(gt_points).reshape(-1, 2), np.array(pred_points).reshape(-1, 2)))  
        # Get max x and y
        max_x = np.max(all_points[:, 0])
        max_y = np.max(all_points[:, 1])
        page_size = (max_x, max_y)

    # Generate Mask for each BB
    # print(f"gt_points: {len(gt_points)}")
    # print(gt_points)
    # print(f"page_size: {page_size}")
    gt_masks = points_to_mask(gt_points, page_size)
    pred_masks = points_to_mask(pred_points, page_size)

    ious = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            ious[i, j] = calculate_iou(gt_mask, pred_mask)     
    
    matched_gt = set()
    matched_pred = set()

    TP = 0
    
    if best_match_strategy == "greedy":
        # Greedy matching by highest IoU
        while True:
            idx = np.unravel_index(np.argmax(ious), ious.shape)
            max_iou = ious[idx]
            
            if max_iou < iou_thresh:
                break
            
            gt_idx, pred_idx = idx
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                TP += 1
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
            
            ious[gt_idx, :] = -1
            ious[:, pred_idx] = -1
    else:
        # Hungarian algorithm (maximize IoU → minimize 1 - IoU)
        cost_matrix = 1 - ious
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)   

        for i, j in zip(gt_indices, pred_indices):
            if ious[i, j] >= iou_thresh:
                TP += 1
    
    FP = len(pred_masks) - TP
    FN = len(gt_masks) - TP
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"precision": precision, "recall": recall, "f1": f1}

    
