import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from torchvision.ops import box_iou
import torch

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

    
def compute_f1_gpu(
    gt_xyxy,
    pred_xyxy,
    iou_thresh=0.5,
    best_match_strategy="greedy",  # "greedy" (GPU) or "hungarian" (CPU fallback)
    device="cuda",
):
    """
    gt_xyxy, pred_xyxy: list/np array/torch tensor of shape (N,4) in xyxy format.
    Returns dict with precision/recall/f1 (Python floats).
    """

    # Convert to tensors on GPU
    gt = torch.as_tensor(gt_xyxy, dtype=torch.float32, device=device)
    pr = torch.as_tensor(pred_xyxy, dtype=torch.float32, device=device)

    n_gt = gt.shape[0]
    n_pr = pr.shape[0]

    # Handle empty cases fast
    if n_gt == 0 and n_pr == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if n_pr == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # IoU matrix on GPU: (n_gt, n_pr)
    ious = box_iou(gt, pr)

    if best_match_strategy == "hungarian":
        # SciPy Hungarian is CPU-only; keep IoU compute on GPU, then move matrix to CPU.
        from scipy.optimize import linear_sum_assignment

        cost = (1.0 - ious).detach().cpu().numpy()
        gt_idx, pr_idx = linear_sum_assignment(cost)

        matches = 0
        # ious is GPU tensor; gather matched IoUs efficiently
        matched_ious = ious[torch.as_tensor(gt_idx, device=device), torch.as_tensor(pr_idx, device=device)]
        matches = int((matched_ious >= iou_thresh).sum().item())

        TP = matches

    else:
        # Greedy matching on GPU (one-to-one, pick highest IoU repeatedly)
        TP = 0

        # Masks of "still available" GT and Pred
        gt_available = torch.ones(n_gt, dtype=torch.bool, device=device)
        pr_available = torch.ones(n_pr, dtype=torch.bool, device=device)

        # We’ll iteratively select the best available pair.
        # This is still fast because IoU is precomputed on GPU.
        while True:
            # Mask out unavailable rows/cols by setting them to -1
            masked = ious.clone()
            masked[~gt_available, :] = -1.0
            masked[:, ~pr_available] = -1.0

            max_val = masked.max()
            if max_val < iou_thresh:
                break

            # Get one argmax pair
            flat_idx = masked.argmax()
            gt_i = (flat_idx // n_pr).item()
            pr_j = (flat_idx % n_pr).item()

            # Count match and mark them unavailable
            TP += 1
            gt_available[gt_i] = False
            pr_available[pr_j] = False

    FP = n_pr - TP
    FN = n_gt - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

def compute_bayes_map_50_95_gpu(
    gt_xyxy,
    pred_xyxy,
    bayes=False,  # If True, use the "Bayes" variant of mAP that considers coverage of GT boxes. If False, use standard IoU-based mAP.
    device="cuda",
):
    """
    Computes COCO-style mAP (IoU 0.50:0.05:0.95) for a single class.

    Returns
    -------
    mAP : float
        Mean AP over IoU thresholds [0.50, 0.55, ..., 0.95]
    """

    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    aps = []
    for iou in iou_thresholds:
        ap = compute_bayes_map_gpu(
            gt_xyxy,
            pred_xyxy,
            bayes=bayes,
            iou_thresh=float(iou),
            device=device,
        )
        aps.append(ap)

    return float(np.mean(aps))

def compute_bayes_map_gpu(
    gt_xyxy,
    pred_xyxy,
    bayes=False,  # If True, use the "Bayes" variant of mAP that considers coverage of GT boxes. If False, use standard IoU-based mAP.
    iou_thresh=0.5,
    device="cuda",
):
    """
    Single-class AP (mAP) computation at a single IoU threshold.

    Args
    ----
    gt_xyxy : (N,4) array-like or tensor
        Ground-truth boxes in xyxy format.
    pred_xyxy : (M,4) or (M,5) array-like or tensor
        Predicted boxes. If shape (M,5), last column is treated as score.
        If shape (M,4) no scores are available and input order is used.
    iou_thresh : float
        IoU threshold to count a detection as a true positive.
    device : str or torch.device
        Device to use (e.g., "cuda" or "cpu").

    Returns
    -------
    ap : float
        Average Precision at the given IoU threshold (scalar).
    """

    # convert to tensors on device
    gt = torch.as_tensor(gt_xyxy, dtype=torch.float32, device=device)
    pred = torch.as_tensor(pred_xyxy, dtype=torch.float32, device=device)

    if gt.ndim == 1:
        gt = gt.unsqueeze(0)
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)

    n_gt = 0 if gt.numel() == 0 else gt.shape[0]
    n_pr = 0 if pred.numel() == 0 else pred.shape[0]

    # Fast edge cases
    if n_gt == 0 and n_pr == 0:
        return 0.0
    if n_gt == 0:
        return 0.0
    if n_pr == 0:
        return 0.0

    # Split boxes and scores (if present)
    if pred.shape[1] == 5:
        scores = pred[:, 4]
        boxes_pr = pred[:, :4]
    else:
        # No scores provided: preserve input order by creating descending pseudo-scores
        # (so first pred in list = highest score). This gives a deterministic AP.
        scores = torch.arange(n_pr, 0, -1.0, device=device)
        boxes_pr = pred[:, :4]

    boxes_gt = gt[:, :4]

    # sort predictions by score desc
    scores_sorted, order = torch.sort(scores, descending=True, stable=True)
    boxes_pr = boxes_pr[order]

    # prepare tracking of matched GTs
    gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

    # helper: compute IoU between one box and all GTs (vectorized)
    def iou_one_to_all(box, boxes):
        # box: (4,), boxes: (K,4), all in xyxy
        # compute intersection
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])

        inter_w = (x2 - x1).clamp(min=0.0)
        inter_h = (y2 - y1).clamp(min=0.0)
        inter = inter_w * inter_h

        area_box = (box[2] - box[0]).clamp(min=0.0) * (box[3] - box[1]).clamp(min=0.0)
        area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)

        # IoU = inter / union
        union = area_box + area_boxes - inter
        ious = torch.zeros_like(inter)
        mask = union > 0
        ious[mask] = (inter[mask] / union[mask])

        # "Bayes" = inter / GT area (coverage of GT)
        bayes = torch.zeros_like(inter)
        mask_gt = area_boxes > 0
        bayes[mask_gt] = inter[mask_gt] / area_boxes[mask_gt]

        return ious, bayes

    # Arrays for TP / FP (in score order)
    tp = torch.zeros(n_pr, dtype=torch.float32, device=device)
    fp = torch.zeros(n_pr, dtype=torch.float32, device=device)

    # Iterate preds in score order and assign TP/FP using greedy matching (one GT per pred)
    for i in range(n_pr):
        box = boxes_pr[i]
        if n_gt == 0:
            # no gt to match against
            fp[i] = 1.0
            continue

        ious, bayes = iou_one_to_all(box, boxes_gt)  # (n_gt,)
        best_iou, best_idx = torch.max(ious, dim=0)
        matched_bayes = bayes[best_idx]

        if bayes:
            best_new_score = (best_iou + matched_bayes) / 2.0  # simple average of IoU and Bayes
        else:
            best_new_score = best_iou

        if best_new_score >= iou_thresh and (not gt_matched[best_idx]):
            tp[i] = 1.0
            gt_matched[best_idx] = True
        else:
            fp[i] = 1.0

    # cumulative sums
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)

    # avoid div by zero
    eps = 1e-8
    precision = tp_cum / (tp_cum + fp_cum + eps)
    recall = tp_cum / (n_gt + eps)

    # convert to CPU numpy for stable numeric integration
    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()

    # If no detections at all, AP=0
    if precision.size == 0:
        return 0.0

    # Append sentinel values at ends (0 and 1) per standard AP computation
    # (make precision start with 1 at recall=0 if tp exists — this is standard machinery)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # make precision monotonic (envelope)
    for i in range(mpre.size - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]

    # integrate area under PR curve by summing (recall delta * precision)
    # only at points where recall changes
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for idx in inds:
        ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]

    return float(ap)