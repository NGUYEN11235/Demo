import numpy as np
import torch


def eval_acc(preds, gts):
  n_total = 0
  n_correct = 0
  for gt, pred in zip(gts, preds):
    n_total += len(gt)
    for i in range(len(gt)):
      if pred[i] == gt[i]:
        n_correct += 1
  return (n_correct/n_total)*100

def get_n_samples(p_label, p_start, p_end, g_label, g_start, g_end, iou_threshold, bg_class=["background"]):
      """
      Args:
          p_label, p_start, p_end: return values of get_segments(pred)
          g_label, g_start, g_end: return values of get_segments(gt)
          threshold: threshold (0.1, 0.25, 0.5)
          bg_class: background class
      Return:
          tp: true positive
          fp: false positve
          fn: false negative
      """

      tp = 0
      fp = 0
      hits = np.zeros(len(g_label))

      for j in range(len(p_label)):
          intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
          union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
          IoU = (1.0 * intersection / union) * (
              [p_label[j] == g_label[x] for x in range(len(g_label))]
          )
          # Get the best scoring segment
          idx = np.array(IoU).argmax()

          if IoU[idx] >= iou_threshold and not hits[idx]:
              tp += 1
              hits[idx] = 1
          else:
              fp += 1

      fn = len(g_label) - sum(hits)

      return float(tp), float(fp), float(fn)

def get_segments(frame_wise_label, id2class_map, bg_class: str = "background"):
    """
    Args:
        frame-wise label: frame-wise prediction or ground truth. 1D numpy array
    Return:
        segment-label array: list (excluding background class)
        start index list
        end index list
    """

    labels = []
    starts = []
    ends = []

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))
    ]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label:
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class:
                ends.append(i)

            # update last label
            last_label = frame_wise_label[i]

    if last_label != bg_class:
        ends.append(i)
    return labels, starts, ends

def levenshtein(pred, gt, norm=True):
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """

    n, m = len(pred), len(gt)


    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]
    return score

def eval_f1_score(preds, gts, iou_thresholds=[0.25, 0.5, 0.75]):

  tp_list = [0 for _ in range(len(iou_thresholds))]  # true positive
  fp_list = [0 for _ in range(len(iou_thresholds))]  # false positive
  fn_list = [0 for _ in range(len(iou_thresholds))]  # false negative
  id2class_map = {0:'background', 1:'Back Swing', 2:'Down Swing', 3:'Follow Through'}
  for gt, pred in zip(gts, preds):
      p_label, p_start, p_end = get_segments(pred, id2class_map)
      g_label, g_start, g_end = get_segments(gt, id2class_map)
      for i, th in enumerate(iou_thresholds):
          tp, fp, fn = get_n_samples(
              p_label, p_start, p_end, g_label, g_start, g_end, th
          )
          tp_list[i] += tp
          fp_list[i] += fp
          fn_list[i] += fn
  f1s = []
  for i in range(len(iou_thresholds)):
      precision = tp_list[i] / float(tp_list[i] + fp_list[i])
      recall = tp_list[i] / float(tp_list[i] + fn_list[i])

      f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
      f1 = np.nan_to_num(f1) * 100

      f1s.append(f1)
  return f1s

def eval_edit(preds, gts):
  n_video = 0
  edit_score = 0
  id2class_map = {0:'background', 1:'Back Swing', 2:'Down Swing', 3:'Follow Through'}
  for gt, pred in zip(gts, preds):
      p_label, p_start, p_end = get_segments(pred, id2class_map)
      g_label, g_start, g_end = get_segments(gt, id2class_map)
      n_video += 1
      edit_score += levenshtein(p_label, g_label, norm=True)
  return edit_score/n_video
