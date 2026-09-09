import numpy as np
import torch
from skimage import measure


class IRSTDMetrics:
    """Metrics aligned with the BasicIRSTD evaluation implementation."""

    def __init__(self, threshold=0.5, ignore_empty_for_niou=False):
        self.threshold = float(threshold)
        self.ignore_empty_for_niou = bool(ignore_empty_for_niou)
        self.reset()

    def reset(self):
        self.total_intersection = 0.0
        self.total_union = 0.0
        self.sample_iou_sum = 0.0
        self.sample_iou_count = 0
        self.false_alarm_pixels = 0.0
        self.all_pixels = 0.0
        self.detected_targets = 0
        self.total_targets = 0
        self.f1_tp = 0.0
        self.f1_pos = 0.0
        self.f1_class_pos = 0.0

    def update(self, probability, target, sizes=None):
        prediction = (probability > self.threshold).float().detach().cpu().numpy().astype("int64")
        target = target.detach().cpu().numpy().astype("float32")
        for index in range(prediction.shape[0]):
            height, width = prediction.shape[-2:]
            if sizes is not None:
                height, width = [int(value) for value in sizes[index]]
            pred = prediction[index, 0, :height, :width]
            gt = target[index, 0, :height, :width]
            gt_binary = (gt > 0).astype("int64")

            intersection = float((pred * ((pred == gt_binary).astype("int64"))).sum())
            pred_area = float(pred.sum())
            gt_area = float(gt_binary.sum())
            union = pred_area + gt_area - intersection
            self.total_intersection += intersection
            self.total_union += union
            if not self.ignore_empty_for_niou or gt_area > 0:
                self.sample_iou_sum += intersection / (union + np.spacing(1))
                self.sample_iou_count += 1

            # Keep the raw normalized labels here to match the reference
            # ROCMetric05 behavior for antialiased mask pixels.
            self._update_pixel_f1(pred, gt)
            self._update_object_metrics(pred, gt_binary, height * width)

    def _update_pixel_f1(self, prediction, target):
        # ROCMetric05 in BasicIRSTD reports F1 at its 0.5 bin and uses 0.001
        # smoothing in recall and precision.
        tp = float((prediction * ((prediction == target).astype("int64"))).sum())
        fp = float((prediction * ((prediction != target).astype("int64"))).sum())
        fn = float((((prediction != target).astype("int64")) * (1 - prediction)).sum())
        self.f1_tp += tp
        self.f1_pos += tp + fn
        self.f1_class_pos += tp + fp

    def _update_object_metrics(self, prediction, target, image_pixels):
        predicted_regions = list(measure.regionprops(measure.label(prediction, connectivity=2)))
        target_regions = list(measure.regionprops(measure.label(target, connectivity=2)))
        self.total_targets += len(target_regions)
        distance_match = []

        for target_region in target_regions:
            target_center = np.array(list(target_region.centroid))
            for index, predicted_region in enumerate(predicted_regions):
                predicted_center = np.array(list(predicted_region.centroid))
                distance = np.linalg.norm(predicted_center - target_center)
                if distance < 3:
                    distance_match.append(distance)
                    del predicted_regions[index]
                    break

        self.false_alarm_pixels += float(sum(region.area for region in predicted_regions))
        self.all_pixels += image_pixels
        self.detected_targets += len(distance_match)

    def compute(self):
        miou = self.total_intersection / (np.spacing(1) + self.total_union)
        niou = self.sample_iou_sum / self.sample_iou_count if self.sample_iou_count else 0.0
        pd = self.detected_targets / (self.total_targets + np.spacing(1))
        fa = self.false_alarm_pixels / self.all_pixels if self.all_pixels else 0.0
        recall = self.f1_tp / (self.f1_pos + 0.001)
        precision = self.f1_tp / (self.f1_class_pos + 0.001)
        f1 = (2.0 * recall * precision) / (recall + precision + 0.00001)
        return {"mIoU": miou, "nIoU": niou, "Pd": pd, "Fa": fa, "F1": f1}
