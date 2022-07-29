import torch
import numpy as np
import torch.nn.functional as F



def multiclass_pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def single_class_pixel_accuracy(output, mask):
    with torch.no_grad():
        output = F.sigmoid(output)
        output = (output > 0.5).float()
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def pixel_accuracy(output, mask, n_classes=23):
    if n_classes == 1:
        return single_class_pixel_accuracy(output, mask)
    else:
        return multiclass_pixel_accuracy(output, mask)

def multiclass_mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


EPSILON = 1e-15


def single_classmIoU(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    if n_classes == 1:
        return single_classmIoU(pred_mask, mask)
    else:
        return multiclass_mIoU(pred_mask, mask, smooth, n_classes)
