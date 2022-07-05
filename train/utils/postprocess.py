import numpy as np

def label_2_class(img_labels, colors ):
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""

    assert len(img_labels.shape) == 2, img_labels.shape
    classes = []
    masks = []
    for label_id in range(1, len(colors)):
        print(label_id)
        mask = img_labels == label_id
        print(mask.shape)
        print(mask[0][0])
        if mask[0][0] == True:
            mask = mask.tolist()
            masks.append(mask)
            classes.append(label_id)
    m = np.array(masks)
    return m, classes

import cv2

def get_segmentation_annotations(segmentation_mask, colors):
    hw = segmentation_mask.shape[:2]
    segmentation_mask = segmentation_mask.reshape(hw)
    polygons = []

    for segtype in range(len(colors)):
        if segtype == 0:
            continue
        temp_img = np.zeros(hw)
        seg_class_mask_over_seg_img = np.where(segmentation_mask==segtype)
        if np.any(seg_class_mask_over_seg_img):
            temp_img[seg_class_mask_over_seg_img] = 1
            contours, _ = cv2.findContours(temp_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            print('contours', contours)
            if len(contours) < 1:
                continue
            for contour in contours:
                if cv2.contourArea(contour) < 2:
                    continue
                polygons.append((contour, segtype))
    return polygons


def get_segmentation_dict(segmentation_mask, colors, img_id="0", starting_annotation_indx=0, DEBUG=False):
    annotations = []
    for indx, (contour, seg_type) in enumerate(get_segmentation_annotations(segmentation_mask, colors)):
        segmentation = contour.ravel().tolist()
        annotations.append({
            "segmentation": segmentation,
            "area": cv2.contourArea(contour),
            "image_id": img_id,
            "category_id": seg_type,
            "id": starting_annotation_indx + indx
        })
    return annotations