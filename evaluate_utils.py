# IOU = Area of Intersection / Area of Union
def get_iou(true_bbox, predicted_bbox):
    x1_intersection = max(true_bbox[0], predicted_bbox[0])
    y1_intersaction = max(true_bbox[1], predicted_bbox[1])
    x2_intersaction = min(true_bbox[2], predicted_bbox[2])
    y2_intersection = min(true_bbox[3], predicted_bbox[3])

    intersection_area = max(0, x2_intersaction - x1_intersection) * \
                        max(0, y2_intersection - y1_intersaction)

    pred_bbox_h = max(0, predicted_bbox[3] - predicted_bbox[1])
    pred_bbox_w = max(0, predicted_bbox[2] - predicted_bbox[0])
    pred_bbox_area = pred_bbox_h * pred_bbox_w

    true_bbox_h = max(0, true_bbox[3] - true_bbox[1])
    true_bbox_w = max(0, true_bbox[2] - true_bbox[0])
    true_bbox_area = true_bbox_h * true_bbox_w

    iou = intersection_area / (pred_bbox_area + true_bbox_area + intersection_area)

    return iou

