import numpy as np
from IOU import iou


def mAP(all_true_boxes, all_pred_boxes, classes, thresholds=0.5):
    """
    :param all_true_boxes: Všechny GT bounding boxy
    :param all_pred_boxes: Všechny predikované bounding boxy
    :param classes: Klasifikační třídy
    :param thresholds: Prahy pro rozlišení TP od FP
    :return: hodnota mAP

    Funkce, která z poskytnutých GT a predikcí spočítá mAP (pro různé thresholdy a třídy)
    """
    # https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2

    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    maps = []
    # https://pypi.org/project/mean-average-precision/
    for thresh in thresholds:
        map_ = []
        for i in range(5, 5 + len(classes)):  # Pro každou třídu
            detections = []
            total_true_boxes = 0

            for image in range(len(all_true_boxes)):

                # Pouze boxy konkrétní třídy
                class_true_boxes = [box for box in all_true_boxes[image] if box[i] == max(box[5:]) and box[i] > 0]
                class_pred_boxes = [box for box in all_pred_boxes[image] if box[i] == max(box[5:]) and box[i] > 0]

                if len(class_pred_boxes) == 0 and len(class_true_boxes) == 0:
                    continue

                total_true_boxes += len(class_true_boxes)

                for p_box in class_pred_boxes:
                    found_coresponding_true_box = False
                    for t_box in class_true_boxes:
                        if iou(t_box, p_box, caller="mAP") >= thresh:
                            found_coresponding_true_box = True
                            break
                    if found_coresponding_true_box:
                        detections.append([p_box[5], 1])

                    else:
                        detections.append([p_box[5], 0])

            detections.sort(key=lambda x: x[0], reverse=True)

            tp = []
            for detection in detections:
                if detection[1] == 1:  # True positives
                    tp.append(1)
                else:  # False positives
                    tp.append(0)

            precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)  # TP / TP + FP
            recall = np.cumsum(tp) / total_true_boxes if total_true_boxes != 0 else np.zeros(len(tp))  # TP / TP + FN

            map_.append(np.trapz(np.insert(precision, 0, 1), np.insert(recall, 0, 0)))

        maps.append(sum(map_) / len(map_))

    return sum(maps) / len(maps)
