from IOU import iou
import itertools
import numpy as np


def nms(scale_boxes, thresh=0.5):
    """
    :param scale_boxes: bounding boxy pro non-max suppression
                    ve tvaru [np(Batch, num_boxes, vector), np(Batch, num_boxes, vector), np(Batch, num_boxes, vector)]
    :param thresh: threshold pro odstranění bounding boxů s menším confidence a pro IoU porovnávání
    :return: list bounding boxů
    """
    preds = []
    for batch in range(len(scale_boxes[0])):
        boxes = np.concatenate((scale_boxes[0][batch], scale_boxes[1][batch], scale_boxes[2][batch]))

        # Smazat všechny BB s malým confidence
        boxes = [list(box) for box in boxes if box[4] >= thresh]

        # Seřadit boxy podle confidence
        boxes = sorted(boxes, key=lambda box: box[4], reverse=True)

        if boxes:
            for i in range(5, len(boxes[0])):  # Pro každou třídu
                # Pouze boxy konkrétní třídy
                class_boxes = [box for box in boxes if box[i] == max(box[5:]) and box[i] != 0]

                while class_boxes:  # Dokud jsou nevyřízené bounding boxy
                    prediction = class_boxes.pop(0)  # Největší confidence BB označit
                    preds.append(prediction)  # Přidat jej do výsledných BB
                    if not class_boxes:  # Pokud už nezbývá žádný BB, skonči
                        break

                    # Jinak z class boxů odebrat všechny, které mají s označeným boxem iou >= threshold
                    class_boxes = []
                    for box in class_boxes:
                        if iou(prediction, box, caller="NMS") <= thresh:
                            class_boxes.append(box)

    return [k for k, _ in itertools.groupby(preds)]


