from IOU import iou
import itertools


def nms(scale_boxes, thresh=0.5):
    #print("NMS", scale_boxes[0].shape, scale_boxes[1].shape, scale_boxes[2].shape)

    preds = []
    for batch in scale_boxes:
        for boxes in batch:
            # boxes = [[x1, x2, y1, y2, conf, t1, t2, t3, t4, t5, t6, t7][x1, x2, y1, y2, conf, t1, t2, t3, t4, t5, t6, t7]...]

            # Delete all with small confidence
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


if __name__ == '__main__':
    boxes = [[0, 0, 30, 30, 0.7, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 35, 35, 0.8, 1, 0, 0, 0, 0, 0, 0],
             [80, 0, 89, 10, 0.3, 1, 0, 0, 0, 0, 0, 0],
             [80, 0, 90, 10, 0.2, 1, 0, 0, 0, 0, 0, 0],
             [40, 40, 51, 50, 0.5, 0, 0, 0, 0, 1, 0, 0],
             [40, 40, 50, 50, 0.2, 0, 0, 0, 0, 1, 0, 0]]
    print(nms(boxes, 0.5, 'corners'))


