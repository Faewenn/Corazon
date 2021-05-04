import torch


def iou(box1, box2, caller=None):
    """
    :param box1: bounding box ve tvaru [X, Y, Šířka, Výška...]
    :param box2: bounding box ve tvaru [X, Y, Šířka, Výška...]
    :param caller:
    :return: hodnotu IoU
    """
    if caller != "loss funkce":
        #print("IOU:", caller, "boxes:", box1, box2)

        for box in (box1, box2):  # Převední z centrovaných souřadnic do rohových souřadnic
            box[0], box[2] = box[0] - box[2] / 2, box[0] + box[2] / 2
            box[1], box[3] = box[1] - box[3] / 2, box[1] + box[3] / 2

        # Vytvoření průnikového boxu
        ibox = (max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3]))

        # Pokud se boxy nepřekrývají, return 0
        if ibox[2] - ibox[0] < 0 or ibox[3] - ibox[1] < 0:
            return 0

        # Výpočet průniku a sjednocení
        inter = (ibox[2] - ibox[0]) * (ibox[3] - ibox[1])
        union = abs((box1[2] - box1[0]) * (box1[3] - box1[1])) + abs((box2[2] - box2[0]) * (box2[3] - box2[1])) - inter

        return inter / union

    box1_x1, box1_y1 = box1[..., 0:1] - box1[..., 2:3] / 2, box1[..., 1:2] - box1[..., 3:4] / 2
    box1_x2, box1_y2 = box1[..., 0:1] + box1[..., 2:3] / 2, box1[..., 1:2] + box1[..., 3:4] / 2
    box2_x1, box2_y1 = box2[..., 0:1] - box2[..., 2:3] / 2, box2[..., 1:2] - box2[..., 3:4] / 2
    box2_x2, box2_y2 = box2[..., 0:1] + box2[..., 2:3] / 2, box2[..., 1:2] + box2[..., 3:4] / 2

    inter = torch.clamp(torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1), 0) * \
            torch.clamp(torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1), 0)
    union = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1)) + abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1)) - inter

    return inter / union

