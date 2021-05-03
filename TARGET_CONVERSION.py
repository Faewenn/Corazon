import numpy as np
import torch
from IOU import iou


def targets_to_boxes(targets, anchors):
    #print("Targets to boxes:", targets[0].shape, targets[1].shape, targets[2].shape)
    boxes = []

    for batch in targets:
        if isinstance(batch, torch.Tensor):
            batch = np.array(batch.to('cpu'))

        for target in batch:
            if len(target.shape) != 5 and len(target.shape) == 4:
                target = target[np.newaxis, :]

            scale = target.shape[1]

            for b, i, j, a in np.argwhere(target[..., 4]):
                target[b][i][j][a][0] = (target[b][i][j][a][0] + j) / scale
                target[b][i][j][a][1] = (target[b][i][j][a][1] + i) / scale
                target[b][i][j][a][2:4] = target[b][i][j][a][2] / scale, target[b][i][j][a][3] / scale

            boxes.append(target.reshape((target.shape[0], scale * scale * len(anchors), target.shape[4])))

    return boxes


def boxes_to_targets(boxes, config, classes, thresh=0.5):
    targets = [np.zeros((scale, scale, len(config[21]), len(classes) + 5)) for scale in config[22]]

    for box in boxes:
        # Zjištění nejlepšího anchoru
        ious = [iou([0, 0, item[0], item[1]], [0, 0, box[2], box[3]], caller="boxes to targets")
                for anchor in config[21] for item in anchor]
        anchor_indices = sorted(range(len(ious)), key=ious.__getitem__, reverse=True)

        # [S, Cx, Cy, A, V] (zeros([[10, 10, 3, 12][20, 20, 3, 12][40, 40, 3, 12]]))

        has_anchor = [False, False, False]  # each scale should have one anchor
        for anchor_number in anchor_indices:
            scale_index, anchor_index = divmod(anchor_number, len(config[21]))
            scale = config[22][scale_index]
            i, j = int(scale * box[1]), int(scale * box[0])  # Souřadnice buňky
            anchor_taken = targets[scale_index][i, j, anchor_index, 4]

            if not anchor_taken and not has_anchor[scale_index]:
                targets[scale_index][i, j, anchor_index, 4] = 1
                targets[scale_index][i, j, anchor_index, 0:4] = [scale * box[0] - j, scale * box[1] - i, box[2] * scale,
                                                                 box[3] * scale]
                targets[scale_index][i, j, anchor_index, 5 + box[4]] = 1
                has_anchor[scale_index] = True

            elif not anchor_taken and ious[anchor_number] > thresh:
                targets[scale_index][i, j, anchor_index, 4] = -1  # ignore prediction

    return targets

