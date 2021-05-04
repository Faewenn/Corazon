import torch
import torch.nn as nn
from IOU import iou


class Architecture(nn.Module):
    def __init__(self, classes):     # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        super().__init__()
        self.vector_size = len(classes) + 5
        re_lu = 0.1

        # Head
        layers = nn.ModuleList()

        # ===== DARKNET53 =====

        # Conv layer 1 {25}
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 2 {35}
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(re_lu))

        # Residual block 1 (1x) {43, 51}
        layers.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(re_lu))
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 3 {65}
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(re_lu))

        # Residual block 2 (2x) {73, 81, 93, 101}
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(128))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 4 {115}
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(re_lu))

        # Residual block 3 (8x) {123, 131, 143, 151, 163, 171, 183, 191, 204, 212, 224, 232, 244, 252, 264, 272}
        for _ in range(8):
            layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(128))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 5 {286}
        layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(re_lu))

        # Residual block 4 (8x) {294, 302, 315, 323, 336, 344, 357, 365, 377, 385, 398, 406, 419, 427, 439, 447}
        for _ in range(8):
            layers.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 6 {461}
        layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(re_lu))

        # Residual block 5 (4x) {469, 477, 489, 497, 509, 517, 529, 537}
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(1024))
            layers.append(nn.LeakyReLU(re_lu))

        # ===== END OF DARKNET53 =====

        # Conv layer 7, 8, 9, 10, 11, 12 {551, 559, 567, 575, 583, 591}
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(1024))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 13 (out Scale 1) {599}
        layers.append(nn.Conv2d(in_channels=1024, out_channels=self.vector_size * 3, kernel_size=1, stride=1, padding=0))

        # Conv layer 14 {621}
        layers.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(re_lu))

        # Upsample 1 {629}
        layers.append(nn.Upsample(scale_factor=2))

        # Conv layer 15 {637}
        layers.append(nn.Conv2d(in_channels=256*3, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 16, 17, 18, 19 {645, 653, 661, 669}
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 20 {677}
        layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 21 (out Scale 2) {685}
        layers.append(nn.Conv2d(in_channels=512, out_channels=self.vector_size * 3, kernel_size=1, stride=1, padding=0))

        # Conv layer 22 {708}
        layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(re_lu))

        # Upsample 2 {716}
        layers.append(nn.Upsample(scale_factor=2))

        # Conv layer 23 {724}
        layers.append(nn.Conv2d(in_channels=128*3, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 24, 25, 26, 27 {732, 740, 748, 756}
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.LeakyReLU(re_lu))
            layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(128))
            layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 28 {764}
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(re_lu))

        # Conv layer 29 (out Scale 3) {772}
        layers.append(nn.Conv2d(in_channels=256, out_channels=self.vector_size * 3, kernel_size=1, stride=1, padding=0))

        self.layers = layers

    def forward(self, x):
        outs = []

        # ===== DARKNET53 =====

        # Conv layer 1
        conv, bn, relu = self.layers[0], self.layers[1], self.layers[2]
        x = relu(bn(conv(x)))

        # Conv layer 2
        conv, bn, relu = self.layers[3], self.layers[4], self.layers[5]
        x = relu(bn(conv(x)))

        # Residual block 1 (1x)
        o = 6
        for i in range(1):
            residual = x
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x))) + residual

        # Conv layer 3
        conv, bn, relu = self.layers[12], self.layers[13], self.layers[14]
        x = relu(bn(conv(x)))

        # Residual block 2 (2x)
        o = 15
        for i in range(2):
            residual = x
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x))) + residual
            o += 5

        # Conv layer 4
        conv, bn, relu = self.layers[27], self.layers[28], self.layers[29]
        x = relu(bn(conv(x)))

        # Residual block 3 (8x)
        o = 30
        for i in range(8):
            residual = x
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x))) + residual
            o += 5

        # Route 1
        route1 = x

        # Conv layer 5
        conv, bn, relu = self.layers[78], self.layers[79], self.layers[80]
        x = relu(bn(conv(x)))

        # Residual block 4 (8x)
        o = 81
        for i in range(8):
            residual = x
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x))) + residual
            o += 5

        # Route 2
        route2 = x

        # Conv layer 6
        conv, bn, relu = self.layers[129], self.layers[130], self.layers[131]
        x = relu(bn(conv(x)))

        # Residual block 5 (8x)
        o = 132
        for i in range(4):
            residual = x
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x))) + residual
            o += 5

        # ===== END OF DARKNET53 =====

        # Conv layer 7, 8, 9, 10
        o = 156
        for i in range(2):
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x)))
            o += 5

        # Conv layer 11
        conv, bn, relu = self.layers[168], self.layers[169], self.layers[170]
        x = relu(bn(conv(x)))

        # Conv layer 12
        conv, bn, relu = self.layers[171], self.layers[172], self.layers[173]
        y = relu(bn(conv(x)))

        # Conv layer 13 (out Scale 1)
        y = self.layers[174](y)
        y = y.reshape(y.shape[0], 3, self.vector_size, y.shape[2], y.shape[3])
        outs.append(y.permute(0, 3, 4, 1, 2))

        # Conv layer 14
        conv, bn, relu = self.layers[175], self.layers[176], self.layers[177]
        x = relu(bn(conv(x)))

        # Upsample 1
        x = self.layers[178](x)
        x = torch.cat([x, route2], dim=1)

        # Conv layer 15
        conv, bn, relu = self.layers[179], self.layers[180], self.layers[181]
        x = relu(bn(conv(x)))

        # Conv layer 16, 17, 18, 19
        o = 182
        for i in range(2):
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x)))
            o += 5

        # Conv layer 20
        conv, bn, relu = self.layers[194], self.layers[195], self.layers[196]
        y = relu(bn(conv(x)))

        # Conv layer 21 (out Scale 2)
        y = self.layers[197](y)
        y = y.reshape(y.shape[0], 3, self.vector_size, y.shape[2], y.shape[3])
        outs.append(y.permute(0, 3, 4, 1, 2))

        # Conv layer 22
        conv, bn, relu = self.layers[198], self.layers[199], self.layers[200]
        x = relu(bn(conv(x)))

        # Upsample 2
        x = self.layers[201](x)
        x = torch.cat([x, route1], dim=1)

        # Conv layer 23
        conv, bn, relu = self.layers[202], self.layers[203], self.layers[204]
        x = relu(bn(conv(x)))

        # Conv layer 24, 25, 26, 27
        o = 205
        for i in range(2):
            conv, bn, relu = self.layers[o + i], self.layers[o + i + 1], self.layers[o + i + 2]
            x = relu(bn(conv(x)))
            conv, bn, relu = self.layers[o + i + 3], self.layers[o + i + 4], self.layers[o + i + 5]
            x = relu(bn(conv(x)))
            o += 5

        # Conv layer 28
        conv, bn, relu = self.layers[217], self.layers[218], self.layers[219]
        x = relu(bn(conv(x)))

        # Conv layer 29 (out Scale 3)
        y = self.layers[220](x)
        y = y.reshape(y.shape[0], 3, self.vector_size, y.shape[2], y.shape[3])
        outs.append(y.permute(0, 3, 4, 1, 2))

        # [B, Cx, Cy, A, V]
        # outs => [[B, 10, 10, 3, 12][B, 20, 20, 3, 12][B, 40, 40, 3, 12]]
        return outs


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, target, anchors):
        is_obj_coords = target[..., 4] == 1
        no_obj_coords = target[..., 4] == 0

        # Confidence loss (is object)
        conf_loss = self.mse(predictions[..., 4:5][is_obj_coords], target[..., 4:5][is_obj_coords])

        # Confidence loss (no object)
        no_conf_loss = self.mse(predictions[..., 4:5][no_obj_coords], target[..., 4:5][no_obj_coords])

        # Box loss
        anchors = anchors.reshape(1, 1, 1, 3, 2)
        predictions[..., 0:2] = self.sigmoid(predictions[..., 0:2])
        predictions[..., 2:4] = torch.exp(predictions[..., 2:4]) * anchors
        box_loss = self.mse(predictions[..., 0:4][is_obj_coords], target[..., 0:4][is_obj_coords])

        # Class loss
        class_loss = self.bce((predictions[..., 5:][is_obj_coords]), (target[..., 5:][is_obj_coords]))

        return conf_loss + no_conf_loss + box_loss + class_loss

