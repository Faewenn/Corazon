from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np


def show_image(image, true_boxes, pred_boxes, classes):
    """
    :param image: obrázek k zobrazení
    :param true_boxes: ground truth bounding boxy
    :param pred_boxes: predikované bounding boxy
    :param classes: posuzované třídy

    Tato funkce má za úkol zobrazit obrázek společně s opravdovými a predikovanými bounding boxy
    """
    fig, ax = plt.subplots(1)
    colors = [plt.get_cmap("tab20c")(i) for i in np.linspace(0, 1, len(classes))]

    image = np.moveaxis(np.array(image, dtype=np.uint8)[0], 0, 2)
    image_size = max(image.shape)
    ax.imshow(image)

    for box in pred_boxes:
        class_index = box[5:].index(max(box[5:]))
        x_coord, y_coord = (box[0] - box[2] / 2) * image_size, (box[1] - box[3] / 2) * image_size
        width, height = box[2] * image_size, box[3] * image_size

        rect = patches.Rectangle((x_coord, y_coord), width, height, lw=1, ec=colors[class_index], fc="none")

        ax.add_patch(rect)
        plt.text(x_coord, y_coord, classes[class_index] + " - " + str(box[4]), bbox={'fc': colors[class_index]})

    if true_boxes is not None:
        for box in true_boxes:
            class_index = box[5:].index(max(box[5:]))
            x_coord, y_coord = (box[0] - box[2] / 2) * image_size, (box[1] - box[3] / 2) * image_size
            width, height = box[2] * image_size, box[3] * image_size

            rect = patches.Rectangle((x_coord, y_coord), width, height, lw=3, ec="white", fc="none")

            ax.add_patch(rect)
            plt.text(x_coord, y_coord, classes[class_index] + " - " + str(box[4]), bbox={'fc': "white"})

    plt.show()
