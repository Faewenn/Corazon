from NMS import nms
from TARGET_CONVERSION import targets_to_boxes
import numpy as np
import torch
import YoloV3
import os
from PIL import Image
import random
from matplotlib import patches
from matplotlib import pyplot as plt


def show(image, pred_boxes, classes):
    fig, ax = plt.subplots(1)
    colors = [plt.get_cmap("tab20c")(i) for i in np.linspace(0, 1, len(classes))]

    image = np.moveaxis(np.array(image, dtype=np.uint8), 0, 2)
    image_size = max(image.shape)
    ax.imshow(image)

    for box in pred_boxes:
        class_index = box[5:].index(max(box[5:]))
        x_coord, y_coord = (box[0] - box[2] / 2) * image_size, (box[1] - box[3] / 2) * image_size
        width, height = box[2] * image_size, box[3] * image_size

        rect = patches.Rectangle((x_coord, y_coord), width, height, lw=1, ec=colors[class_index], fc="none")

        ax.add_patch(rect)
        plt.text(x_coord, y_coord, classes[class_index]+" - "+str(box[4]), bbox={'fc': colors[class_index]})

    plt.show()


def train(classes):
    config = [0.001,  # 0 Learning rate
              0.0005,  # 1 Weight decay
              1,  # 2 Batch size
              3,  # 3 Validace každých X epoch
              10,  # 4 Epochs
              320,  # 5 Velikost vstupu (320 pro 320x320, 416 pro 416x416, 608 pro 608x608)
              [0.1],  # 6 mAP thresholds
              0.5,  # 7 NMS Threshold
              0.5,  # 8 Boxes to targets Threshold
              True,  # 9 Pin memory
              0,  # 10 Num workers
              'dataset_light',  # 11 Složka s datasetem
              'dataset_light/images',  # 12 Složka s obrázky
              "center",  # 13 Souřadniceový systém (center / corners)
              False,  # 14 Train?
              True,  # 15 Load checkpoint?
              True,  # 16 Ukázat testovací obrázky?
              True,  # 17 Zamíchat validační a testovací dataset?
              "BC",  # 18 Jméno
              "images"]  # 19 not used
    torch.backends.cudnn.benchmark = True

    # Přidání dalších konfiguračních konstant
    config.append("cuda" if torch.cuda.is_available() else "cpu")
    config.append([[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # 116, 90, 156, 198, 373, 326 / 416
                   [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # 30, 61, 62, 45,  59, 119 / 416
                   [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]])  # 10, 13, 16, 30, 33, 23 / 416

    # Načtení modelu a optimizeru
    model = YoloV3.Architecture(classes).to(config[20])
    print("YoloV3 úspěšně inicializován :-)")

    if config[15]:
        b_number = 999
        for file in os.listdir("checkpoints"):
            if file[-8:] == '.pth.tar' and file[:file.find('_')] == config[18]:
                name, number = file.rsplit("_")
                if float(number[:-8]) < b_number:
                    b_number = float(number[:-8])
                    config[15] = os.path.join("checkpoints", file)

        if isinstance(config[15], str):
            checkpoint = torch.load(config[15])
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\nSuccesfully loaded model")

        else:
            return

        for image_name in os.listdir(config[19]):
            image = Image.open(os.path.join(config[19], image_name)).convert("RGB")

            # Zmenšení a doplnění na čtverec
            scale = config[5] / max(image.size)
            if image.size[0] != image.size[1]:  # Pokud obraz nemá tvar čtverce
                old_size = image.size
                image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                offset = random.randint(0, max(image.size) - min(image.size))
                square_image = Image.fromarray(np.zeros(shape=(max(image.size), max(image.size), 3)), 'RGB')
                if old_size[0] > old_size[1]:  # Pokud je obraz širší, než vyšší
                    square_image.paste(image, (0, offset))
                else:  # Pokud je obraz vyšší, než širší
                    square_image.paste(image, (offset, 0))
                image = np.array(square_image)
            else:  # Pokud je obraz ve tvaru čtverce
                image = np.array(image)

            image = torch.from_numpy(np.moveaxis(image, 2, 0))
            image = image.to(config[20]).float()
            with torch.no_grad():
                out = model(image.unsqueeze(0))
                pred_boxes = targets_to_boxes(out, config[21])
                show(image.to('cpu'), nms(pred_boxes, config[7]), classes)


if __name__ == "__main__":
    classes = ['Gumicka', 'Kolicek', 'Paratko', 'Pripinacek', 'Sirka', 'Sponka', 'Vlajecka']
    train(classes)

