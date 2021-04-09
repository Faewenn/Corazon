import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
from TARGET_CONVERSION import boxes_to_targets


class ImageLoader(Dataset):
    def __init__(self, csv_file, classes, config):
        self.annotations = pd.read_csv(csv_file)
        self.classes = {k: v for v, k in enumerate(classes)}
        self.config = config

    def __getitem__(self, index):
        # Načtení obrázku
        img_path = os.path.join(self.config[12], self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")

        # Zmenšení a doplnění na čtverec
        scale = self.config[5] / max(image.size)
        if image.size[0] != image.size[1]:  # Pokud má obraz nemá tvar čtverce
            old_size = image.size
            image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
            offset = random.randint(0, max(image.size) - min(image.size))
            square_image = Image.fromarray(np.zeros(shape=(max(image.size), max(image.size), 3)), 'RGB')

            if old_size[0] > old_size[1]:  # Pokud je obraz širší, než vyšší
                square_image.paste(image, (0, offset))

                # Načtení souřadnic a třídy a převedení na interval < 0 ; 1 >
                box = [self.annotations.iloc[index, 1] * scale / max(image.size),
                       (self.annotations.iloc[index, 2] * scale + offset) / max(image.size),
                       self.annotations.iloc[index, 3] * scale / max(image.size),
                       self.annotations.iloc[index, 4] * scale / max(image.size),
                       self.classes[self.annotations.iloc[index, 5]]]

            else:  # Pokud je obraz vyšší, než širší
                square_image.paste(image, (offset, 0))

                # Načtení souřadnic a třídy a převedení na interval < 0 ; 1 >
                box = [(self.annotations.iloc[index, 1] * scale + offset) / max(image.size),
                       self.annotations.iloc[index, 2] * scale / max(image.size),
                       self.annotations.iloc[index, 3] * scale / max(image.size),
                       self.annotations.iloc[index, 4] * scale / max(image.size),
                       self.classes[self.annotations.iloc[index, 5]]]
            image = np.array(square_image)

        else:  # Pokud je obraz ve tvaru čtverce
            image = np.array(image)

            # Načtení souřadnic a třídy a převedení na interval < 0 ; 1 >
            box = [self.annotations.iloc[index, 1] * scale / max(image.size),
                   self.annotations.iloc[index, 2] * scale / max(image.size),
                   self.annotations.iloc[index, 3] * scale / max(image.size),
                   self.annotations.iloc[index, 4] * scale / max(image.size),
                   self.classes[self.annotations.iloc[index, 5]]]

        # Přeedení rohových souřadnic do středových
        if self.config[13] != "center":
            box[2:4] = abs(box[2] - box[0]), abs(box[3] - box[1])
            box[0:2] = box[0] + box[2] / 2, box[1] + box[3] / 2

        # Vytvoření targetů
        targets = boxes_to_targets([box], self.config, self.classes.keys())

        """
        print("DATASET OUT:")
        print("Image:", np.moveaxis(image, 2, 0).shape, "targets:", targets[0].shape, targets[1].shape, targets[2].shape,
              "name:", self.annotations.iloc[index, 0])
        print("Box:", box)
        """

        return np.moveaxis(image, 2, 0), tuple(targets)

    def __len__(self):
        return len(self.annotations)

