import os
import pandas as pd
import numpy as np
import cv2 as cv
import random


def rename_resize(source_path, end_path, new_resolution, max_images):
    """
    :param source_path: cesta ve které jsou všechny fotografie
    :param end_path: cesta, kam se mají všechny nové fotografie uložit
    :param new_resolution: nové rozlišení fotografií
    :param max_images: maximální počet fotografií, které mají být přesunuty a změněny

    První funkce ze tří

    Změní se rozměr originálních fotografií, a následně se uloží do formátu "trida_cislo.jpg"
    """
    base_path = os.path.dirname(__file__)
    source_path = os.path.join(base_path, source_path)
    end_path = os.path.join(base_path, end_path)
    for directory in os.listdir(source_path):
        files = os.listdir(os.path.join(source_path, directory))
        random.shuffle(files)
        for i, file in enumerate(files):
            if i == max_images:
                break
            new_name = str(directory.lower()) + "_" + str(i) + ".jpg"
            image = cv.imread(os.path.join(os.path.join(source_path, directory), file))
            image = cv.resize(image, new_resolution)
            cv.imwrite(os.path.join(end_path, new_name), image)


def color_shift(img, low=10, high=35):
    """
    :param img: obrázek k úpravě
    :param low: minimální posunutí
    :param high: maximální posunutí
    :return: upravený obrázek

    Augmentace pomocí posunutí barevného kanálu
    """
    img = img.copy()
    tmp = random.random()
    if 0 < tmp <= 1 / 2:
        img[:, :, 0] += np.random.randint(low, high, dtype=np.uint8)
    if 1 / 3 < tmp <= 5 / 6:
        img[:, :, 1] += np.random.randint(low, high, dtype=np.uint8)
    if 2 / 3 < tmp or tmp <= 1 / 6:
        img[:, :, 2] += np.random.randint(low, high, dtype=np.uint8)
    return img


def brightness(img, low=-35, high=35):
    """
    :param img: Obrázek k úpravě
    :param low: minimální změna jasu
    :param high: maxiální změna jasu
    :return: upravený obrázek

    Augmentace pomocí změny jasu
    """
    img = img.copy()
    img = np.array(cv.cvtColor(img, cv.COLOR_BGR2HSV), dtype=np.float64)
    img[:, :, 2] += np.random.randint(low, high)
    img[:, :, 2][img[:, :, 2] < 0] = 0
    img[:, :, 2][img[:, :, 2] > 255] = 255
    return cv.cvtColor(np.array(img, dtype=np.uint8), cv.COLOR_HSV2BGR)


def salt_pepper(img, strength):
    """
    :param img: obrázek k úpravě
    :param strength: % pixelů které mají být změněny
    :return: upravený obrázek

    Augmentace pomocí salt & pepper
    """
    img = img.copy()
    height, width, channel = img.shape
    mask = np.random.choice((0, 1, 2), size=(height, width, 1), p=[strength, (1 - strength) / 2, (1 - strength) / 2])
    mask = np.repeat(mask, channel, axis=2)
    img[mask == 1] = 0
    img[mask == 2] = 255
    return img


def augment(path):
    """
    :param path: Cesta k fotografiím, které mají být zaugmentovány

    Druhá funkce ze tří

    Zaugmentují se obrázky ve složce.

    Doporučení: Před zavoláním této funkce oanotovat upravené fotografie z první funkce, než se k nim přidají další
    """
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, path)

    for file in os.listdir(path):
        image = cv.imread(os.path.join(path, file))
        name, remainder = str(file).split("_", 1)

        cv.imwrite(os.path.join(path, name + '_hFlip_' + remainder), cv.flip(image, 1))
        cv.imwrite(os.path.join(path, name + '_vFlip_' + remainder), cv.flip(image, 0))
        cv.imwrite(os.path.join(path, name + '_rot180_' + remainder), cv.rotate(image, cv.ROTATE_180))

    for file in os.listdir(path):
        image = cv.imread(os.path.join(path, file))
        name, remainder = str(file).split("_", 1)
        cv.imwrite(os.path.join(path, name + '_colorShift_' + remainder), color_shift(image))
        cv.imwrite(os.path.join(path, name + '_brightness_' + remainder), brightness(image))
        cv.imwrite(os.path.join(path, name + '_salt&pepper_' + remainder), salt_pepper(image, random.random() / 10 + 0.9))
        cv.imwrite(os.path.join(path, name + '_blur_' + remainder),
                   cv.blur(image, (np.random.randint(2, 5), np.random.randint(2, 5))))

def anotate(source_path, end_path):
    """
    :param source_path: Cesta k anotacím každé třídy
    :param end_path: Cílová cesta

    Třetí funkce ze tří

    Tato funkce má za úkol vzít anotace (každé třídy zvlášť) z VIA anotatoru a převést je na jednodušší zápis.
    Tato funkce také rozdělí dataset na trénovací, validační a trénovací vzorky (70 % train, 15% val, 15% test)
    """
    base_path = os.path.dirname(__file__)
    source_path = os.path.join(base_path, source_path)
    end_path = os.path.join(base_path, end_path)
    first_prefixes = ['rot180_', 'hFlip_', 'vFlip_']
    second_prefixes = ['colorShift_', 'brightness_', 'blur_', 'salt&pepper_']
    tmp_dataset = pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'class'])
    dataset = pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'class'])
    traindf = pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'class'])
    valdf = pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'class'])
    testdf = pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'class'])

    for csvfile in os.listdir(source_path):
        df = pd.read_csv(os.path.join(source_path, csvfile))
        X, Y, W, H, C = [], [], [], [], []

        for a, i in enumerate(df['region_shape_attributes']):
            if len(i) == 2:
                X.append(np.NaN)
                Y.append(np.NaN)
                W.append(np.NaN)
                H.append(np.NaN)
                continue

            # Vyhledání souřadnice X
            if i[20] == ',':
                tmp_x_end = 20
            elif i[21] == ',':
                tmp_x_end = 21
            else:
                tmp_x_end = 22

            X.append(int(i[19:tmp_x_end]))

            # Vyhledání souřadnice Y
            if i[tmp_x_end + 6] == ',':
                tmp_y_end = tmp_x_end + 6
            elif i[tmp_x_end + 7] == ',':
                tmp_y_end = tmp_x_end + 7
            else:
                tmp_y_end = tmp_x_end + 8

            Y.append(int(i[tmp_x_end + 5:tmp_y_end]))

            # Vyhledání šířky
            if i[tmp_y_end + 10] == ',':
                tmp_w_end = tmp_y_end + 10
            elif i[tmp_y_end + 11] == ',':
                tmp_w_end = tmp_y_end + 11
            else:
                tmp_w_end = tmp_y_end + 12

            W.append(int(i[tmp_y_end + 9:tmp_w_end]))

            # Vyhledání výšky
            if i[tmp_w_end + 11] == '}':
                tmp_h_end = tmp_w_end + 11
            elif i[tmp_w_end + 12] == '}':
                tmp_h_end = tmp_w_end + 12
            else:
                tmp_h_end = tmp_w_end + 13

            H.append(int(i[tmp_w_end + 10:tmp_h_end]))

            C.append(df.loc[a]['filename'][:df.loc[a]['filename'].find('_')].capitalize())

        df['x'], df['y'], df['width'], df['height'], df['class'] = X, Y, W, H, C

        for column in df.columns:
            if not column in tmp_dataset.columns:
                df.pop(column)

        tmp_dataset = tmp_dataset.append(df, ignore_index=True)

    classes = list(dict.fromkeys(tmp_dataset['class'].tolist()))

    for c in classes:
        c_df = tmp_dataset.loc[tmp_dataset['class'] == c]
        c_df = c_df.reset_index(drop=True)

        tmp = pd.DataFrame(columns=c_df.columns)
        for prefix in first_prefixes:
            tmp_copy = c_df.copy()
            for i, j in enumerate(c_df['filename']):
                tmp_copy['filename'][i] = j[:j.find('_') + 1] + prefix + j[j.find('_') + 1:]
                if prefix == "rot180_":
                    tmp_copy['x'][i] = 599 - c_df.loc[i]['x'] - c_df.loc[i]['width']
                    tmp_copy['y'][i] = 399 - c_df.loc[i]['y'] - c_df.loc[i]['height']
                elif prefix == "hFlip_":
                    tmp_copy['x'][i] = 599 - c_df.loc[i]['x'] - c_df.loc[i]['width']
                elif prefix == "vFlip_":
                    tmp_copy['y'][i] = 399 - c_df.loc[i]['y'] - c_df.loc[i]['height']

            tmp = tmp.append(tmp_copy, ignore_index=True)

        c_df = c_df.append(tmp, ignore_index=True)

        tmp = pd.DataFrame(columns=c_df.columns)
        for prefix in second_prefixes:
            tmp_copy = c_df.copy()
            for i, j in enumerate(c_df['filename']):
                tmp_copy['filename'][i] = j[:j.find('_') + 1] + prefix + j[j.find('_') + 1:]
            tmp = tmp.append(tmp_copy, ignore_index=True)

        c_df = c_df.append(tmp, ignore_index=True)
        dataset = dataset.append(c_df, ignore_index=True)

    train, val, test = [], [], []

    originals = 1000
    whole = 20000

    for i in range(len(classes)):
        for j in range(20):
            numbers = np.random.choice(originals, originals, replace=False)
            train.append(list(np.sort(numbers[:int(len(numbers) * 0.7)]) + i * whole + j * originals))
            val.append(list(np.sort(numbers[int(len(numbers) * 0.7):int(len(numbers) * 0.85)]) + i * whole + j * originals))
            test.append(list(np.sort(numbers[int(len(numbers) * 0.85):]) + i * whole + j * originals))

    train = [item for sublist in train for item in sublist]
    val = [item for sublist in val for item in sublist]
    test = [item for sublist in test for item in sublist]

    for i in range(len(dataset)):
        anotace = dataset.iloc[i].tolist()
        if i in train:
            traindf.loc[i] = anotace
            continue
        if i in val:
            valdf.loc[i] = anotace
            continue
        if i in test:
            testdf.loc[i] = anotace
            continue

    traindf.reset_index(drop=True)
    valdf.reset_index(drop=True)
    testdf.reset_index(drop=True)

    traindf.to_csv(os.path.join(end_path, "train.csv"), index=False)
    valdf.to_csv(os.path.join(end_path, "val.csv"), index=False)
    testdf.to_csv(os.path.join(end_path, "test.csv"), index=False)


# rename_resize(source_path="source/images", end_path="dataset_light/images", new_resolution=(600, 400), max_images=1000)
augment(path="dataset_light/images")
# anotate(source_path="source/anotations", end_path="dataset_light", train_size=98, val_size=21, test_size=21)
