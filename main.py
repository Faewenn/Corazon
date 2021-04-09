import TRAIN
import torch


if __name__ == '__main__':

    # Seznam tříd
    classes = ['Gumicka', 'Kolicek', 'Paratko', 'Pripinacek', 'Sirka', 'Sponka', 'Vlajecka']

    # Obecná konfigurace
    config = [0.001,  # 0 Learning rate	
              0.0005,  # 1 Weight decay
              16,  # 2 Batch size
              8,  # 3 Validace každých X epoch
              100,  # 4 Epochs
              320,  # 5 Velikost vstupu (320 pro 320x320, 416 pro 416x416, 608 pro 608x608)
              [0.5],  # 6 mAP thresholds
              0.5,  # 7 NMS Threshold
              0.5,  # 8 Boxes to targets Threshold
              True,  # 9 Pin memory
              0,  # 10 Num workers
              'dataset',  # 11 Složka s datasetem
              'dataset/images',  # 12 Složka s obrázky
              "center",  # 13 Souřadniceový systém (center / corners)
              True,  # 14 Train?
              True,  # 15 Load checkpoint?
              False,  # 16 Ukázat testovací obrázky?
              True,  # 17 Zamíchat validační a testovací dataset?
              "BC",  # 18 Jméno
              1]  # 19 not used

    TRAIN.train(classes, config)







