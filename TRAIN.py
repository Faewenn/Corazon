from torch.utils.data import DataLoader
from NMS import nms
from DATASET import ImageLoader
from TARGET_CONVERSION import targets_to_boxes
from MAP import mAP
from matplotlib import pyplot as plt
import numpy as np
import torch
import tqdm
import YoloV3
import os
from SHOW_IMAGE import show_image


def train(classes, config):
    torch.backends.cudnn.benchmark = True
    history = {'train_losses': [], 'val_maps': [], 'val_losses': []}

    # Přidání dalších konfiguračních konstant
    config.append("cuda" if torch.cuda.is_available() else "cpu")
    config.append([[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # 116, 90, 156, 198, 373, 326 / 416
                   [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # 30, 61, 62, 45,  59, 119 / 416
                   [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]])  # 10, 13, 16, 30, 33, 23 / 416
    config.append([config[5] // 32, config[5] // 16, config[5] // 8])  # [10, 20, 40] pro 320
    config.append(torch.from_numpy(np.float32(np.array(config[21]) * np.array(config[22])[:, None, None]))
                  .to(config[20]))

    # Načtení modelu a optimizeru
    model = YoloV3.Architecture(classes).to(config[20])
    optimizer = torch.optim.Adam(model.parameters(), lr=config[0], weight_decay=config[1])
    loss_fn = YoloV3.LossFunction()
    scaler = torch.cuda.amp.GradScaler()  # https://pytorch.org/docs/stable/amp.html#gradient-scaling
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint['history']
            try:
                print("\nSuccesfully loaded model with train_loss:", history['train_losses'][-1], "val_loss:",
                      history['val_losses'][-1], "val_mAP:", history['val_maps'][-1])
            except:
                print("\nSuccesfully loaded model with train_loss:", history['train_losses'][-1])
        else:
            print("Checkpoint file not found, continuing without loaded model")

    train_dataset = ImageLoader(config[11] + "/train.csv", classes, config)
    train_loader = DataLoader(train_dataset, config[2], True, pin_memory=config[9], num_workers=config[10])
    print("\nTrénovací dataset načten :-)")

    validation_dataset = ImageLoader(config[11] + "/val.csv", classes, config)
    validation_loader = DataLoader(validation_dataset, 1, config[17], pin_memory=config[9], num_workers=config[10])
    print("Validační dataset načten :-)")

    # Načtení testovacího datasetu
    test_dataset = ImageLoader(config[11] + "/test.csv", classes, config)
    test_loader = DataLoader(test_dataset, 1, config[17], pin_memory=config[9], num_workers=config[10])
    print("Testovací dataset načten :-)")

    if config[14]:
        for epoch in range(config[4]):
            print('\n' + 28 * "#")
            print("# Trénovací epocha", epoch, "z", config[4], " #")
            print(28 * "#")
            model, history = play(model, optimizer, loss_fn, scaler, config, classes, train_loader, history, 'train')

            if epoch % config[3] == 0:
                print('\n' + 28 * "#")
                print("#      Validační fáze      #")
                print(28 * "#")
                model, history = play(model, optimizer, loss_fn, scaler, config, classes, validation_loader, history, 'val')

    print('\n' + 28 * "#")
    print("#      Testovací fáze      #")
    print(28 * "#")
    model, history = play(model, optimizer, loss_fn, scaler, config, classes, test_loader, history, 'test')



def play(model, optimizer, loss_fn, scaler, config, classes, dataset, history, function='train'):
    if function == 'train':
        model.train()
    else:
        model.eval()
        all_true_boxes, all_pred_boxes = [], []

    # Učící proces
    p_bar = tqdm.tqdm(dataset, leave=True)

    losses = []
    for x, y in p_bar:
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        if function == 'train':
            optimizer.zero_grad()

        x, y0, y1, y2 = x.to(config[20]).float(), y[0].to(config[20]).float(), \
                        y[1].to(config[20]).float(), y[2].to(config[20]).float()

        # Zjistit výsledky
        use = torch.cuda.amp.autocast() if function == 'train' else torch.no_grad()
        with use:
            out = model(x)
            loss = (loss_fn(out[0], y0, config[23][0]) + loss_fn(out[1], y1, config[23][1])
                    + loss_fn(out[2], y2, config[23][2]))
            losses.append(loss.item())

        # Učení - https://pytorch.org/docs/stable/notes/amp_examples.html Typical Mixed Precision Training
        if function == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            true_boxes = targets_to_boxes(y, config[21])
            pred_boxes = targets_to_boxes(out, config[21])

            all_true_boxes.append(nms(true_boxes, config[7]))
            all_pred_boxes.append(nms(pred_boxes, config[7]))

            # Ukázat každý obrázek
            if function == 'test' and config[16]:
                show_image(x.to('cpu'), all_true_boxes[-1], all_pred_boxes[-1], classes)

        p_bar.set_postfix(loss=sum(losses) / len(losses))

    if function == 'train':
        history['train_losses'].append(sum(losses) / len(losses))
    else:
        meanAP = mAP(all_true_boxes, all_pred_boxes, classes, config[6])
        print('Calculated mAP:', meanAP)

        if function == 'val':
            history['val_losses'].append(sum(losses) / len(losses))
            history['val_maps'].append(meanAP)

    if function == 'val':
        path = "checkpoints/" + config[18] + "_" + str(sum(losses) / len(losses)) + ".pth.tar"
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history},
                   path)
        print("\nModel succesfully saved with loss:", sum(losses) / len(losses), "and mAP:", meanAP)

    elif function == 'test':
        fig, axs = plt.subplots(2)
        axs[0].plot(history['train_losses'], 'r-x', label="Train losses")
        axs[0].plot(history['val_losses'], 'b-x', label="Validation losses")
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(history['val_maps'], 'g-x', label="Validation mAPs")
        axs[1].grid()
        axs[1].legend()
        plt.show()

    return model, history

