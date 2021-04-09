from torch.utils.data import DataLoader
from NMS import nms
from DATASET import ImageLoader
from TARGET_CONVERSION import targets_to_boxes
from MAP import mAP
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import tqdm
import YoloV3
import os
from SHOW_IMAGE import show_image


def train(classes, config):
    torch.backends.cudnn.benchmark = True
    val_maps = []
    val_losses = []
    train_losses = []

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
        for file in os.listdir("checkpoints"):
            if file[-8:] == '.pth.tar' and file[:file.find('_')] == config[18]:
                config[15] = os.path.join("checkpoints", file)
        if isinstance(config[15], str):
            checkpoint = torch.load(config[15])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            val_maps.append(value for value in checkpoint['val_maps'])
            val_losses.append(value for value in checkpoint['val_losses'])
            train_losses.append(value for value in checkpoint['train_losses'])
            print("\nSuccesfully loaded model with val_loss:", val_losses[-1], "val_map:", val_maps[-1], "train_loss:", train_losses[-1])
        else:
            print("Checkpoint file not found, continuing without loaded model")

    # Trénink
    if config[14]:
        # Načtení train a validačního dataset
        train_dataset = ImageLoader(config[11] + "/train.csv", classes, config)
        train_loader = DataLoader(train_dataset, config[2], True, pin_memory=config[9], num_workers=config[10])
        print("Trénovací dataset načten :-)")

        validation_dataset = ImageLoader(config[11] + "/test.csv", classes, config)
        validation_loader = DataLoader(validation_dataset, config[2], config[17], pin_memory=config[9], num_workers=config[10])
        print("Validační dataset načten :-)")

        for epoch in range(config[4]):
            print("\nProbíhá epocha", epoch, "z", config[4])
            model.train()

            # Učící proces
            p_bar = tqdm.tqdm(train_loader, leave=True)

            losses = []
            for batch_idx, (x, y) in enumerate(p_bar):
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad()

                x, y0, y1, y2 = x.to(config[20]).float(), y[0].to(config[20]).float(), \
                                y[1].to(config[20]).float(), y[2].to(config[20]).float()

                """
                print("TRAIN DATA IN:")
                print("Image:", x.shape, "targets:", y0.shape, y1.shape, y2.shape)
                """

                # Zjistit výsledky
                with torch.cuda.amp.autocast():
                    out = model(x)

                    """
                    print("YoloV3 OUT:")
                    print("Targets:", out[0].shape, out[1].shape, out[2].shape)
    
                    print("LOSS IN:")
                    print("Predicted targets:", out[0].shape, out[1].shape, out[2].shape)
                    print("True targets:     ", y0.shape, y1.shape, y2.shape)
                    """

                    loss = (loss_fn(out[0], y0, config[23][0]) + loss_fn(out[1], y1, config[23][1])
                            + loss_fn(out[2], y2, config[23][2]))
                    losses.append(loss.item())

                    """
                    print("LOSS OUT:")
                    print("Loss =", loss)
                    """

                # Učení - https://pytorch.org/docs/stable/notes/amp_examples.html Typical Mixed Precision Training
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                p_bar.set_postfix(loss=sum(losses) / len(losses))
	    
            train_losses.append(sum(losses) / len(losses))

            # Validace (+ mAP)
            if epoch % config[3] == 0:
                model.eval()

                v_p_bar = tqdm.tqdm(validation_loader, leave=True)
                v_all_true_boxes = []
                v_all_pred_boxes = []
                v_losses = []

                for v_batch_idx, (v_x, v_y) in enumerate(v_p_bar):
                    """
                    print("TEST DATA IN:")
                    print("Image:", t_x.shape, "targets:", t_y[0].shape, t_y[1].shape, t_y[2].shape)
                    """

                    with torch.no_grad():
                        pred = model(v_x.to(config[20]).float())
                        loss = (loss_fn(pred[0], y0, config[23][0]) + loss_fn(pred[1], y1, config[23][1])
                            + loss_fn(pred[2], y2, config[23][2])).item()
                        v_losses.append(loss)

                    """
                    print("TEST TARGETS:")
                    print("Predicted targets:", pred[0].shape, pred[1].shape, pred[2].shape)
                    print("True targets:     ", t_y[0].shape, t_y[1].shape, t_y[2].shape)
                    """

                    true_boxes = targets_to_boxes(v_y, config[21])
                    pred_boxes = targets_to_boxes(pred, config[21])

                    """
                    print("TEST BOXES:")
                    print("True boxes:", true_boxes[0].shape, true_boxes[1].shape, true_boxes[2].shape)
                    print("Predicted boxes", pred_boxes[0].shape, pred_boxes[1].shape, pred_boxes[2].shape)
                    """

                    v_all_true_boxes.append(nms(true_boxes, thresh=config[7]))
                    v_all_pred_boxes.append(nms(pred_boxes, thresh=config[7]))


                val_maps.append(mAP(v_all_true_boxes, v_all_pred_boxes, classes, thresholds=config[6]))
                val_losses.append(sum(v_losses) / len(v_losses))

            path = "checkpoints/" + "loss" + "_" + config[18] + "_" + str(sum(losses) / len(losses)) + ".pth.tar"
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_maps': val_maps,
			'val_losses': val_losses,
			'train_losses': train_losses},
                        path)
            print("\nModel succesfully saved with loss:", sum(losses) / len(losses), "val loss:", val_losses[-1], "and mAP:", val_maps[-1])

    # Ukázat validační mAP v grafu
    if config[16] and len(val_results) > 1:
        plt.plot(val_results)
        plt.show()

    model.eval()

    # Načtení testovacího datasetu
    test_dataset = ImageLoader(config[11] + "/test.csv", classes, config)
    test_loader = DataLoader(test_dataset, 1, config[17], pin_memory=config[9], num_workers=config[10])
    print("Testovací dataset načten :-)")

    t_p_bar = tqdm.tqdm(test_loader, leave=True)
    t_all_true_boxes = []
    t_all_pred_boxes = []

    for t_batch_idx, (t_x, t_y) in enumerate(t_p_bar):
        with torch.no_grad():
            pred = model(t_x.to(config[20]).float())

        true_boxes = targets_to_boxes(t_y, config[21])
        pred_boxes = targets_to_boxes(pred, config[21])

        nmsed_true_boxes = nms(true_boxes, thresh=config[7])
        nmsed_pred_boxes = nms(pred_boxes, thresh=config[7])

        # Ukázat každý obrázek
        if config[16]:
            show_image(t_x.to('cpu'), nmsed_true_boxes, nmsed_pred_boxes, classes)

        t_all_true_boxes.append(nmsed_true_boxes)
        t_all_pred_boxes.append(nmsed_pred_boxes)

    result_map = mAP(t_all_true_boxes, t_all_pred_boxes, classes, thresholds=config[6])

    print("Testovací dataset mAP:", result_map)



