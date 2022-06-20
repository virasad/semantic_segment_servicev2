import torch
from tqdm import tqdm
import time

import numpy as np
from model import model as model_utils
from utils import dataset_utils as ds_utils, dataloaders, metrics, mobilenetv2_pre
from pathlib import Path


def fit(device, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += metrics.mIoU(output, mask, n_classes=23)
            accuracy += metrics.pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(model_utils.get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += metrics.mIoU(output, mask, n_classes=2)
                    test_accuracy += metrics.pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))

            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(
                    min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                if decrease % 3 == 0:
                    print('saving model...')
                    torch.save(
                        model, 'Deeplabv3Plus-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 20:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(
                      running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time() - fit_time)/60))
    return history


def trainer():
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['background', 'burr', 'hole_end']
    ACTIVATION = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using Device: ' + str(DEVICE))
    w_size = 1536
    h_size = 2046
    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4
    our_model = model_utils.create_model(
        ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        our_model.parameters(), lr=max_lr, weight_decay=weight_decay)

    dataset_p = Path('dataset')
    images_dir = dataset_p / 'images'
    masks_dir = dataset_p / 'pre_encoded'
    train_images, train_masks, test_images, test_masks = ds_utils.split_train_test(
        images_dir, masks_dir, 0.2)

    preprocessing_fn = mobilenetv2_pre.get_preprocessing()
    train_aug = dataloaders.get_training_augmentation(w_size, h_size)
    val_aug = dataloaders.get_validation_augmentation(w_size, h_size)
    train_dataset = dataloaders.Dataset(
        train_images,
        train_masks, 
        preprocessing=preprocessing_fn,
        augmentation=train_aug
        )

    test_dataset = dataloaders.Dataset(
        test_images, 
        test_masks, 
        preprocessing=preprocessing_fn, 
        augmentation=val_aug
        )

    train_dataloader = dataloaders.create_dataloader(
        train_dataset, batch_size=2, num_workers=8, shuffle=True)

    test_dataloader = dataloaders.create_dataloader(
        test_dataset, batch_size=2, num_workers=8, shuffle=False)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dataloader))

    fit(DEVICE, epoch, our_model, train_dataloader,
        test_dataloader, criterion, optimizer, sched, patch=False)

if __name__ == '__main__':
    trainer()