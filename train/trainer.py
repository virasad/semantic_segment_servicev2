import os
import os.path
import shutil
import time
import uuid

import click
import numpy as np
import torch
from tqdm import tqdm

from model import model as model_utils
from utils import callbacks as cb
from utils import dataset_utils as ds_utils, dataloaders, mobilenetv2_pre, metrics as mtr

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def fit(device, n_classes, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, model_name,
        model_dir, callbacks, patch=False):
    def write_callbacks(callbacks, metrics):
        for callback in callbacks:
            callback.write(metrics)

    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_ious = []
    val_accs = []
    train_ious = []
    train_accs = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0
    best_metrics = {}
    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        metrics = {}
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
            iou_score += mtr.mIoU(output, mask, n_classes=n_classes)
            accuracy += mtr.pixel_accuracy(output, mask)
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
                    val_iou_score += mtr.mIoU(output, mask, n_classes=n_classes)
                    test_accuracy += mtr.pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch

            train_loss = running_loss / len(train_loader)
            val_loss = test_loss / len(val_loader)
            train_miou = iou_score / len(train_loader)
            val_miou = val_iou_score / len(val_loader)
            train_acc = accuracy / len(train_loader)
            val_acc = test_accuracy / len(val_loader)
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_miou': train_miou,
                'val_miou': val_miou,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'epoch': e + 1,
                'total_epochs': epochs,
                'state': 'train'
            }
            val_ious.append(val_miou)
            train_ious.append(train_miou)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_losses.append(train_loss)
            test_losses.append(val_loss)
            write_callbacks(callbacks, metrics)
            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(
                    min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 3 == 0:
                    print('saving model...')
                    # model_name = '{}_Deeplabv3Plus-Mobilenet_v2_mIoU-{:.3f}.pt'.format(model_name,
                    #                                                                    val_iou_score / len(val_loader))

                    model_name = '{}_Deeplabv3Plus-Mobilenet_v2_mIoU.pt'.format(model_name)
                    model_p = os.path.join(model_dir, model_name)
                    torch.save(
                        model, model_p)
                    metrics['state'] = 'best'
                    metrics['model_path'] = model_p
                    print(model_dir, model_p)
                    write_callbacks(callbacks, metrics)
                    best_metrics = metrics

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 10:
                    print('Loss not decrease for 10 times, Stop Training')
                    break

            # iou
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(
                      running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses,
               'val_loss': test_losses,
               'train_miou': train_ious,
               'val_miou': val_ious,
               'train_acc': train_accs,
               'val_acc': val_accs,
               'lrs': lrs}

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history, best_metrics, model_p


def trainer(images_dir, masks_dir, n_classes=19, w_size=1024, h_size=1024, batch_size=5, epochs=100, response_url=None,
            log_url=None, model_name='', task_id=None, data_type='coco', redis_cb=True, file_cb=True, ):
    """
    TODO: add description
    :param images_dir:
    :param masks_dir:
    :param n_classes:
    :param w_size:
    :param h_size:
    :param batch_size:
    :param epochs:
    :param response_url:
    :param log_url:
    :param model_name:
    :param task_id:
    :param data_type:
    :return:
    """

    if data_type == 'coco':
        from utils import dataset

        if not os.path.exists(images_dir):
            raise FileExistsError("images path not found")
        if not os.path.exists(masks_dir):
            raise FileExistsError("json annotation path not found")

        raw_mask = os.path.join(os.path.dirname(images_dir), 'raw_mask')

        try:
            os.mkdir(raw_mask)
        except FileExistsError:
            shutil.rmtree(raw_mask)
            os.mkdir(raw_mask)

        dataset.CocoHandler(masks_dir,
                            images_dir).convert_dataset_to_masks(raw_mask)
        masks_dir = raw_mask

    # TODO: add mask_raw and voc converter
    elif data_type == 'mask_raw':
        pass

    elif data_type == 'voc':
        pass

    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using Device: ' + str(DEVICE))
    max_lr = 1e-3
    weight_decay = 1e-4
    our_model = model_utils.create_model(
        encoder=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        num_classes=n_classes,
        activation=ACTIVATION
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        our_model.parameters(), lr=max_lr, weight_decay=weight_decay)

    train_images, train_masks, test_images, test_masks = ds_utils.split_train_test(
        images_dir, masks_dir, 0.2)

    preprocessing_fn = mobilenetv2_pre.get_preprocessing()
    train_aug = dataloaders.get_training_augmentation(w=w_size, h=h_size)
    val_aug = dataloaders.get_validation_augmentation(w=w_size, h=h_size)
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
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    test_dataloader = dataloaders.create_dataloader(
        test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_dataloader))
    if not task_id:
        task_id = str(uuid.uuid4())

    task_dir = os.path.join(ROOT_DIR, 'runs', task_id)
    log_dir = os.path.join(task_dir, 'logs')
    model_dir = os.path.join(task_dir, 'models')
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    cbs = []
    if file_cb:
        file_cb = cb.FileCallback(log_dir)
        cbs.append(file_cb)
    if response_url:
        response_cb = cb.PostCallback(log_url)
        cbs.append(response_cb)

    if redis_cb:
        redis_cb = cb.RedisCallback(task_id)
        cbs.append(redis_cb)

    history, best_metrics, best_p = fit(device=DEVICE,
                                        n_classes=n_classes,
                                        epochs=epochs,
                                        model=our_model,
                                        train_loader=train_dataloader,
                                        val_loader=test_dataloader,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scheduler=sched,
                                        patch=False,
                                        model_name=model_name,
                                        callbacks=cbs,
                                        model_dir=model_dir)

    if response_url:
        response_cb = cb.PostCallback(response_url)
        best_metrics['weights_p'] = best_p
        best_metrics['task_id'] = task_id
        response_cb.write(best_metrics)
    return history


if __name__ == '__main__':
    @click.command()
    @click.option('--images_dir', '-i', default='../dataset2/images', help='Path to images directory')
    @click.option('--masks_dir', '-m', default='../dataset2/pre_encoded', help='Path to masks directory')
    @click.option('--model_name', '-n', default='', help='Name of the model')
    @click.option('--n_classes', '-n', default=19, help='Number of classes')
    @click.option('--w_size', '-w', default=1024, help='Width of image')
    @click.option('--h_size', '-h', default=1024, help='Height of image')
    @click.option('--batch_size', '-bs', default=5, help='Batch size')
    @click.option('--epochs', '-e', default=100, help='Number of epochs')
    @click.option('--response_url', '-r', default=None, help='URL to send response')
    @click.option('--log_url', '-l', default=None, help='URL to send logs')
    @click.option('--task_id', '-t', default=None, help='Task ID')
    @click.option('--data_type', '-d', default='coco', help='Data type')
    def main(images_dir, masks_dir, model_name, n_classes, w_size, h_size, batch_size, epochs, response_url, log_url,
             task_id, data_type):
        trainer(images_dir=images_dir,
                masks_dir=masks_dir,
                model_name=model_name,
                n_classes=n_classes,
                w_size=w_size,
                h_size=h_size,
                batch_size=batch_size,
                epochs=epochs,
                response_url=response_url,
                log_url=log_url,
                task_id=task_id,
                data_type=data_type)


    main()
