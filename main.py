import train3
import segmentation_models_pytorch as smp
import torch

# python3 main.py --preprocess --augment-count=20 --dataset=dataset --image-extention=tif --val-test-split True 0.1 0.1
# python3 main.py --dataset=dataset --train --batch-size=8 --epochs=20 --image-extention=tif
# python3 main.py --test --dataset=full_dataset --image-extention=tif --model-path=resnet50-4000-1080.pth
# python3 main.py --predict --image-path=sample/JPEGImages/train/2021年10月19日13時09分22秒971_result.tif

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
ACTIVATION = 'sigmoid'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using Device: ' + str(DEVICE))
SIZE = 400
LOSS = smp.utils.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
METRICS = [
    smp.utils.metrics.IoU()
]


def trrrr(images_dir,masks_dir, epochs, batch_size):

    train3.train_data(images_dir, masks_dir, ENCODER, ENCODER_WEIGHTS,
                     CLASSES, ACTIVATION, LOSS, METRICS, epochs, DEVICE, batch_size, SIZE)


if __name__ == '__main__':
    from pathlib import Path
    dataset_p = Path('carla-capture-20180513A')
    images_dir = dataset_p / 'images'
    masks_dir = dataset_p / 'masks'
    trrrr(images_dir, masks_dir, 20, 2)
