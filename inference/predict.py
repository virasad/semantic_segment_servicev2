import os
from datetime import datetime

import albumentations as A
import cv2
# import matplotlib
import numpy as np
import torch
from preimutils.segmentations.voc.utils import utils as preutils

from model import model
from utils import postprocess


# matplotlib.use('TkAgg')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing_predict(preprocessing_fn, h, w):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.augmentations.geometric.resize.Resize(h, w),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)


# from flash.image
class InferenceSeg:
    def __init__(self, encoder, encoder_weights, model_path=None):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print('Using Device: ' + str(self.device))
        self.model_path = model_path
        self.preprocessing_fn = model.prep(encoder, encoder_weights)
        self.weight, self.height = 0, 0
        if model_path is not None:
            self.set_model(model_path)

    def set_model(self, model_path):
        self.best_model = torch.load(model_path, map_location=self.device)
        self.best_model.to(self.device)
        self.best_model.eval()
        self.weight, self.height = self.best_model.width, self.best_model.height
        self.preprocessing = get_preprocessing_predict(self.preprocessing_fn, h=self.height, w=self.weight)
        self.n_classes = self.best_model.n_classes
        self.colors = self.generate_random_colors(self.n_classes)
        print('Model loaded from {}'.format(model_path))
        print('Model size w {} h {}'.format(self.best_model.width, self.best_model.height))
        print('Model has {} classes'.format(self.n_classes))

    def predict(self, image):
        x_tensor = torch.from_numpy(image['image']).to(
            self.device).unsqueeze(0)
        pr_mask = self.best_model.predict(x_tensor)
        if self.n_classes == 1:
            return pr_mask.sigmoid().squeeze().cpu()
        return pr_mask.squeeze().cpu()

    def predict_data(self, image_p, return_image=False, return_coco=False, save=False, output_dir=None, image_id=0,
                     starting_ann_id=0, task_id=0):
        """
        Predict the segmentation of an image using the model.
        :param image_p: cv2 image or path to image
        :param return_image: if True, return the image
        :param output_dir: directory to save the image
        :return: cv2 mask of the segmentation
        """
        if isinstance(image_p, str):
            # check if the image is a path
            if not os.path.isfile(image_p):
                raise ValueError('Image path does not exist')
            image = cv2.imread(image_p)
        elif isinstance(image_p, np.ndarray):
            image = image_p
        else:
            raise ValueError('Image must be a path or numpy array')
        w, h = image.shape[1], image.shape[0]
        # resize ratio
        w_ratio, h_ratio = w / self.weight, h / self.height
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_prep = self.preprocessing(image=image)
        mask = self.predict(image_prep)
        if self.n_classes == 1:
            # where the mask is 0.5 or higher, set it to 1
            mask[mask >= 0.5] = 1
            # where the mask is 0.5 or lower, set it to 0
            mask[mask < 0.5] = 0
            # print(mask)
            labels = mask
        else:
            labels = torch.argmax(mask, dim=-3)  # HxW

        if return_image:
            labels = labels.cpu().numpy()
            rgb_label = preutils.decode_segmap(labels, self.colors, False)
            alpha = 0.7
            # image = cv2.resize(image, (rgb_label.shape[1], rgb_label.shape[0]))
            image = cv2.resize(image, (self.weight, self.height))
            overlay = cv2.addWeighted(image, alpha, rgb_label, 1 - alpha, 0, rgb_label)
            _, encoded_img = cv2.imencode('.JPG', overlay)
            return encoded_img

        elif return_coco:
            labels = labels.cpu().numpy()
            res = postprocess.get_segmentation_dict(segmentation_mask=labels,
                                                    n_classes=self.n_classes,
                                                    img_id=image_id,
                                                    starting_annotation_indx=starting_ann_id,
                                                    task_id=task_id,
                                                    w_ratio=w_ratio,
                                                    h_ratio=h_ratio)
            return res

        elif save:
            labels = labels.cpu().numpy()
            rgb_label = preutils.decode_segmap(labels, self.colors, False)
            alpha = 0.7
            image = cv2.resize(image, (self.weight, self.height))
            overlay = cv2.addWeighted(image, alpha, rgb_label, 1 - alpha, 0, rgb_label)
            save_dst = os.path.join(output_dir, 'result_{}.png'.format(datetime.now().strftime("%Y%m%d_%H%M%S%f")))
            cv2.imwrite(save_dst, overlay)
        else:
            labels = labels.cpu().numpy()
            return labels

    @staticmethod
    def generate_random_colors(n):
        if n == 1:
            return [(255, 0, 0)]
        colors = []
        colors.append((0, 0, 0))
        for i in range(1, n):
            colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        return colors


if __name__ == '__main__':
    import click
    import glob


    @click.command()
    @click.option('--images_dir', default='/home/james/Documents/data/images', help='Path to images directory')
    @click.option('--model_path', default='/home/james/Documents/data/best_model.pth', help='Path to model')
    @click.option('--save', default=False, is_flag=True, help='Save output')
    @click.option('--output_dir', default='/home/james/Documents/data/output', help='Path to output directory')
    def main(images_dir, model_path, save, output_dir):
        inference = InferenceSeg('mobilenet_v2', 'imagenet', model_path)
        inference.set_model(model_path)
        for image_path in glob.glob(os.path.join(images_dir, '*')):
            a = inference.predict_data(image_path, save=save, output_dir=output_dir, return_coco=True)
            if len(a):
                print(a)

    main()
